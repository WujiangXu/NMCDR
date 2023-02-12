import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
# from thop import profile
# from thop import clever_format

class embUserLayerEnhance(nn.Module):
    def __init__(self,user_length,emb_dim):
        super(embUserLayerEnhance, self).__init__()
        self.emb_user_share = nn.Embedding(user_length,emb_dim)
        self.transd1 = nn.Linear(emb_dim,emb_dim)
        self.transd2 = nn.Linear(emb_dim,emb_dim)

    def forward(self, user_id):
        user_nomarl = self.emb_user_share(user_id)
        user_spf1 = self.transd1(user_nomarl)
        user_spf2 = self.transd2(user_nomarl)
        return user_spf1, user_spf2#, user_nomarl

class embItemLayerEnhance(nn.Module):
    def __init__(self,item_length,emb_dim):
        super(embItemLayerEnhance, self).__init__()
        self.emb_item = nn.Embedding(item_length,emb_dim)

    def forward(self,item_id):
        item_f = self.emb_item(item_id)
        return item_f

class SpecificUIGraphLayer(nn.Module):
    def __init__(self, emb_dim):
        super(SpecificUIGraphLayer, self).__init__()
        self.i_to_u = nn.Linear(emb_dim,emb_dim)

    def forward(self, user_feat_d1, seq_d):
        seq_mess = torch.mean(self.i_to_u(seq_d),dim=1)
        user_feat_d1 = user_feat_d1 + seq_mess
        return user_feat_d1

class SpecificUIGrapModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(SpecificUIGrapModule, self).__init__()
        self.gat_module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.gat_module.append(SpecificUIGraphLayer(emb_dim))
    
    def forward(self, user_feat, seq_feat):
        for i in range(self.layer_nums): 
            user_feat = self.relu(self.gat_module[i](user_feat, seq_feat))
        return user_feat

class GateFuseCell(nn.Module):
    def __init__(self, emb_dim):
        super(GateFuseCell, self).__init__()
        self.trans1 = nn.Linear(emb_dim,emb_dim)
        self.trans2 = nn.Linear(emb_dim,emb_dim)
        self.act = nn.ReLU()

    def forward(self, long_feat, tail_feat): # user_s = shared
        gate_att = torch.sigmoid(self.trans1(long_feat)+self.trans2(tail_feat))
        long_tail_fuse = gate_att * long_feat + (1-gate_att) * tail_feat
        return long_tail_fuse#torch.mean(long_tail_fuse,dim=1).squeeze()

class InnerGraphMatchingLayer(nn.Module):
    def __init__(self, emb_dim):
        super(InnerGraphMatchingLayer, self).__init__()
        self.long_trans = nn.Linear(emb_dim,emb_dim)
        self.tail_trans = nn.Linear(emb_dim,emb_dim)
        self.relu = nn.ReLU() # need to test
        self.gru_fuse = GateFuseCell(emb_dim)

    def forward(self, user_feat, long_tail_mask):
        att = torch.matmul(user_feat,user_feat.T) #[bs,bs]
        att = att-torch.diag_embed(torch.diag(att)) # self-att = 0 [bs,bs]
        att = F.normalize(att,p=1,dim=1)
        message = att.unsqueeze(-1) * user_feat.unsqueeze(0) 
        # [bs,bs,1] * [1,bs,emb_dim] -->broadcast : [bs,bs,emb_dim]

        long_message = message * long_tail_mask.unsqueeze(0).unsqueeze(-1)
        long_message = self.long_trans(long_message)
        tail_message = message * (1-long_tail_mask.unsqueeze(0).unsqueeze(-1))
        tail_message = self.tail_trans(tail_message)
        # [bs,bs,emb_dim]
        message_fuse = torch.mean(self.gru_fuse(long_message,tail_message),dim=1).squeeze()    #[bs,1,emb_dim]
        return user_feat + message_fuse

class InnerGraphMatchingModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(InnerGraphMatchingModule, self).__init__()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.module.append(InnerGraphMatchingLayer(emb_dim))
    
    def forward(self, user_feat, long_tail_mask):
        for i in range(self.layer_nums): 
            user_feat = self.relu(self.module[i](user_feat, long_tail_mask))
        return user_feat

class InterGraphMatchingLayer(nn.Module):
    def __init__(self, emb_dim):
        super(InterGraphMatchingLayer, self).__init__()
        self.trans_d1 = nn.Linear(emb_dim,emb_dim)
        self.trans_d2 = nn.Linear(emb_dim,emb_dim)
        self.self_d1 = nn.Linear(emb_dim,emb_dim)
        self.self_d2 = nn.Linear(emb_dim,emb_dim)
        self.relu = nn.ReLU() # need to test
        self.gru_fuse1 = GateFuseCell(emb_dim)
        self.gru_fuse2 = GateFuseCell(emb_dim)

    def forward(self, user_feat_d1, user_feat_d2):
        # input : [bs , emb_dim]

        # pass message d2 to d1
        att_1 = torch.matmul(user_feat_d1,user_feat_d2.T) #[bs,bs]
        att_1 = att_1-torch.diag_embed(torch.diag(att_1)) # self-att = 0 [bs,bs]
        att_1 = F.normalize(att_1,p=1,dim=1)
        message_d2_d1 = torch.mean(self.trans_d1(att_1.unsqueeze(-1) * user_feat_d2.unsqueeze(0)),dim=1).squeeze() # [N,N,D]
        message_self = self.self_d1(user_feat_d2)
        message_sum = self.gru_fuse1(message_d2_d1,message_self)
        user_feat_d1 = user_feat_d1 + message_sum

        # pass message d1 to d2
        att_2 = torch.matmul(user_feat_d2,user_feat_d1.T) #[bs,bs]
        att_2 = att_2-torch.diag_embed(torch.diag(att_2)) # self-att = 0 [bs,bs]
        att_2 = F.normalize(att_2,p=1,dim=1)
        message_d1_d2 = torch.mean(self.trans_d2(att_2.unsqueeze(-1) * user_feat_d1.unsqueeze(0)),dim=1).squeeze() # [N,N,D]
        message_self = self.self_d2(user_feat_d1)
        message_sum = self.gru_fuse2(message_d1_d2,message_self)
        user_feat_d2 = user_feat_d2 + message_sum
        return user_feat_d1, user_feat_d2

class InterGraphMatchingModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(InterGraphMatchingModule, self).__init__()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.module.append(InterGraphMatchingLayer(emb_dim))
    
    def forward(self, user_feat_d1, user_feat_d2):
        for i in range(self.layer_nums): 
            user_feat_d1, user_feat_d2 = self.module[i](user_feat_d1, user_feat_d2)
            user_feat_d1 = self.relu(user_feat_d1)
            user_feat_d2 = self.relu(user_feat_d2)
        return user_feat_d1, user_feat_d2

class RefineUIGraphLayer(nn.Module):
    def __init__(self, emb_dim):
        super(RefineUIGraphLayer, self).__init__()
        self.trans = nn.Linear(emb_dim,emb_dim)
    
    def forward(self, user_feat, seq_feat):
        seq_feat = seq_feat.reshape(-1,seq_feat.shape[-1]) # [N*Seq_len,D]
        att = torch.matmul(user_feat,seq_feat.T) #[bs,bs]
        att = F.normalize(att,p=1,dim=1)
        message = self.trans(att.unsqueeze(-1) * seq_feat.unsqueeze(0))
        user_feat = user_feat + torch.mean(message,dim=1).squeeze()
        return user_feat

class RefineUIGraphModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(RefineUIGraphModule, self).__init__()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.module.append(RefineUIGraphLayer(emb_dim))
    
    def forward(self, user_feat, seq_feat):
        for i in range(self.layer_nums): 
            user_feat = self.relu(self.module[i](user_feat, seq_feat))
        return user_feat

class predictModule(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(predictModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,1))
    
    def forward(self, user_spf1, user_spf2, i_feat):
        '''
            user_spf : [bs,dim]
            i_feat : [bs,dim]
            neg_samples_feat: [bs,1/99,dim] 1 for train, 99 for test
        '''
        user_spf1 = user_spf1.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d1 = torch.cat((user_spf1,i_feat),-1)
        logits_d1 = torch.sigmoid(self.fc(user_item_concat_feat_d1))

        user_spf2 = user_spf2.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d2 = torch.cat((user_spf2,i_feat),-1)
        logits_d2 = torch.sigmoid(self.fc(user_item_concat_feat_d2))

        return logits_d1.squeeze(), logits_d2.squeeze()

class NGMCDR(nn.Module):

    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, m1_layers, m2_layers, m3_layers, m4_layers, hid_dim):
        super(NGMCDR, self).__init__()
        self.user_emb_layer = embUserLayerEnhance(user_length, user_emb_dim)
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)

        self.m1_layers = m1_layers
        self.m2_layers = m2_layers
        self.m3_layers = m3_layers
        self.m4_layers = m4_layers

        self.UIGraph_m1_d1 = SpecificUIGrapModule(user_emb_dim,m1_layers)#GATGraphModule(user_emb_dim,head_nums,layers)#nn.ModuleList()
        self.UIGraph_m1_d2 = SpecificUIGrapModule(user_emb_dim,m1_layers)#GATGraphModule(user_emb_dim,head_nums,layers)#nn.ModuleList()
        self.innerGM_m2_d1 = InnerGraphMatchingModule(user_emb_dim,m2_layers)
        self.innerGM_m2_d2 = InnerGraphMatchingModule(user_emb_dim,m2_layers)
        self.interGM_m3 = InterGraphMatchingModule(user_emb_dim,m3_layers)
        self.refine_m4_d1 = RefineUIGraphModule(user_emb_dim,m4_layers)
        self.refine_m4_d2 = RefineUIGraphModule(user_emb_dim,m4_layers)
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.w1 = nn.ParameterList([nn.Parameter(torch.rand(user_emb_dim, user_emb_dim), requires_grad=True)])
        self.w2 = nn.ParameterList([nn.Parameter(torch.rand(user_emb_dim, user_emb_dim), requires_grad=True)])
        # self.fc_aux = nn.Sequential(
        #                 nn.Linear(emb_dim*2,hid_dim),
        #                 nn.ReLU(),
        #                 nn.Linear(hid_dim,1))
        # self.predictModule_aux1 = predictModule(user_emb_dim,hid_dim)
        self.predictModule = predictModule(user_emb_dim,hid_dim)
        self.predictModule2 = predictModule(user_emb_dim,hid_dim)
        self.predictModule3 = predictModule(user_emb_dim,hid_dim)
        self.predictModule4 = predictModule(user_emb_dim,hid_dim)
        self.predictModule5 = predictModule(user_emb_dim,hid_dim)

    def forward(self,u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2,isTrain=True):
        user_spf1, user_spf2 = self.user_emb_layer(u_node)
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        seq_d1_feat = self.item_emb_layer(seq_d1)
        seq_d2_feat = self.item_emb_layer(seq_d2)

        u_feat_enhance_m1_d1 = self.UIGraph_m1_d1(user_spf1,seq_d1_feat)
        u_feat_enhance_m1_d2 = self.UIGraph_m1_d2(user_spf2,seq_d2_feat)

        u_feat_enhance_m2_d1 = self.innerGM_m2_d1(u_feat_enhance_m1_d1, long_tail_mask_d1)
        u_feat_enhance_m2_d2 = self.innerGM_m2_d2(u_feat_enhance_m1_d2, long_tail_mask_d2)

        u_feat_enhance_m3_d1,u_feat_enhance_m3_d2 = self.interGM_m3(u_feat_enhance_m2_d1,u_feat_enhance_m2_d2)

        u_feat_enhance_m4_d1 = self.refine_m4_d1(u_feat_enhance_m3_d1,seq_d1_feat)
        u_feat_enhance_m4_d2 = self.refine_m4_d2(u_feat_enhance_m3_d2,seq_d2_feat)
        # print(i_feat.shape,neg_samples_feat.shape)
        i_feat = torch.cat((i_feat,neg_samples_feat),1)     
        user_feat_d1 =  torch.matmul(u_feat_enhance_m4_d1,self.w1[0]) + torch.matmul(u_feat_enhance_m4_d2,1-self.w1[0])
        user_feat_d2 =  torch.matmul(u_feat_enhance_m4_d2,self.w2[0]) + torch.matmul(u_feat_enhance_m4_d1,1-self.w2[0]) 
        u_feat_enhance_m4_d1 = user_feat_d1
        u_feat_enhance_m4_d2 = user_feat_d2

        logits_d1, logits_d2 = self.predictModule(u_feat_enhance_m4_d1, u_feat_enhance_m4_d2, i_feat)

        if isTrain:
            logits_m3_d1, logits_m3_d2 = self.predictModule2(u_feat_enhance_m3_d1, u_feat_enhance_m3_d2, i_feat)
            logits_m2_d1, logits_m2_d2 = self.predictModule3(u_feat_enhance_m2_d1, u_feat_enhance_m2_d2, i_feat)
            logits_m1_d1, logits_m1_d2 = self.predictModule4(u_feat_enhance_m1_d1, u_feat_enhance_m1_d2, i_feat)
            logits_m0_d1, logits_m0_d2 = self.predictModule5(user_spf1, user_spf2, i_feat)

            return logits_d1, logits_d2, logits_m0_d1, logits_m0_d2, logits_m1_d1, logits_m1_d2, logits_m2_d1, logits_m2_d2, logits_m3_d1, logits_m3_d2
        else:
            return logits_d1, logits_d2#, u_feat_enhance_m1_d2, u_feat_enhance_m3_d2, u_feat_enhance_m4_d2


if __name__ == '__main__':
    u_node = torch.LongTensor(torch.ones(1000).long()).cuda()
    i_node = torch.LongTensor(torch.ones(1000).long()).cuda()
    seq_d1 = torch.LongTensor(torch.ones(1000,25).long()).cuda()
    seq_d2 = torch.LongTensor(torch.ones(1000,25).long()).cuda()
    long_tail_mask_d1 = torch.LongTensor(torch.ones(1000).long()).cuda()
    long_tail_mask_d2 = torch.LongTensor(torch.ones(1000).long()).cuda()
    domain_id = torch.LongTensor(torch.ones(1000).long()).cuda()
    neg_samples = torch.LongTensor(torch.ones(1000,99).long()).cuda()

    u_feat = torch.LongTensor(torch.ones(1000,128).long()).float().cuda()
    innermatch = InnerGraphMatchingLayer(128).cuda()
    output = innermatch(u_feat,long_tail_mask_d1)
    #print(output.shape) #[bs,emb]
    model = NGMCDR(user_length=63275, user_emb_dim=128, item_length=1740, item_emb_dim=128, seq_len=25, m1_layers=2, m2_layers=2, m3_layers=2, m4_layers=2, hid_dim=32).cuda()
    logits_d1, logits_d2, u_feat_enhance_m1_d1, u_feat_enhance_m1_d2, u_feat_enhance_m2_d1,u_feat_enhance_m2_d2, u_feat_enhance_m3_d1,u_feat_enhance_m3_d2, u_feat_enhance_m4_d1, u_feat_enhance_m4_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
    print(u_feat_enhance_m1_d1.shape, u_feat_enhance_m1_d2.shape)
    logit_cl_m1 = torch.matmul(u_feat_enhance_m1_d1,u_feat_enhance_m1_d2.T)
    logit_cl_m1 = -torch.log(torch.diag(torch.exp(logit_cl_m1).sum(dim=0)))
    loss_cl_m1 = logit_cl_m1.mean()
    # print(pos_logit_d1.shape,neg_logit_d1.shape,pos_logit_d2.shape,neg_logit_d2.shape,user_enc_d1.shape,user_dec_d1.shape,user_enc_d2.shape,user_dec_d2.shape)
    # model = multiDomainGraphABL(user_length=63275, user_emb_dim=128, item_length=1740, item_emb_dim=128, seq_len=25, head_nums=8, layers=2, hid_dim=32, mask_rate_enc=0.5, mask_rate_dec=0.5).cuda()
    # pos_logit_d1, neg_logit_d1, pos_logit_d2, neg_logit_d2 = model(u_node,i_node,seq_d1,seq_d2,domain_id,neg_samples)
    # print(pos_logit_d1.shape,neg_logit_d1.shape,pos_logit_d2.shape,neg_logit_d2.shape)
    # macs, params = profile(model, inputs=(u_node,neigh_node,relat_feat,i_node,u_i_mask,i_u_mask,domain_ids,interact_nums, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(params)
    # output = model(u_node,neigh_node,relat_feat,i_node,i_neigh,u_i_mask,i_u_mask,domain_ids,interact_nums,u_i_mask_d1,i_u_mask_d1,u_i_mask_d2,i_u_mask_d2,u_i_mask_d3,i_u_mask_d3) 
    # print(output.shape)

    # seq_d1_feat = torch.zeros(8, 25, 64)
    # num_nodes = seq_d1_feat.shape[1]
    # perm = torch.randperm(num_nodes, device=seq_d1_feat.device)
    # mask_rate = 0.4
    # num_mask_nodes = int(mask_rate * num_nodes)

    # mask_nodes = perm[: num_mask_nodes]
    # keep_nodes = perm[num_mask_nodes: ]
    # print(mask_nodes)
    # seq_d1_feat[:,mask_nodes,:] = 1 # mask_feat

    # uigraph = GATGraphLayer(128,32)
    # user_feat = torch.ones(512,128)
    # seq_feat = torch.ones(512,25,128)
    # print(uigraph(user_feat,seq_feat).shape)