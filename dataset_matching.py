import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json

def seq_padding(seq, length_enc, long_length, pad_id):
    if len(seq)>= long_length:
        long_mask = 1
    else:
        long_mask = 0
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1:]
    else:
        enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq

    return enc_in, long_mask

class DualDomainSeqDataset(data.Dataset):
    def __init__(self,seq_len,isTrain,neg_nums,long_length,pad_id,csv_path=''):
        super(DualDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
    
    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        domain_id_old = self.domain_id[idx]
        label = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
        seq_d1_tmp,long_tail_mask_d1 = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp,long_tail_mask_d2 = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['long_tail_mask_d1'] = np.array([long_tail_mask_d1])
        sample['long_tail_mask_d2'] = np.array([long_tail_mask_d2])
        sample['domain_id'] = np.array([domain_id])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        # copy neg item
        # sample['user_node'] = np.repeat(sample['user_node'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d1'] = np.repeat(sample['seq_d1'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d2'] = np.repeat(sample['seq_d2'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['domain_id'] = np.repeat(sample['domain_id'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['i_node'] = np.concatenate((sample['i_node'],sample['neg_samples']),axis=0)
        sample['label'] = sample['label']
        # print("user_node:{}".format(sample['user_node']))
        # print("i_node:{}".format(sample['i_node']))
        # print("seq_d1:{}".format(sample['seq_d1']))
        # print("seq_d2:{}".format(sample['seq_d2']))
        # print("domain_id:{}".format(sample['domain_id']))
        # print("neg_samples:{}".format(sample['neg_samples']))
        return sample

def collate_fn_enhance(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    long_tail_mask_d1 = torch.cat([ torch.Tensor(sample['long_tail_mask_d1']) for sample in batch],dim=0)
    long_tail_mask_d2 = torch.cat([ torch.Tensor(sample['long_tail_mask_d2']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'long_tail_mask_d1' : long_tail_mask_d1,
            'long_tail_mask_d2': long_tail_mask_d2,
            'label':label,
            'domain_id' : domain_id,
            'neg_samples':neg_samples
            }
    return data

if __name__ == '__main__':
    # cross_csv_dir = "/ossfs/workspace/MRHG/mybank/CDR12MYBankRehash.csv"
    # data = pd.read_csv(cross_csv_dir).set_index(['user_id'],drop=False).sample(frac=1.0)#.reset_index(drop=True)
    # train_len = int(data.shape[0] * 0.80)
    # save_data_train = data.iloc[ : train_len]
    # save_data_val = data.iloc[ train_len: ]
    train_name = "/ossfs/workspace/MRHG/mybank/CDR12MYBankTrain.csv"
    val_name = "/ossfs/workspace/MRHG/mybank/CDR12MYBankTest.csv"
    # save_data_train.to_csv(train_name, index=False)
    # save_data_val.to_csv(val_name, index=False)
    dataset_cross = DualDomainSeqDataset(seq_len=25,isTrain=False,neg_nums=99,csv_path=train_name,long_length=5)
    trainLoader = data.DataLoader(dataset_cross, batch_size=512, shuffle=True, num_workers=0,collate_fn=collate_fn_enhance)
    for i,sample in enumerate(trainLoader):
        u_node = torch.LongTensor(sample['user_node'].long()).cuda()
        i_node = torch.LongTensor(sample['i_node'].long()).cuda()
        seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
        seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()

        long_tail_mask_d1 = torch.LongTensor(sample['long_tail_mask_d1'].long()).cuda()
        long_tail_mask_d2 = torch.LongTensor(sample['long_tail_mask_d2'].long()).cuda()
        label = torch.LongTensor(sample['label'].long()).cuda()
        domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
        neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
        print("u_node shape :{}".format(u_node.shape))
        print("i_node shape :{}".format(i_node.shape))
        print("seq_d1 shape :{}".format(seq_d1.shape))
        print("seq_d2 shape :{}".format(seq_d2.shape))
        print("long_tail_mask_d1 shape :{}".format(long_tail_mask_d1.shape))
        print("long_tail_mask_d2 shape :{}".format(long_tail_mask_d2.shape))
        print("label shape :{}".format(label.shape))
        print("domain_id shape :{}".format(domain_id.shape))
        print("neg_samples shape :{}".format(neg_samples.shape))
        break
    # a = [-1,-1,-1,-1,-1]
    # for i in range(len(a)) :
    #     a[i] += 20000
    # print(a)

