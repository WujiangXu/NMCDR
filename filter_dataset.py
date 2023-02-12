import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from random import sample

def select_overlap_user(train_name,save_train_name,save_train_d1_name,save_train_d2_name,overlap_ratio):
    data = pd.read_csv(train_name)
    user_node = data['user_id'].tolist()
    seq_d1 = data['seq_d1'].tolist()
    seq_d2 = data['seq_d2'].tolist()
    domain_id = data['domain_id'].tolist()
    user_node_overlap,seq_d1_overlap, seq_d2_overlap, domain_id_overlap  = [], [], [], []
    user_node_nolap,seq_d1_nolap, seq_d2_nolap, domain_id_nolap  = [], [], [], []
    for i in range(len(user_node)):
        seq1_tmp = json.loads(seq_d1[i])
        seq2_tmp = json.loads(seq_d2[i])
        if len(seq1_tmp)!=0 and len(seq2_tmp)!=0:
            user_node_overlap.append(user_node[i])
            seq_d1_overlap.append(seq1_tmp)
            seq_d2_overlap.append(seq2_tmp)
            domain_id_overlap.append(domain_id[i])
        else :
            user_node_nolap.append(user_node[i])
            seq_d1_nolap.append(seq1_tmp)
            seq_d2_nolap.append(seq2_tmp)
            domain_id_nolap.append(domain_id[i])
    print(len(user_node_overlap),len(user_node_nolap)) # 3384 69945
    #nolap_num = int(len(user_node_overlap)/overlap_ratio-len(user_node_overlap)) # 3384 + 
    sample_overlap_num = int(len(user_node_overlap)*overlap_ratio)
    idx_lst = [i for i in range(len(user_node_overlap))]
    select_idx = sample(idx_lst, sample_overlap_num)
    print(sample_overlap_num)
    # print(select_idx)
    for idx_tmp in select_idx:
        user_node_nolap.append(user_node_overlap[idx_tmp])
        seq_d1_nolap.append(seq_d1_overlap[idx_tmp])
        seq_d2_nolap.append(seq_d2_overlap[idx_tmp])
        domain_id_nolap.append(domain_id_overlap[idx_tmp])
    # print(len(user_node_nolap))
    dataframe = pd.DataFrame({'user_id':user_node_nolap,'seq_d1':seq_d1_nolap,'seq_d2':seq_d2_nolap,'domain_id':domain_id_nolap})
    dataframe.to_csv(save_train_name,index=False,sep=',')
    data_d1 = (dataframe.loc[dataframe['domain_id']==0])
    data_d2 = (dataframe.loc[dataframe['domain_id']==1])
    data_d1.to_csv(save_train_d1_name,index=False)
    data_d2.to_csv(save_train_d2_name,index=False)


overlap_ratio = 0.001

# all_name = "music_movie_train.csv" # name change in next use
# save_data_val = pd.read_csv(all_name)#.set_index(['user_id'],drop=False).sample(frac=1.0)#.reset_index(drop=True)
# train_len = int(data.shape[0] * 0.80)
# save_data_train = data.iloc[ : train_len]
# save_data_val = data.iloc[ train_len: ]
# data_d1 = (save_data_val.loc[save_data_val['domain_id']==0])
# data_d2 = (save_data_val.loc[save_data_val['domain_id']==1])
# data_d1.to_csv("music_movie_testD1.csv",index=False)
# data_d2.to_csv("music_movie_testD2.csv",index=False)
train_name = "music_movie_train.csv"
# val_name = "music_movie_test.csv"
# save_data_train.to_csv(train_name, index=False)
# save_data_val.to_csv(val_name, index=False)
#game video 90 refine
save_train_name = "music_movie_train"+str(int(overlap_ratio*100))+".csv"
# save_val_name = "music_movie_test"+str(int(overlap_ratio*100))+".csv"
save_train_d1_name = "music_movie_train"+str(int(overlap_ratio*100))+"D1.csv"
save_train_d2_name = "music_movie_train"+str(int(overlap_ratio*100))+"D2.csv"
select_overlap_user(train_name,save_train_name,save_train_d1_name,save_train_d2_name,overlap_ratio)
# select_overlap_user(val_name,save_val_name,overlap_ratio)
