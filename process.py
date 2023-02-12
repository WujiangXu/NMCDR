import pandas as pd
import json
from collections import defaultdict

def filter_data(filePath):
    data = []
    ratings = pd.read_csv(filePath, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating', 'timesteamp']

    rate_size_dic_i = ratings.groupby('itemId').size()
    # choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    ratings = ratings[~ratings['itemId'].isin(list(choosed_index_del_i))] # item freq more than 10

    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())

    u = len(user_unique)
    i = len(movie_unique)
    rating_num = len(ratings)
    ratings = ratings.drop(columns=['timesteamp'])
    return u, i, rating_num, user_unique, ratings

def filter_user(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('userId').size()
    rate_size_dic_u2 = ratings2.groupby('userId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2

def filter_item(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('itemId').size()
    rate_size_dic_u2 = ratings2.groupby('itemId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2

def reindex_ratings(ratings1,ratings2):
    user_unique1 = list(ratings1['userId'].unique())
    user_unique2 = list(ratings2['userId'].unique())
    item_unique1 = list(ratings1['itemId'].unique())
    item_unique2 = list(ratings2['itemId'].unique())
    user_dict = dict()
    item_dict = dict()
    for i in range(len(user_unique1)):
        user_dict[user_unique1[i]] = i
    for i in range(len(user_unique2)):
        user_dict[user_unique2[i]] = i + len(user_dict)
    for i in range(len(item_unique1)):
        item_dict[item_unique1[i]] = i
    for i in range(len(item_unique2)):
        item_dict[item_unique2[i]] = i + len(item_dict)
    ratings1['userId'] = ratings1['userId'].apply(lambda x :user_dict[x])
    ratings2['userId'] = ratings2['userId'].apply(lambda x :user_dict[x])
    ratings1['itemId'] = ratings1['itemId'].apply(lambda x :item_dict[x])
    ratings2['itemId'] = ratings2['itemId'].apply(lambda x :item_dict[x])
    print("all user number :{}, item number :{}".format(len(user_dict),len(item_dict)))
    return ratings1,ratings2
    
def find_dict(ratings):
    seq = defaultdict(list)  
    uid = ratings['userId'].tolist()
    iid = ratings['itemId'].tolist()
    for i in range(len(uid)):
        seq[uid[i]].append(iid[i])
    return seq

# music domain 0 movie domain 1 all filter user number :110073, item number :44741
music_csv = "ratings_Digital_Music.csv"
movie_csv = "ratings_Movies_and_TV.csv"
u1, i1, rating_num1, user_unique1, ratings1 = filter_data(music_csv)
u2, i2, rating_num2, user_unique2, ratings2 = filter_data(movie_csv)
# save_csv_name = "music_movie_all.csv"
print(u1,i1,rating_num1,u2,i2,rating_num2)

ratings1, ratings2 = filter_user(ratings1, ratings2) # del overlap user < 5
ratings1, ratings2 = filter_item(ratings1, ratings2) # del overlap user < 5
# print(len(ratings1),len(ratings2))

ratings1 = ratings1.loc[ratings1['Rating']>=3.0]
ratings1 = ratings1.drop(columns=['Rating'])
ratings2 = ratings2.loc[ratings2['Rating']>=3.0]
ratings2 = ratings2.drop(columns=['Rating'])
print(len(list(ratings1['userId'].unique())),len(list(ratings1['itemId'].unique())),len(ratings1))
print(len(list(ratings2['userId'].unique())),len(list(ratings2['itemId'].unique())),len(ratings2))
# print(len(ratings1),len(ratings2)) # paper dataset anlysis here 80% for train 20% for test

# ratings1,ratings2 = reindex_ratings(ratings1,ratings2)
# # print(ratings1,ratings2)

# seq1 = find_dict(ratings1)
# seq2 = find_dict(ratings2)
# # print(seq1,seq2)
# user_unique1 = list(ratings1['userId'].unique())
# user_unique2 = list(ratings2['userId'].unique())
# user_node,seq_d1, seq_d2, domain_id  = [], [], [], []

# for u_id_tmp in user_unique1:
#     if len(seq1[u_id_tmp])>=5 and (len(seq2[u_id_tmp])>=5 or len(seq2[u_id_tmp])==0):
#         user_node.append(u_id_tmp)
#         seq_d1.append(seq1[u_id_tmp])
#         seq_d2.append(seq2[u_id_tmp])
#         domain_id.append(0)

# for u_id_tmp in user_unique2:
#     if len(seq2[u_id_tmp])>=5 and (len(seq1[u_id_tmp])>=5 or len(seq1[u_id_tmp])==0):
#         user_node.append(u_id_tmp)
#         seq_d1.append(seq1[u_id_tmp])
#         seq_d2.append(seq2[u_id_tmp])
#         domain_id.append(1)


# dataframe = pd.DataFrame({'user_id':user_node,'seq_d1':seq_d1,'seq_d2':seq_d2,'domain_id':domain_id})
# print(len(dataframe))
# user_unique1 = list(dataframe['user_id'].unique())
# user_dict = dict()
# for i in range(len(user_unique1)):
#     user_dict[user_unique1[i]] = i
# dataframe['user_id'] = dataframe['user_id'].apply(lambda x :user_dict[x])
# print(dataframe)
# dataframe.to_csv(save_csv_name,index=False,sep=',')