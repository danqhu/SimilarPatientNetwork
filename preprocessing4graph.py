import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
from sklearn.model_selection import  train_test_split,StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys,os
import json
import random
import copy

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import torch
import config


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# # 设置随机数种子
# setup_seed(config.random_seed)

# deepfeatures = False


# deepfeatures = True




# if deepfeatures:
#     data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot.csv')
#     data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized.csv')
#     data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized.csv')
# else:
#     data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_nodeep.csv')
#     data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized_nodeep.csv')
#     data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized_nodeep.csv')


# # data = data_norm
data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot.csv')
labels = pd.read_csv(config.local_clinical_data_folder_path + 'labels_binary_N2.csv').values.squeeze()


tumor_col = [
    "主肿物长径",
    "主肿物短径",
    "主肿物形状_毛糙",
    "主肿物形状_分叶",
    "是否侵犯血管",
    "是否侵犯胸膜",
    "主肿物位置_其他",
    "主肿物位置_右肺上叶",
    "主肿物位置_右肺下叶",
    "主肿物位置_右肺中叶",
    "主肿物位置_左肺上叶",
    "主肿物位置_左肺下叶",
    "主肿物磨玻璃或实性_实性",
    "主肿物磨玻璃或实性_混杂磨玻璃",
    "主肿物磨玻璃或实性_磨玻璃"
]

lymph_node_col = [
    "肺门淋巴结长径",
    "肺门淋巴结短径",
    "纵隔淋巴结长径",
    "纵隔淋巴结短径"
]

demographic_col = [
    "Age",
    "Smoking_history",
    "Drinking_history",
    "Family_tumor_history",
    "Gender"
]

biomarker_col = [
    "Pretreatment_CEA1",
    "Pretreatment_CA1991",
    "Pretreatment_CA1251",
    "Pretreatment_NSE1",
    "Pretreatment_CYFRA2111",
    "Pretreatment_SCC1"
]

comorbidity_col = [
    "高血压",
    "糖尿病",
    "肺结核",
    "心血管疾病",
    "脑血管疾病"
]


deepfeature_col = [
    "2","3","4","5","6","7","8","9"
]




'''使用全部数据构建相似患者网络'''
def construct_graph(
    data, 
    labels, 
    weights=None,
    method='heat',
    sigma=0.01,
    output_dir=None,
    topk = 10,
    deepfeature=False,
    save_results=True):


    all_data = data.values
    tumor_data = data[tumor_col].values
    lymph_node_data = data[lymph_node_col].values
    demographic_data = data[demographic_col].values
    biomarker_data = data[biomarker_col].values
    comorbidity_data = data[comorbidity_col].values
    if deepfeature:
        deepfeature_data = data[deepfeature_col].values
   

    n_samples = len(labels)
    dist = np.zeros((n_samples,n_samples))
    inds = []

    if method!='random':
        if method == 'heat':
            dist = pairwise_distances(all_data) **2
            sigma = np.mean(dist)
            dist = np.exp(-dist/sigma)
        elif method == 'cos':

            dist += weights[i] * cosine_similarity(all_data)
            # dist += np.dot(feature, feature.T)
#                 print('dist= {}'.format(dist))  

        # dist/=len(data)
        
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
            inds.append(ind)
        
        if save_results:
            file_path=output_dir+ 'graph_{}_{}.csv'.format(method,topk)
            with open(file_path, 'w',encoding='utf8') as f:
                counter = 0
                for index, neighs in enumerate(inds):
                    for neigh in neighs:
                        if neigh == index:
                            # pass
                            f.write('{},{}\n'.format(index,neigh ))
                        else:
                            if labels[neigh] != labels[index]:
                                counter += 1
                                # pass
                            else:
                                f.write('{},{}\n'.format(index,neigh ))
    return inds


    # else:
        
    #     with open(output_dir, 'w',encoding='utf8') as f:
    #         for index in range(len(features[0])):
    #             neighs = []
    #             neigh_index = 0
    #             while neigh_index < topk:
    #                 neigh = np.random.randint(len(features[0]))
    #                 if neigh!=index and neigh not in neighs:
    #                     neighs.append(neigh)
    #                     neigh_index+=1
    #             for neigh in neighs:
    #                 f.write('{},{}\n'.format(index, neigh))



def construct_graph_seperated(features, label, 
                    method='heat',
                    output_dir=None,topk = 10,seed=10,add_image=False):
    ix = 1
    kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for ix,(train_index,test_index) in enumerate(kfold.split(features[0],label)):
        
        output_train_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_trainsplit{}.txt'.format(method,topk,seed,ix + 1))
        output_valid_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_validsplit{}.txt'.format(method,topk,seed,ix + 1))
        output_test_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_testsplit{}.txt'.format(method,topk,seed,ix + 1))
        train_valid_index = copy.deepcopy(train_index)
        train_index = train_index[:int(len(train_index)*0.8)]
        valid_index = train_index[int(len(train_index)*0.8):]
        train_test_index = np.concatenate([train_index,test_index])
        dist_train = np.zeros((features[0][train_index].shape[0],features[0][train_index].shape[0]))
        dist_valid = np.zeros((features[0][train_valid_index].shape[0],features[0][train_valid_index].shape[0]))
        dist_test = np.zeros((features[0][train_test_index].shape[0],features[0][train_test_index].shape[0]))

        if method!='random':
            if method == 'heat':
                for feature in features:
                    dis = -0.5 * pair(feature[train_index]) ** 2
                    dis = np.exp(dis)
                    dist_train += dis
                    dis = -0.5 * pair(feature[train_valid_index]) ** 2
                    dis = np.exp(dis)
                    dist_valid += dis
                    dis = -0.5 * pair(feature[train_test_index]) ** 2
                    dis = np.exp(dis)
                    dist_test += dis
            elif method == 'cos':
                for feature in features:
                    dist_train += np.dot(feature[train_index], feature[train_index].T)
                    dist_valid += np.dot(feature[train_valid_index], feature[train_valid_index].T)
                    dist_test += np.dot(feature[train_test_index], feature[train_test_index].T)
                    
            elif method == 'ncos':
                for feature in features:
                    feature = normalize(feature, axis=1, norm='l1')
                    dist_train += np.dot(feature[train_index], feature[train_index].T)
                    dist_valid += np.dot(feature[train_valid_index], feature[train_valid_index].T)                    
                    dist_test += np.dot(feature[train_test_index], feature[train_test_index].T)
                    
            dist_train/=len(features)
            dist_valid/=len(features)
            dist_test/=len(features)
            inds = []
            for i in range(dist_train.shape[0]):
                ind = np.argpartition(dist_train[i, :], -(topk+1))[-(topk+1):]
                inds.append(ind)

            with open(output_train_filename, 'w',encoding='utf8') as f:
                for index, neighs in enumerate(inds):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh ))
                        
            inds_valid = copy.deepcopy(inds)
            for i in range(len(dist_train),dist_valid.shape[0]):
                ind = np.argpartition(dist_valid[i, :], -(topk+1))[-(topk+1):]
                inds_valid.append(ind)

            with open(output_valid_filename, 'w',encoding='utf8') as f:
                for index, neighs in enumerate(inds_valid):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh ))
            inds_test = copy.deepcopy(inds)
            for i in range(len(dist_train),dist_test.shape[0]):
                ind = np.argpartition(dist_test[i, :], -(topk+1))[-(topk+1):]
                inds_test.append(ind)

            with open(output_test_filename, 'w',encoding='utf8') as f:
                for index, neighs in enumerate(inds_test):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh ))            
        else:
            
            inds = []
            for index in range(dist_train.shape[0]):
                neigh_index = 0
                neighs = []
                while neigh_index < topk:
                    neigh = np.random.randint(dist_train.shape[0])
                    if neigh!=index and neigh not in neighs:

                        neighs.append(neigh)
                        neigh_index+=1
                inds.append(neighs)
            with open(output_train_filename, 'w',encoding='utf8') as f:
                for index, neighs in enumerate(inds):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh ))
            #valid graph                        
            inds_valid = copy.deepcopy(inds)
            for index in range(len(train_index),dist_valid.shape[0]):
                neigh_index = 0
                neighs = []
                while neigh_index < topk:
                    neigh = np.random.randint(dist_valid.shape[0])
                    if neigh!=index and neigh not in neighs:
                        neighs.append(neigh)
                        neigh_index+=1
                inds_valid.append(neighs)

            with open(output_valid_filename, 'w',encoding='utf8') as f:
                counter = 0
                for index, neighs in enumerate(inds_valid):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh ))
            #test graph
            inds_test = copy.deepcopy(inds)
            for index in range(len(train_index),dist_test.shape[0]):
                neigh_index = 0
                neighs = []
                while neigh_index < topk:
                    neigh = np.random.randint(dist_test.shape[0])
                    if neigh!=index and neigh not in neighs:
                        neighs.append(neigh)
                        neigh_index+=1
                inds_test.append(neighs)


            with open(output_test_filename, 'w',encoding='utf8') as f:
                counter = 0
                for index, neighs in enumerate(inds_test):
                    for neigh in neighs:
                        if neigh == index:continue
                        f.write('{} {}\n'.format(index,neigh )) 


def construct_PosNeg_graphs(    
    data, 
    labels,
    train_val_index=None,
    train_index=None,
    val_index=None,
    test_index=None, 
    topk = 10,
    method='heat',
    sigma=0.01,
    output_dir=None,
    deepfeature=False,
    save_results=False,
    test=False):

    train_val_data = data.loc[train_val_index, :].reset_index(inplace=False, drop=True)
    train_val_labels = labels[train_val_index]

    tumor_data = train_val_data[tumor_col].values
    lymph_node_data = train_val_data[lymph_node_col].values
    demographic_data = train_val_data[demographic_col].values
    biomarker_data = train_val_data[biomarker_col].values
    comorbidity_data = train_val_data[comorbidity_col].values
    if deepfeature:
        deepfeature_data = train_val_data[deepfeature_col].values
    
    pos_data = train_val_data.loc[train_val_labels==1, :]
    neg_data = train_val_data.loc[train_val_labels==0, :]
    pos_data_index = pos_data.index
    neg_data_index = neg_data.index



    n_pos = pos_data.shape[0]
    n_neg = neg_data.shape[0]


    pos_dist = np.zeros((n_pos,n_pos))
    neg_dist = np.zeros((n_neg,n_neg))
    inds = np.zeros((train_val_data.shape[0], topk+1))

    edges_train_val = []
    edges_train_val_test_pair = []

    if method!='random':
        if method == 'heat':
            pos_dist = pairwise_distances(pos_data) **2
            pos_sigma = np.mean(pos_dist)
            pos_dist = np.exp(-pos_dist/pos_sigma)

            neg_dist = pairwise_distances(neg_data) **2
            neg_sigma = np.mean(neg_dist)
            neg_dist = np.exp(-neg_dist/neg_sigma)


        elif method == 'cos':

            dist += cosine_similarity(data)
            # dist += np.dot(feature, feature.T)
#                 print('dist= {}'.format(dist))  

        # dist/=len(data)
        
        for i in range(pos_dist.shape[0]):
            ind = np.argpartition(pos_dist[i, :], -(topk+1))[-(topk+1):]
            inds[pos_data_index[i],:] = pos_data_index[ind]
        
        for i in range(neg_dist.shape[0]):
            ind = np.argpartition(neg_dist[i, :], -(topk+1))[-(topk+1):]
            inds[neg_data_index[i],:] = neg_data_index[ind]

        
        
        for row in range(inds.shape[0]):
            for col in range(inds.shape[1]):
                edges_train_val.append([row, int(inds[row,col])])
                if train_val_labels[row] != train_val_labels[int(inds[row,col])]:
                    print('counter node link')
        edges_train_val = np.array(edges_train_val, dtype=np.int32)
    
    
    
    
    
    if test:
        topk = 10
        test_data = data.loc[test_index, :].values
        inds_pos = np.zeros((test_data.shape[0], topk+1))
        inds_neg = np.zeros((test_data.shape[0], topk+1))
        test_pos_dist = np.zeros((test_data.shape[0], n_pos))
        test_neg_dist = np.zeros((test_data.shape[0], n_neg))

        if method!='random':
            if method == 'heat':
                test_pos_dist = pairwise_distances(test_data,pos_data) **2
                test_pos_sigma = np.mean(test_pos_dist)
                test_pos_dist = np.exp(-test_pos_dist/test_pos_sigma)

                test_neg_dist = pairwise_distances(test_data,neg_data) **2
                test_neg_sigma = np.mean(test_neg_dist)
                test_neg_dist = np.exp(-test_neg_dist/test_neg_sigma)

            
            for i in range(test_pos_dist.shape[0]):
                ind = np.argpartition(test_pos_dist[i, :], -(topk+1))[-(topk+1):]
                inds_pos[i,:] = pos_data_index[ind]
            
            for i in range(test_neg_dist.shape[0]):
                ind = np.argpartition(neg_dist[i, :], -(topk+1))[-(topk+1):]
                inds_neg[i,:] = neg_data_index[ind]

            edges_train_val_test_pair = []
            node_index = train_val_data.shape[0]
            inds_pos[:,-1] = node_index
            inds_neg[:,-1] = node_index
            for i in range(test_data.shape[0]):
                edges_train_val_test_pos = copy.deepcopy(edges_train_val.tolist())
                edges_train_val_test_neg = copy.deepcopy(edges_train_val.tolist())
                

                for col in range(inds_pos.shape[1]):
                    edges_train_val_test_pos.append([node_index, int(inds_pos[i,col])])
                    # if train_val_labels[int(inds_pos[i,col])] != 1:
                    #     print('counter node link')

                for col in range(inds_neg.shape[1]):
                    edges_train_val_test_neg.append([node_index, int(inds_neg[i,col])])
                    # if train_val_labels[int(inds_neg[i,col])] != 0:
                    #     print('counter node link')
                edges_train_val_test_pos = np.array(edges_train_val_test_pos, dtype=np.int32)
                edges_train_val_test_neg = np.array(edges_train_val_test_neg, dtype=np.int32)
                edges_train_val_test_pair.append([edges_train_val_test_pos, edges_train_val_test_neg, node_index])

        # if save_results:
        #     file_path=output_dir+ 'graph_{}_{}.csv'.format(method,topk)
        #     with open(file_path, 'w',encoding='utf8') as f:
        #         counter = 0
        #         for index, neighs in enumerate(inds):
        #             for neigh in neighs:
        #                 if neigh == index:
        #                     # pass
        #                     f.write('{},{}\n'.format(index,neigh ))
        #                 else:
        #                     if labels[neigh] != labels[index]:
        #                         counter += 1
        #                         # pass
        #                     else:
        #                         f.write('{},{}\n'.format(index,neigh ))
    return edges_train_val, edges_train_val_test_pair



# weights = [1,1,0.1,1,0.1,1]

# for K in [0,2,4,6,8,10,12]:
#     construct_graph(data, labels, weights, method='heat',output_dir=config.local_clinical_data_folder_path,topk=K)





# data = pd.read_csv(config.local_clinical_data_folder_path + 'data_merge.csv')


# def parse_location(loc):
#     if not loc:
#         return 0
#     elif  '左'  in loc and '上' in loc:
#         return 1
#     elif '左'  in loc and '下' in loc:
#         return 2
#     elif '右'  in loc and '上' in loc:
#         return 3
#     elif '右'  in loc and '下' in loc:
#         return 4
#     elif '右'  in loc and '中' in loc:
#         return 5
#     elif ('左' in loc and '右' in loc ) or '双' or '两' in loc :
#         return 6
#     else:
#         return 7
# def construct_graph(features, label, 
#                     method='heat',
#                     output_dir=None,topk = 10,add_image=False):
   
#     num = len(label)
#     dist = np.zeros((features[0].shape[0],features[0].shape[0]))
#     if method!='random':
#         if method == 'heat':
#             for feature in features:
#                 dis = -0.5 * pair(feature) ** 2
#                 dis = np.exp(dis)
#                 dist += dis

#         elif method == 'cos':
#             for feature in features:
#                 dist += np.dot(feature, feature.T)
# #                 print('dist= {}'.format(dist))  
#         elif method == 'ncos':
#             for feature in features:
#                 feature = normalize(feature, axis=1, norm='l1')
#                 dist += np.dot(feature, feature.T)
#         dist/=len(features)
#         inds = []
#         for i in range(dist.shape[0]):
#             ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
#             inds.append(ind)
#         with open(output_dir, 'w',encoding='utf8') as f:
#             counter = 0
#             for index, neighs in enumerate(inds):
#                 for neigh in neighs:
#                     if neigh == index:
#                         pass
#                     else:
#                         if label[neigh] != label[index]:
#                             counter += 1
                            
#                         f.write('{} {}\n'.format(index,neigh ))

#     else:
        
#         with open(output_dir, 'w',encoding='utf8') as f:
#             for index in range(len(features[0])):
#                 neighs = []
#                 neigh_index = 0
#                 while neigh_index < topk:
#                     neigh = np.random.randint(len(features[0]))
#                     if neigh!=index and neigh not in neighs:
#                         neighs.append(neigh)
#                         neigh_index+=1
#                 for neigh in neighs:
#                     f.write('{} {}\n'.format(index, neigh))
                    
# # 训练图单独构建，验证和测试图在训练图的基础上加入验证和测试节点
# def construct_train_graph(features, label, 
#                     method='heat',
#                     output_dir=None,topk = 10,seed=10,add_image=False):
#     ix = 1
#     kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
#     for ix,(train_index,test_index) in enumerate(kfold.split(features[0],label)):
        
#         output_train_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_trainsplit{}.txt'.format(method,topk,seed,ix + 1))
#         output_valid_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_validsplit{}.txt'.format(method,topk,seed,ix + 1))
#         output_test_filename = os.path.join(output_dir,'graph_{}_{}_seed{}_testsplit{}.txt'.format(method,topk,seed,ix + 1))
#         train_valid_index = copy.deepcopy(train_index)
#         train_index = train_index[:int(len(train_index)*0.8)]
#         valid_index = train_index[int(len(train_index)*0.8):]
#         train_test_index = np.concatenate([train_index,test_index])
#         dist_train = np.zeros((features[0][train_index].shape[0],features[0][train_index].shape[0]))
#         dist_valid = np.zeros((features[0][train_valid_index].shape[0],features[0][train_valid_index].shape[0]))
#         dist_test = np.zeros((features[0][train_test_index].shape[0],features[0][train_test_index].shape[0]))

#         if method!='random':
#             if method == 'heat':
#                 for feature in features:
#                     dis = -0.5 * pair(feature[train_index]) ** 2
#                     dis = np.exp(dis)
#                     dist_train += dis
#                     dis = -0.5 * pair(feature[train_valid_index]) ** 2
#                     dis = np.exp(dis)
#                     dist_valid += dis
#                     dis = -0.5 * pair(feature[train_test_index]) ** 2
#                     dis = np.exp(dis)
#                     dist_test += dis
#             elif method == 'cos':
#                 for feature in features:
#                     dist_train += np.dot(feature[train_index], feature[train_index].T)
#                     dist_valid += np.dot(feature[train_valid_index], feature[train_valid_index].T)
#                     dist_test += np.dot(feature[train_test_index], feature[train_test_index].T)
                    
#             elif method == 'ncos':
#                 for feature in features:
#                     feature = normalize(feature, axis=1, norm='l1')
#                     dist_train += np.dot(feature[train_index], feature[train_index].T)
#                     dist_valid += np.dot(feature[train_valid_index], feature[train_valid_index].T)                    
#                     dist_test += np.dot(feature[train_test_index], feature[train_test_index].T)
                    
#             dist_train/=len(features)
#             dist_valid/=len(features)
#             dist_test/=len(features)
#             inds = []
#             for i in range(dist_train.shape[0]):
#                 ind = np.argpartition(dist_train[i, :], -(topk+1))[-(topk+1):]
#                 inds.append(ind)

#             with open(output_train_filename, 'w',encoding='utf8') as f:
#                 for index, neighs in enumerate(inds):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh ))
                        
#             inds_valid = copy.deepcopy(inds)
#             for i in range(len(dist_train),dist_valid.shape[0]):
#                 ind = np.argpartition(dist_valid[i, :], -(topk+1))[-(topk+1):]
#                 inds_valid.append(ind)

#             with open(output_valid_filename, 'w',encoding='utf8') as f:
#                 for index, neighs in enumerate(inds_valid):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh ))
#             inds_test = copy.deepcopy(inds)
#             for i in range(len(dist_train),dist_test.shape[0]):
#                 ind = np.argpartition(dist_test[i, :], -(topk+1))[-(topk+1):]
#                 inds_test.append(ind)

#             with open(output_test_filename, 'w',encoding='utf8') as f:
#                 for index, neighs in enumerate(inds_test):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh ))            
#         else:
            
#             inds = []
#             for index in range(dist_train.shape[0]):
#                 neigh_index = 0
#                 neighs = []
#                 while neigh_index < topk:
#                     neigh = np.random.randint(dist_train.shape[0])
#                     if neigh!=index and neigh not in neighs:

#                         neighs.append(neigh)
#                         neigh_index+=1
#                 inds.append(neighs)
#             with open(output_train_filename, 'w',encoding='utf8') as f:
#                 for index, neighs in enumerate(inds):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh ))
#             #valid graph                        
#             inds_valid = copy.deepcopy(inds)
#             for index in range(len(train_index),dist_valid.shape[0]):
#                 neigh_index = 0
#                 neighs = []
#                 while neigh_index < topk:
#                     neigh = np.random.randint(dist_valid.shape[0])
#                     if neigh!=index and neigh not in neighs:
#                         neighs.append(neigh)
#                         neigh_index+=1
#                 inds_valid.append(neighs)

#             with open(output_valid_filename, 'w',encoding='utf8') as f:
#                 counter = 0
#                 for index, neighs in enumerate(inds_valid):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh ))
#             #test graph
#             inds_test = copy.deepcopy(inds)
#             for index in range(len(train_index),dist_test.shape[0]):
#                 neigh_index = 0
#                 neighs = []
#                 while neigh_index < topk:
#                     neigh = np.random.randint(dist_test.shape[0])
#                     if neigh!=index and neigh not in neighs:
#                         neighs.append(neigh)
#                         neigh_index+=1
#                 inds_test.append(neighs)


#             with open(output_test_filename, 'w',encoding='utf8') as f:
#                 counter = 0
#                 for index, neighs in enumerate(inds_test):
#                     for neigh in neighs:
#                         if neigh == index:continue
#                         f.write('{} {}\n'.format(index,neigh )) 




# tumor_data= data[['主肿物位置', '主肿物长径', '主肿物短径', '主肿物形状_毛刺','SUV_tumor','主肿物形状_分叶','主肿物形状_不规则',
#                     '主肿物形状_模糊','主肿物磨玻璃或实性','有无强化','是否侵犯血管','是否侵犯胸膜','是否侵犯支气管'
# ]]
# lymph_data = data[['肺门淋巴结长径','肺门淋巴结短径','纵隔淋巴结长径','纵隔淋巴结短径','SUV_feimen', 'SUV_zongge','side_feimen', 'side_zongge']]
# tumor_sign = data[[ 'CEA', 'CA199', 'CA125', 'NSE', 'CYFRA21', 'SCC','CEA_is_normal', 'CA199_is_normal', 'CA125_is_normal', 'NSE_is_normal', 'CYFRA21_is_normal',
#        'SCC_is_normal' ]]
# user_data = data[['Sys_patientNo', '性别', '年龄']]

# user_data['age_category'] = user_data['年龄'] // 10
# tumor_data['主肿物位置'] = tumor_data['主肿物位置'].apply(parse_location)


# tumor_cat_feat = ['主肿物磨玻璃或实性','主肿物位置']
# lymph_cat_feat = ['side_feimen', 'side_zongge']
# user_cat_feat = ['age_category']
 
# tumor_duplies = pd.get_dummies(tumor_data,columns=tumor_cat_feat)
# lymph_duplies = pd.get_dummies(lymph_data,columns=lymph_cat_feat)
# user_duplies = pd.get_dummies(user_data,columns=user_cat_feat)
# label = data.loc[:, '病理N_category']
# label = label.replace({
#             1: 0,
#             2: 1,
#             3: 1
#         })



# tumor_norm_feat = [ '主肿物长径', '主肿物短径','SUV_tumor']
# lymph_norm_feat= ['肺门淋巴结长径','肺门淋巴结短径','纵隔淋巴结长径','纵隔淋巴结短径','SUV_feimen', 'SUV_zongge']
# tumor_sign_norm_feat = [ 'CEA', 'CA199', 'CA125', 'NSE', 'CYFRA21', 'SCC']
# for norm_feature,node_data in zip([tumor_norm_feat,lymph_norm_feat,tumor_sign_norm_feat],[tumor_duplies,lymph_duplies,tumor_sign]):
    
#     for norm_item in norm_feature:
#         if norm_item in node_data.columns:
#             node_data[norm_item] = (node_data[norm_item] - node_data[norm_item].mean())/(node_data[norm_item].std())







# np.save(config.local_clinical_data_folder_path + 'feat.npy',pd.concat([tumor_duplies,lymph_duplies,tumor_sign],axis=1).values)
# np.save(config.local_clinical_data_folder_path + 'label.npy',label.values)




# for K in [2,4,6,8,10,12]:
#     construct_train_graph([tumor_duplies,lymph_duplies,tumor_sign], label, method='cos',output_dir=config.local_clinical_data_folder_path+'graph_ncos_{}.txt'.format(K),topk=K)




# for K in [2,4,6,8,10,12]:
#     construct_graph([tumor_duplies,lymph_duplies,tumor_sign], label, method='random',output_dir=config.local_clinical_data_folder_path+ 'graph_random_{}.txt'.format(K),topk=K)





