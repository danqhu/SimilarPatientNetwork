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

from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import torch
import config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(config.random_seed)

deepfeatures = False


deepfeatures = True




if deepfeatures:
    data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot.csv')
    data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized.csv')
    data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized.csv')
else:
    data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_nodeep.csv')
    data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized_nodeep.csv')
    data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized_nodeep.csv')


data = data_norm

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




tumor_data = data[tumor_col]
lymph_node_data = data[lymph_node_col]
demographic_data = data[demographic_col]
biomarker_data = data[biomarker_col]
comorbidity_data = data[comorbidity_col]
deepfeature_data = data[deepfeature_col]



'''使用全部数据构建相似患者网络'''
def construct_graph(
    features, 
    labels, 
    weights=None,
    method='heat',
    output_dir=None,
    topk = 10,
    add_image=False):



    
   

    n_samples = len(labels)
    dist = np.zeros((n_samples,n_samples))
    if method!='random':
        if method == 'heat':
            for feature in features:
                dis = -0.5 * pair(feature) ** 2
                dis = np.exp(dis)
                dist += dis
        elif method == 'cos':
            for i, feature in enumerate(features):
                dist += cosine_similarity(feature)
                # dist += np.dot(feature, feature.T)
#                 print('dist= {}'.format(dist))  
        elif method == 'ncos':
            for feature in features:
                feature = normalize(feature, axis=1, norm='l1')
                dist += np.dot(feature, feature.T)
        dist/=len(features)
        inds = []
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
            inds.append(ind)
        with open(output_dir, 'w',encoding='utf8') as f:
            counter = 0
            for index, neighs in enumerate(inds):
                for neigh in neighs:
                    if neigh == index:
                        pass
                    else:
                        if labels[neigh] != labels[index]:
                            counter += 1
                            
                        f.write('{},{}\n'.format(index,neigh ))

    else:
        
        with open(output_dir, 'w',encoding='utf8') as f:
            for index in range(len(features[0])):
                neighs = []
                neigh_index = 0
                while neigh_index < topk:
                    neigh = np.random.randint(len(features[0]))
                    if neigh!=index and neigh not in neighs:
                        neighs.append(neigh)
                        neigh_index+=1
                for neigh in neighs:
                    f.write('{},{}\n'.format(index, neigh))



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





weights = [1,1,1,1,1,0.1]

for K in [2,4,6,8,10,12]:
    construct_graph([tumor_data,lymph_node_data,demographic_data,biomarker_data,comorbidity_data,deepfeature_data], labels, weights, method='cos',output_dir=config.local_clinical_data_folder_path+ 'graph_cos_{}.csv'.format(K),topk=K)





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





