import numpy as np
import pandas as pd
import pickle
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, auc, fbeta_score
from sklearn.decomposition import PCA
 



class PESeries(object):
    def __init__(self,
                 feature_array,
                 image_to_feature,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.image_to_feature=image_to_feature
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list
        self.seq_len=seq_len
    def __len__(self):
        return len(self.series_list)
    def get_features(self):
        x_ser=np.zeros((len(self.series_list), self.feature_array.shape[1])) 
        y_npe=np.zeros(len(self.series_list)) 
        for index in range(len(self.series_list)):
            image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
            # if len(image_list)>self.seq_len:
            #     x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            #     y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            #     mask = np.ones((self.seq_len,), dtype=np.float32)
            #     for i in range(len(image_list)):      
            #         x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
            #         y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            #     x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            #     y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
            # else:
            x = np.zeros((len(image_list), self.feature_array.shape[1]), dtype=np.float32)
            mask = np.zeros(len(image_list), dtype=np.float)
            y_pe = np.zeros(len(image_list), dtype=np.float)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
        #x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        #x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
            x_ser[index]=np.max(x, axis=0) #mean
            #y_pe_ser[index]=np.mean(y_pe, axis=0)
        #x_series = torch.tensor(x_series, dtype=torch.float32)
        #y_pe_ser = torch.tensor(y_pe_ser, dtype=torch.float32)
        #mask = torch.tensor(mask, dtype=torch.float32)
            y_npe[index] = self.series_dict[self.series_list[index]]['chronic_pe']#negative_exam_for_pe']
        # y_idt = self.series_dict[self.series_list[index]]['indeterminate']
        # y_lpe = self.series_dict[self.series_list[index]]['leftsided_pe']
        # y_rpe = self.series_dict[self.series_list[index]]['rightsided_pe']
        # y_cpe = self.series_dict[self.series_list[index]]['central_pe']
        # y_gte = self.series_dict[self.series_list[index]]['rv_lv_ratio_gte_1']
        # y_lt = self.series_dict[self.series_list[index]]['rv_lv_ratio_lt_1']
        # y_chronic = self.series_dict[self.series_list[index]]['chronic_pe']
        # y_acute_and_chronic = self.series_dict[self.series_list[index]]['acute_and_chronic_pe']
        return x_ser,  mask, y_npe #y_pe_ser,

# prepare input
with open('../process_input/split2/series_list_train.pickle', 'rb') as f:
    series_list_train = pickle.load(f)
with open('../process_input/split2/series_list_valid.pickle', 'rb') as f:
    series_list_valid = pickle.load(f) 
with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
    image_list_train = pickle.load(f)
with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
    image_list_valid = pickle.load(f) 
with open('../process_input/split2/image_dict.pickle', 'rb') as f:
    image_dict = pickle.load(f) 
with open('../process_input/split2/series_dict.pickle', 'rb') as f:
    series_dict = pickle.load(f)
feature_train = np.load('../seresnext50/features0/feature_train.npy')
feature_valid = np.load('../seresnext50/features0/feature_valid.npy')
print(feature_train.shape, feature_valid.shape, len(series_list_train), len(series_list_valid), len(image_list_train), len(image_list_valid), len(image_dict), len(series_dict))

image_to_feature_train = {}
image_to_feature_valid = {}
for i in range(len(feature_train)):
    image_to_feature_train[image_list_train[i]] = i
for i in range(len(feature_valid)):
    image_to_feature_valid[image_list_valid[i]] = i

seed = 2001
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

seq_len=128
# training

# # iterator for training
train_datagen = PESeries(feature_array=feature_train,
                          image_to_feature=image_to_feature_train,
                          series_dict=series_dict,
                          image_dict=image_dict,
                          series_list=series_list_train,
                          seq_len=seq_len)
# train_generator = DataLoader(dataset=train_datagen,
#                              batch_size=batch_size,
#                              shuffle=True,
#                              num_workers=8,
#                              pin_memory=True)
valid_datagen = PESeries(feature_array=feature_valid,
                          image_to_feature=image_to_feature_valid,
                          series_dict=series_dict,
                          image_dict=image_dict,
                          series_list=series_list_valid,
                          seq_len=seq_len)
# valid_generator = DataLoader(dataset=valid_datagen,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=8,
#                              pin_memory=True)

#series_features=np.array(len(series_list_train), feature_train.shape[1]) 
out_dir = 'linear_clf/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    x_train, mask, y_train = train_datagen.get_features()
    x_val, mask, y_val = valid_datagen.get_features()
    np.save(out_dir+'x_train', x_train)
    np.save(out_dir+'x_val', x_val)
    np.save(out_dir+'y_train_chro', y_train)
    np.save(out_dir+'y_val_chro', y_val)
else:
    x_train=np.load(out_dir+'x_train.npy')
    x_val=np.load(out_dir+'x_val.npy')
    y_train=np.load(out_dir+'y_train.npy')
    y_val=np.load(out_dir+'y_val.npy')
 
scaler = StandardScaler(with_mean=True,with_std=True)
scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
 
pca = PCA()
x_train_p= pca.fit_transform(x_train_s)
explained_variance_ratio_vec=pca.explained_variance_ratio_
res, = np.where(np.cumsum(explained_variance_ratio_vec) >= 0.95)
pc = res[0] + 1
variance = np.sum(explained_variance_ratio_vec[:pc])
print(f'Min PCA components that retain over 50% variance = {pc} (variance retained = {variance:.3f})')

clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, C=0.25, tol=1e-5, dual=False, class_weight='balanced', max_iter=100000))
clf.fit(x_train, y_train)
y_pred=clf.predict(x_val)
print('score: ', clf.score(x_val,y_val))
print('auc:', roc_auc_score(y_val,clf.decision_function(x_val)))
print('fbeta', fbeta_score(y_val, y_pred, beta=2))

from sklearn.linear_model import RidgeClassifier
clf = make_pipeline(StandardScaler(), RidgeClassifier(random_state=0, tol=1e-5, class_weight='balanced', max_iter=100000))
clf.fit(x_train, y_train)
y_pred=clf.predict(x_val)
print('score: ', clf.score(x_val,y_val))
print('auc:', roc_auc_score(y_val,clf.decision_function(x_val)))
print('fbeta', fbeta_score(y_val, y_pred, beta=2))


