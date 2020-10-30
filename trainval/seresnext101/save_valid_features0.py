import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext101_32x4d
import random
from sklearn.metrics import roc_auc_score
import pickle
import pydicom

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread('../../input/train/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread('../../input/train/'+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread('../../input/train/'+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class seresnext101(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x

def main():

    # checkpoint list
    checkpoint_list = [
                       'epoch0',
                      ]

    # prepare input
    import pickle
    with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
        image_list_valid = pickle.load(f) 
    with open('../process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    with open('../lung_localization/split2/bbox_dict_valid.pickle', 'rb') as f:
        bbox_dict_valid = pickle.load(f)
    print(len(image_list_valid), len(image_dict), len(bbox_dict_valid))

    # hyperparameters
    batch_size = 96
    image_size = 576
    criterion = nn.BCEWithLogitsLoss().cuda()

    # start validation
    for ckp in checkpoint_list:

        # build model
        model = seresnext101()
        model.load_state_dict(torch.load('weights/'+ckp))
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.eval()

        feature = np.zeros((len(image_list_valid), 2048),dtype=np.float32)
        pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)

        # iterator for validation
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=18, pin_memory=True)

        losses = AverageMeter()
        for i, (images, labels) in enumerate(generator):
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(generator)-1:
                    end = len(generator.dataset)
                images = images.cuda()
                labels = labels.float().cuda()
                features, logits = model(images)
                loss = criterion(logits.view(-1),labels)
                losses.update(loss.item(), images.size(0))
                feature[start:end] = np.squeeze(features.cpu().data.numpy())
                pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy())

        out_dir = 'features0/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir+'feature_valid', feature)
        np.save(out_dir+'pred_prob_valid', pred_prob)

        label = np.zeros((len(image_list_valid),),dtype=int)        
        for i in range(len(image_list_valid)):
            label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        auc = roc_auc_score(label, pred_prob)

        print("checkpoint {} ...".format(ckp))
        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()

if __name__ == "__main__":
    main()
