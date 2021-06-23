import argparse
import numpy as np
import pandas as pd
import os
import cv2
#from torch._C import int16
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations #fix_match.FixMatch.
import pydicom
import copy
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

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
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform=None):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        self.transform=transform
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
        if self.transform is None:
            x = transforms.ToTensor()(x)
            x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
        else:
            x = self.transform(image=x)['image']
            x = x.transpose(2, 0, 1)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class seresnext50(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 2001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    import pickle
    with open('../process_input/split2/image_list_train.pickle', 'rb') as f:
        image_list_train = pickle.load(f)#[:1000]
    with open('../process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    with open('../process_input/split2/series_dict.pickle', 'rb') as f:
        series_dict = pickle.load(f) 
    with open('../lung_localization/split2/bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f) 
    with open('../process_input/split2/series_list_train.pickle', 'rb') as f:
        series_list_train = pickle.load(f) 
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))

    with open('../process_input/split2/image_list_valid.pickle', 'rb') as f:
        image_list_valid = pickle.load(f)#[:1000] 
    #with open('../process_input/split2/image_dict.pickle', 'rb') as f:
    #    image_dict = pickle.load(f) 
    with open('../lung_localization/split2/bbox_dict_valid.pickle', 'rb') as f:
        bbox_dict_valid = pickle.load(f)
    print(len(image_list_valid), len(image_dict), len(bbox_dict_valid), len(series_list_train))
    gt_ser=[]
    gt_img=[]
    data_ratio=0.2
    count_pos=0
    if data_ratio<1:
        image_list_train=[]
        num_series= round(data_ratio* len(series_list_train))
        ser_idx= np.random.choice(len(series_list_train), size=num_series, replace=False)
        
        series_list_train=np.array(series_list_train)
        for series_id in series_list_train[ser_idx]:
            tmp_list=list(series_dict[series_id]['sorted_image_list'])
            gt_ser.append(series_dict[series_id]['negative_exam_for_pe'])
            image_list_train += tmp_list
            for img in tmp_list:
                count_pos+= image_dict[img]['pe_present_on_image']
                gt_img.append(image_dict[img]['pe_present_on_image'])
    
    
    print('reduced data: ',data_ratio, num_series,len(image_list_train), 'pos ratio: ', count_pos/len(image_list_train))
    def x_u_split_equal(num_labeled, labels_ser, series_list, series_dict, image_dict):
        total=0
        num_classes=2 ###########
        label_per_class = num_labeled // num_classes
        print(label_per_class)
        #labels = np.array(labels)
        labels_ser = np.array(labels_ser)
        labeled_idx = []

        img_pe=[]
        img_non_pe=[]
        labeled_img_list=[img_non_pe, img_pe]

        labeled_pe=[]
        labeled_non_pe=[]
        gt_labeled=[labeled_non_pe, labeled_pe]
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        #unlabeled_idx = np.array(range(len(labels)))
        
        for i in range(num_classes):
            replace=False
            idx = np.where(labels_ser == i)[0]
            if idx.shape[0]<label_per_class:
                replace = True
            idx = np.random.choice(idx, label_per_class, replace=replace)
            print('lllll', labels_ser.shape, idx.shape)
            for j in idx:
                #images for series j
                images=np.array(series_dict[series_list[j]]['sorted_image_list'])
                total+=len(images)
                #labeled_img_list+=images
                images_lbls=np.zeros(len(images), dtype=int)
                for m in range(len(images)):
                    images_lbls[m]=image_dict[images[m]]['pe_present_on_image']
                     
                #idx_class=np.where(images_lbls==i)[0]
                idx_class=(images_lbls==i)
               # print('444',idx_class.shape, images_lbls.shape, images_lbls[~idx_class].shape,images_lbls[idx_class].shape)
                gt_labeled[i].append(images_lbls[idx_class])
                labeled_img_list[i].append(images[idx_class])
                gt_labeled[(i+1)%2].append(images_lbls[~idx_class])
                #print('nu', np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
                labeled_img_list[(i+1)%2].append(images[~idx_class])
                    #gt_labeled.append(ser_lbls)

        #
            #labeled_idx.extend(idx)
        
        #assert len(labeled_idx) == num_labeled
       
        num_pe=int(total/2)
        lst_img=[]
        lst_lbl=[]
        for i in [0,1]:
            #print(num_pe, np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
            gt_labeled[i]=np.concatenate(gt_labeled[i])
            
            labeled_img_list[i]=np.concatenate(labeled_img_list[i])
            idx=np.random.choice(gt_labeled[i].size, num_pe, replace=(i==1))
            print("3333", idx.shape, gt_labeled[i][idx].shape, num_pe, gt_labeled[i].size)
            lst_lbl.append(gt_labeled[i][idx])
            lst_img.append(labeled_img_list[i][idx])
        #print(lst_img[0].shape, lst_img[1].shape)    
        labeled_img_arr=np.concatenate(lst_img)
        gt_labeled=np.concatenate(lst_lbl)
        # labeled_idx = np.array(labeled_idx)
        print("bbbbb", gt_labeled.shape, gt_labeled.sum(), labeled_img_arr.shape)
        return labeled_img_arr, gt_labeled
    # hyperparameters
    #image_list_train, samp_lbl=x_u_split_equal(num_series,gt_ser,series_list_train, series_dict, image_dict)
    learning_rate = 0.0002#4
    batch_size = 16#32
    image_size = 576
    num_epoch = 7#1
    best_auc=0
    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = seresnext50()
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    # print('bef we')
    # model.load_state_dict(torch.load('weights_0.2/'+'epoch1', map_location='cpu'))
    # print('af we')
    model.to(args.device)

    num_train_steps = int(len(image_list_train)/(batch_size*4)*num_epoch)   ##### 4 GPUs
    print('num train steps:', num_train_steps)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    print('opt')
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    print('cre')
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    print('iterator for training')
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen,  batch_size=batch_size, sampler=sampler,num_workers=5, pin_memory=True)#, drop_last=True) #
    #print(len(generator), len(datagen))
    print('iterator for validation')
    ######
    image_list_valid=image_list_valid[:20000]
    datagenV = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
    generatorV = DataLoader(dataset=datagenV, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    feature = np.zeros((len(image_list_train), 2048),dtype=np.float32)
    feature_val = np.zeros((len(image_list_valid), 2048),dtype=np.float32)
    pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)

    print('start trainnnnn')
    for ep in range(0,num_epoch):
        losses = AverageMeter()
        sampler.set_epoch(ep)
        model.train()
        for i,(images,labels) in tqdm(enumerate(generator)):
            start = i*batch_size
            end = start+batch_size
            #if i == len(generator)-1:
             #   end = len(generator.dataset)
            images = images.to(args.device)
            labels = labels.float().to(args.device)

            features, logits = model(images)
            loss = criterion(logits.view(-1),labels)
            losses.update(loss.item(), images.size(0))
            #print(i, start, end)
            #feature[start:end] = np.squeeze(features.cpu().data.numpy())
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()

       # if args.local_rank == 0:
        print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)
#validaion phase
        model.eval()

        pos=0
        y_true=[]
        losses = AverageMeter()
        for i, (images, labels) in tqdm(enumerate(generatorV)):
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(generatorV)-1:
                    end = len(generatorV.dataset)
                images = images.cuda()
                labels = labels.float().cuda()

                features, logits = model(images)
                loss = criterion(logits.view(-1),labels)
                losses.update(loss.item(), images.size(0))
                pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy())
                lbl_num=labels.cpu().detach().numpy().reshape(-1)
                if lbl_num.sum()>0.0:
                    #print(lbl_num)
                    pos+=lbl_num.sum()
                    idx=np.where(lbl_num>0)
                    print(lbl_num.sum(), pred_prob[idx].mean())

                y_true.append(lbl_num)
                #feature_val[start:end] = np.squeeze(features.cpu().data.numpy())
        y_true=np.concatenate(y_true)
        label = np.zeros((len(image_list_valid),),dtype=int)        
        for i in range(len(image_list_valid)):
            label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        print('pos:', label.sum()/20000, pos)
        auc = roc_auc_score(y_true, pred_prob)
        #if args.local_rank == 0:
        print("checkpoint {} ...".format(ep))
        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()

        if args.local_rank == 0:
            out_dir = 'weights/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if auc>best_auc:
                best_auc=auc
                print('best ', best_auc, ' saving...')
                torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
                out_dir = 'features0/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save(out_dir+'feature_train', feature)
                np.save(out_dir+'feature_valid', feature_val)
                np.save(out_dir+'pred_prob_valid', pred_prob)
                

if __name__ == "__main__":
    main()
