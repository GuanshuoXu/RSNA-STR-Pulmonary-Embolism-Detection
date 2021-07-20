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
#from fix_match.FixMatch.dataset.pe import get_data
from pe0 import get_data
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
 
def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X
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
 

def pre_train(args):
     
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    ##torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 2001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

    # hyperparameters
    #samp_img, samp_lbl=x_u_split_equal(num_series//10,gt_ser,series_list_train, series_dict, image_dict)
    labeled_dataset, unlabeled_dataset, test_dataset = get_data(args)
    
    learning_rate = 0.0004 #4
    batch_size = 16#32
    image_size = 576
    num_epoch = 1#1
    best_auc=0
    print('bef צםגק')
    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    print('bef we')
    model = seresnext50()
    if args.local_rank == 0:
        torch.distributed.barrier()
    
   
    #model.load_state_dict(torch.load('weightsmore_aug_wd_th/'+'epoch5'))
    print('af we')
    model=model.to(args.device) or None

    num_train_steps = int(len(labeled_dataset)/(batch_size*4)*num_epoch)   ##### 4 GPUs
    #print('num train steps:', num_train_steps)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) #1e-4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    print('opt')
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    print('cre')

    if args.pos>0: 
        criterion=nn.MSELoss().to(args.device)
    else:
        criterion=nn.BCEWithLogitsLoss().to(args.device)
    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])
    ####

    #DATASET_GETTERS[args.dataset]( args, './data')

    # if args.local_rank == 0:
    #     torch.distributed.barrier()
    mu=1
    num_workers=5
    train_sampler = DistributedSampler#RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True)

     

    # test_loader = DataLoader(
    #     test_dataset,
    #     sampler=SequentialSampler(test_dataset),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers)

    #print("dataloaders",len(labeled_trainloader), len(unlabeled_trainloader), len(test_loader))

    ####
    print('iterator for training')
    # datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform)
    # sampler = DistributedSampler(datagen)
    # generator = DataLoader(dataset=datagen,  batch_size=batch_size, sampler=sampler,num_workers=5, pin_memory=True)#, drop_last=True) #
    # #print(len(generator), len(datagen))
    print('iterator for validation')
    ######
    if args.pos<0:
        image_list_valid=image_list_valid[:10000]
        test_dataset= PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
    generatorV = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    #######
    #feature = np.zeros((len(image_list_train), 2048),dtype=np.float32)
    feature_val = np.zeros((len(image_list_valid), 2048),dtype=np.float32)
    pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)
    lambda_u=1
   
    name=args.name
    
    best_mse=9999
    print(  name, 'weights_decay 5e-4')
    print('start trainnnnn')
    for ep in range(0,num_epoch):
        losses = AverageMeter()
         
        losses_x = AverageMeter()
      
       # if args.world_size > 1:
        # labeled_epoch = 0
        # unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(ep)
         
         
        model.train()
        for i,(inputs_x,labels) in tqdm(enumerate(labeled_trainloader)):
            start = i*batch_size
            end = start+batch_size
            #if i == len(generator)-1:
             #   end = len(generator.dataset)
            inputs_x = inputs_x.to(args.device)
            labels = labels.float().to(args.device)
             
             
           # features, logits_x= model(images)
            batch_size = inputs_x.shape[0]
             
            #inputs=torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            #targets_x = targets_x.to(args.device)
            #print('iiiii', inputs.shape, args.device)
            features,logits = model(inputs_x)#_x.to(args.device))
             
            loss=criterion(logits.view(-1), labels)
            #Lx=torch.mean(Lx)
            
             
            if (args.local_rank == 0) & (i%10==0):
                print(f'loss: {loss.item()}')#' loss_x: {loss.item()} ')
                 
            losses.update(loss.item(), inputs_x.size(0))
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
                #print('11123333', logits.view(-1),labels)
                losses.update(loss.item(), images.size(0))
                
                if args.pos<0:        
                    pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy())
                    lbl_num=labels.cpu().detach().numpy().reshape(-1)
                    y_true.append(lbl_num)
        #         #feature_val[start:end] = np.squeeze(features.cpu().data.numpy())
       
        #label = np.zeros((len(image_list_valid),),dtype=float)        
        # for i in range(len(image_list_valid)):
        #     label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        # print('pos:', label.sum()/20000, pos)
        if args.pos<0:
            y_true=np.concatenate(y_true)
            auc = roc_auc_score(y_true, pred_prob)
            print("auc: ", auc)
        #if args.local_rank == 0:
        print("checkpoint {} ...".format(ep))
        loss_ep=losses.avg
        print('loss:{}'.format(loss_ep), flush=True)
        print()

        if args.local_rank == 0:
            out_dir = 'weights'+name+'/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if loss_ep<best_mse:
                best_mse=loss_ep
                print('saving')
                torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
                out_dir = 'features0'+name+'/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                #np.save(out_dir+'feature_train', feature)
                np.save(out_dir+'feature_valid', feature_val)
                np.save(out_dir+'pred_prob_valid', pred_prob)
                
    return model, ep
# if __name__ == "__main__":
#     main()
