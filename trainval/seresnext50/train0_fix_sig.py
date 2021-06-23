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
from train0_fix_sigR import pre_train
from pe0 import get_data
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
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

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--dist", type=int, default=-1, help="use dist alignment")
    parser.add_argument("--fl", type=int, default=-1, help="use focal loss")
    parser.add_argument("--max", type=float, default=0.9, help="max threshold")
    parser.add_argument("--mu", type=int, default=1, help="ratio of unlabeled in batch")
    parser.add_argument("--name", type=str, default="fix9", help="local_rank for distributed training on gpus")
    parser.add_argument("--up", type=float, default=0.05, help="upsample ratio")
    parser.add_argument("--win", type=int, default=-1, help="different windows")
    parser.add_argument("--pos", type=int, default=-1, help="pre train position")
    parser.add_argument("--z", type=int, default=-1, help="pre train z value")
    parser.add_argument("--resume", type=int, default=-1, help="resume")
    #image size, tal, lr ,reg ,up pe ratio
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

    if args.local_rank in [-1, 0]:
        os.makedirs('tensorboard', exist_ok=True)
        writer = SummaryWriter('tensorboard')

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

    # data_ratio=0.02
    # count_pos=0
    # gt_ser=[]
    # if data_ratio<1:
    #     image_list_train=[]
    #     num_series= round(data_ratio* len(series_list_train))
    #     ser_idx= np.random.choice(len(series_list_train), size=num_series, replace=False)
        
    #     series_list_train=np.array(series_list_train)
    #     series_list_train=series_list_train[ser_idx]
    #     for series_id in series_list_train:
    #         tmp_list=list(series_dict[series_id]['sorted_image_list'])
    #         gt_ser.append(series_dict[series_id]['negative_exam_for_pe'])
    #         image_list_train += tmp_list
    #         for img in tmp_list:
    #             count_pos+= image_dict[img]['pe_present_on_image']
                
    
    
    # print('reduced data: ',data_ratio, num_series,len(image_list_train), 'pos ratio: ', count_pos/len(image_list_train))
    # def x_u_split_equal(num_labeled, labels_ser, series_list, series_dict, image_dict):
    #     total=0
    #     num_classes=2 ###########
    #     label_per_class = num_labeled // num_classes
    #     print(label_per_class)
    #     #labels = np.array(labels)
    #     labels_ser = np.array(labels_ser)
    #     labeled_idx = []

    #     img_pe=[]
    #     img_non_pe=[]
    #     labeled_img_list=[img_non_pe, img_pe]

    #     labeled_pe=[]
    #     labeled_non_pe=[]
    #     gt_labeled=[labeled_non_pe, labeled_pe]
    #     # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    #     #unlabeled_idx = np.array(range(len(labels)))
        
    #     for i in range(num_classes):
    #         idx = np.where(labels_ser == i)[0]
            
    #         idx = np.random.choice(idx, label_per_class, False)
    #         print('lllll', labels_ser.shape, idx.shape)
    #         for j in idx:
    #             #images for series j
    #             images=np.array(series_dict[series_list[j]]['sorted_image_list'])
    #             total+=len(images)
    #             #labeled_img_list+=images
    #             images_lbls=np.zeros(len(images), dtype=int)
    #             for m in range(len(images)):
    #                 images_lbls[m]=image_dict[images[m]]['pe_present_on_image']
                     
    #             #idx_class=np.where(images_lbls==i)[0]
    #             idx_class=(images_lbls==i)
    #            # print('444',idx_class.shape, images_lbls.shape, images_lbls[~idx_class].shape,images_lbls[idx_class].shape)
    #             gt_labeled[i].append(images_lbls[idx_class])
    #             labeled_img_list[i].append(images[idx_class])
    #             gt_labeled[(i+1)%2].append(images_lbls[~idx_class])
    #             #print('nu', np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
    #             labeled_img_list[(i+1)%2].append(images[~idx_class])
    #                 #gt_labeled.append(ser_lbls)

    #     #
    #         #labeled_idx.extend(idx)
        
    #     #assert len(labeled_idx) == num_labeled
       
    #     num_pe=int(total/2)
    #     lst_img=[]
    #     lst_lbl=[]
    #     for i in [0,1]:
    #         #print(num_pe, np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
    #         gt_labeled[i]=np.concatenate(gt_labeled[i])
            
    #         labeled_img_list[i]=np.concatenate(labeled_img_list[i])
    #         idx=np.random.choice(gt_labeled[i].size, num_pe, replace=(i==1))
    #         print("3333", idx.shape, gt_labeled[i][idx].shape, num_pe, gt_labeled[i].size)
    #         lst_lbl.append(gt_labeled[i][idx])
    #         lst_img.append(labeled_img_list[i][idx])
    #     #print(lst_img[0].shape, lst_img[1].shape)    
    #     labeled_img_arr=np.concatenate(lst_img)
    #     gt_labeled=np.concatenate(lst_lbl)
    #     # labeled_idx = np.array(labeled_idx)
    #     print("bbbbb", gt_labeled.shape, gt_labeled.sum(), labeled_img_arr.shape)
    #     return labeled_img_arr, gt_labeled
        # np.random.shuffle(labeled_idx)
    
        # print('# of labels',len(labeled_img_list))
        # labeled_img_list= np.array(labeled_img_list)
        # gt_labeled=np.array(gt_labeled)
        # print('sss', gt_labeled.shape, labeled_img_list.shape, gt_labeled.mean())
        # if args.expand_labels or args.num_labeled < args.batch_size:
        #     num_expand_x = math.ceil(
        #         args.batch_size * args.eval_step / (args.num_labeled*200))
        #     labeled_img_list= np.hstack([labeled_img_list for _ in range(num_expand_x)])
        #     gt_labeled=np.hstack([gt_labeled for _ in range(num_expand_x)])
        
    class WeightedFocalLoss(nn.Module):
    #"Non weighted version of Focal Loss"
        def __init__(self, alpha=1.0, gamma=0.0):
            super(WeightedFocalLoss, self).__init__()
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
            self.gamma = gamma

        def forward(self, inputs, targets):
            loss = nn.BCEWithLogitsLoss(reduction='none')#nn.CrossEntropyLoss(reduction='none')
            BCE_loss =loss(inputs, targets) # F.binary_cross_entropy_with_logits
            targets = targets.type(torch.long)
            at = self.alpha.gather(0, targets.data.view(-1))
            pt = torch.exp(-BCE_loss)
            F_loss = at*(1-pt)**self.gamma * BCE_loss
            return F_loss.mean()
    # hyperparameters
    #samp_img, samp_lbl=x_u_split_equal(num_series//10,gt_ser,series_list_train, series_dict, image_dict)
   
    learning_rate = 0.0004#4
    batch_size = 16#32
    image_size = 432#576
    num_epoch = 7#1
    best_auc=0
    # build model
    if args.pos>0:
       model2, epoch=pre_train(args)
    epoch=0

    if args.local_rank != 0:
        torch.distributed.barrier()
    if args.pos>0:
         print('loading')
         model = seresnext50()
         model.load_state_dict(torch.load('weights'+args.name+'/' +'epoch{}'.format(epoch),map_location='cpu'))
         in_features = model.net.last_linear.in_features
         model.last_linear = nn.Linear(in_features, 1)
    else:
        model = seresnext50()
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    print('bef res')
    if args.resume >0:
        model.load_state_dict(torch.load('weightsmore_aug_wd_th/'+'epoch5',map_location='cpu'))
        #model.cuda()
    print('af res')
    model=model.to(args.device) or None
    args.pos=-1
    labeled_dataset, unlabeled_dataset, test_dataset = get_data(args)
    num_train_steps = int(len(labeled_dataset)/(batch_size*4)*num_epoch)   ##### 4 GPUs
    #print('num train steps:', num_train_steps)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) #1e-4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    print('opt')
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    print('cre')
    if args.fl >0:
        criterion = WeightedFocalLoss(alpha=0.2, gamma=1).to(args.device)#
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

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=batch_size*mu,
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
    image_list_valid=image_list_valid[:20000]
    datagenV = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
    generatorV = DataLoader(dataset=datagenV, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    #######
    #feature = np.zeros((len(image_list_train), 2048),dtype=np.float32)
    feature_val = np.zeros((len(image_list_valid), 2048),dtype=np.float32)
    pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)
    lambda_u=1
    T=1
    threshold_max=args.max
    threshold_min=0.05
    max_prob=0
    name=args.name
    alpha=0.2/(len(labeled_dataset)/batch_size)
    fac=1
    
    print(threshold_max, threshold_min, name, 'weights_decay 5e-4')
    print('start trainnnnn')
    for ep in range(1,num_epoch):
        losses = AverageMeter()
         
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        num_lbl=0
        num_pe=0
        pe_norm=1
        max_pseudo=0
       # if args.world_size > 1:
        # labeled_epoch = 0
        # unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(ep)
        unlabeled_trainloader.sampler.set_epoch(ep)
        ###############################
         
        model.train()
        for i,(inputs_x,labels) in tqdm(enumerate(labeled_trainloader)):
            start = i*batch_size
            end = start+batch_size
            #if i == len(generator)-1:
             #   end = len(generator.dataset)
            #inputs_x = images.to(args.device)

            labels = labels.float().to(args.device)
             
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
           # features, logits_x= model(images)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*mu+1).to(args.device)
            # #print('3')
            #inputs=torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            #targets_x = targets_x.to(args.device)
            #print('iiiii', inputs.shape, args.device)
            features,logits = model(inputs)#_x.to(args.device))
            #print('l1',logits.shape)
            logits = de_interleave(logits, 2*mu+1)
            #print('l2',logits.shape)
            logits_x = logits[:batch_size]
            #print('l3',logits_x.shape)
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            #print('l4',logits_u_w.shape, logits_u_s.shape)
            del logits
            # loss = criterion(logits.view(-1),labels)
            # 
             #Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            Lx=criterion(logits_x.view(-1), labels)
            #Lx=torch.mean(Lx)
            
            pseudo_label = torch.sigmoid(logits_u_w.detach()/T)#, dim=-1) #softmax
            # if max_prob < pseudo_label.max().item():
            #     max_prob = pseudo_label.max().item()
            if args.dist>0:
                
                pseudo_label=pe_norm* pseudo_label
            
                 
                
            #print(pseudo_label.shape, pseudo_label.max().item())
            max_prob=(pseudo_label.ge(threshold_max).float())
            max_, targets_u = torch.max(pseudo_label, dim=-1)
            min_probs, _= torch.min(pseudo_label, dim=-1)
            mask = max_prob + (pseudo_label.lt(threshold_min).float())
            num_pe+=max_prob.sum().item()
            num_lbl+=mask.sum().item()
            if max_pseudo< pseudo_label.max().item():
                max_pseudo=pseudo_label.max().item()
            per_pe= 0.06 /((num_pe+1)/ (num_lbl+ 1))
            per_no=0.94/((num_lbl-num_pe+1)/(num_lbl+ 1))
            pe_norm=per_pe /(per_pe+ per_no)
            
           # print('probs:',logits_u_w.detach()/args.T)#, max_probs[:20])
            Lu= (F.binary_cross_entropy_with_logits(logits_u_s.view(-1), pseudo_label.view(-1),  reduction='none') * mask).mean()
            #Lu = (nn.BCEWithLogitsLoss(logits_u_s, targets_u.float(),  reduction='none') )#* mask).mean()
            #print(Lu.shape, mask.shape)
            # Lu = (F.cross_entropy(logits_u_s, targets_u,
            #                       reduction='none') * mask).mean()
            loss = Lx + lambda_u * Lu
            if (args.local_rank == 0) & (i%50==0):
                print(f'loss: {loss.item()} loss_x: {Lx.item()}, loss u:{Lu.item()} lbls:{num_lbl} pos {num_pe/num_lbl} max {max_pseudo}  fac {fac} above {pseudo_label[max_prob.bool()].mean().item()}')
                if args.dist>0:
                    print("dist ", num_lbl,num_pe, per_pe, per_no, pe_norm)
            #print(i, start, end)
            #feature[start:end] = np.squeeze(features.cpu().data.numpy())
            fac=fac-alpha
            if args.local_rank in [-1, 0]:         

                    writer.add_scalar('train/1.train_loss', loss.item(), i)
                    writer.add_scalar('train/2.train_loss_x', Lx.item(), i)
                    writer.add_scalar('train/3.train_loss_u', Lu.item(), i)
                    writer.add_scalar('train/4.pe', num_pe/(num_lbl+1), i)
            # args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            # args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            losses.update(loss.item(), inputs.size(0))
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()

       # if args.local_rank == 0:
        print('epoch: {}, train_loss: {} Lx: {} Lu: {} '.format(ep,losses.avg,losses_x.avg,losses_u.avg), flush=True)
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
                    #print(lbl_num.sum(), pred_prob[idx].mean())

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
            out_dir = 'weights'+name+'/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if auc>best_auc:
                best_auc=auc
                print('best ', best_auc, ' saving...')
                torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
                torch.save(optimizer.state_dict(), out_dir+'opt_epoch{}'.format(ep))
                torch.save(scheduler.state_dict(), out_dir+'scd_epoch{}'.format(ep))
                out_dir = 'features0'+name+'/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                #np.save(out_dir+'feature_train', feature)
                np.save(out_dir+'feature_valid', feature_val)
                np.save(out_dir+'pred_prob_valid', pred_prob)
                np.save(out_dir+'y_true', y_true)
                

if __name__ == "__main__":
    main()
