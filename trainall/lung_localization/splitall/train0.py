import argparse
import numpy as np
import pandas as pd
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
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations
import pydicom
import copy

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

def window(x, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform):
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
        data = pydicom.dcmread('../../../input/train/'+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        x = data.pixel_array.astype(np.float32)
        x = x*data.RescaleSlope+data.RescaleIntercept
        x1 = window(x, WL=100, WW=700)
        x = np.zeros((x1.shape[0], x1.shape[1], 3), dtype=np.float32)
        x[:,:,0] = x1
        x[:,:,1] = x1
        x[:,:,2] = x1
        x = cv2.resize(x, (self.target_size,self.target_size))
        bboxes = [self.bbox_dict[self.image_list[index]]]
        class_labels = ['lung']
        transformed = self.transform(image=x, bboxes=bboxes, class_labels=class_labels)
        x = transformed['image']
        x = x.transpose(2, 0, 1)
        y = transformed['bboxes'][0]
        y = torch.from_numpy(np.array(y))
        return x, y

class efficientnet(nn.Module):
    def __init__(self ):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.net._fc.in_features
        self.last_linear = nn.Linear(in_features, 4)
    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

def main():
    print('hello')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    print(args.local_rank)
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
    with open('../../process_input/splitall/series_list_train.pickle', 'rb') as f:
        series_list_train = pickle.load(f) 
    with open('../../process_input/splitall/series_dict.pickle', 'rb') as f:
        series_dict = pickle.load(f) 
    with open('../../process_input/splitall/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f)
    df = pd.read_csv('../lung_bbox.csv')
    bbox_image_id_list = df['Image'].values
    bbox_Xmin_list = df['Xmin'].values
    bbox_Ymin_list = df['Ymin'].values
    bbox_Xmax_list = df['Xmax'].values
    bbox_Ymax_list = df['Ymax'].values
    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        bbox_dict[bbox_image_id_list[i]] = [max(0.0, bbox_Xmin_list[i]), max(0.0, bbox_Ymin_list[i]), min(1.0, bbox_Xmax_list[i]), min(1.0, bbox_Ymax_list[i])]
    image_list_train = []
    for series_id in series_list_train:
        sorted_image_list = series_dict[series_id]['sorted_image_list']
        num_image = len(sorted_image_list)
        selected_idx = [int(0.2*num_image), int(0.3*num_image), int(0.4*num_image), int(0.5*num_image)]
        print(selected_idx)
        if selected_idx[0]>=len(sorted_image_list):
            continue
        image_list_train.append(sorted_image_list[selected_idx[0]])
        image_list_train.append(sorted_image_list[selected_idx[1]])
        image_list_train.append(sorted_image_list[selected_idx[2]])
        image_list_train.append(sorted_image_list[selected_idx[3]])
    print(len(image_list_train))

    # hyperparameters
    learning_rate = 0.0004
    batch_size = 32
    image_size = 512
    num_polyak = 32
    num_epoch = 20

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = efficientnet()
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    #criterion = nn.MSELoss().to(args.device)
    criterion = nn.L1Loss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        #albumentations.RandomBrightness(limit=0.2, p=1.0),
        #albumentations.RandomContrast(limit=0.2, p=1.0),
        #albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        albumentations.Cutout(num_holes=1, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        #albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ], bbox_params=albumentations.BboxParams(format='albumentations', label_fields=['class_labels']))

    # iterator for training
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict, image_list=image_list_train, target_size=image_size, transform=train_transform)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=5, pin_memory=True)

    for ep in range(num_epoch):
        losses = AverageMeter()
        print('epoch22: {}'.format(ep, flush=True))
        model.train()
        for j,(images,labels) in enumerate(generator):
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = model(images)
            loss = criterion(logits,labels)
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)
            # if args.local_rank == 0:
            #     if j==len(generator)-num_polyak:
            #         #print("Polyak averaging start ...")
            #         averaged_model = copy.deepcopy(model)
            #     if j>len(generator)-num_polyak:
            #         for k in averaged_model.module.state_dict().keys():
            #             averaged_model.module.state_dict()[k].data += model.module.state_dict()[k].data
            #     if j==len(generator)-1:
            #         for k in averaged_model.module.state_dict().keys():
            #             averaged_model.module.state_dict()[k].data /= num_polyak
            #         #print("Polyak averaging end ...")

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)

        if args.local_rank == 0:
            out_dir = 'weights/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
           # torch.save(averaged_model.module.state_dict(), out_dir+'epoch{}_polyak'.format(ep))

if __name__ == "__main__":
    main()
