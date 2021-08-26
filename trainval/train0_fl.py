import argparse
import math

import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations
import pydicom
import copy
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def binary_focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input (torch.Tensor): input data tensor with shape :math:`(N, 1, *)`.
        target (torch.Tensor): the target tensor with shape :math:`(N, 1, *)`.
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`. Default: 0.25.
        gamma (float): Focusing parameter :math:`\gamma >= 0`. Default: 2.0.
        reduction (str, optional): Specifies the reduction to apply to the. Default: 'none'.
        eps (float): for numerically stability when dividing. Default: 1e-8.
    Returns:
        torch.tensor: the computed loss.
    Examples:
    """

    input = input.unsqueeze(dim=1).unsqueeze(dim=1)
    target = target.unsqueeze(dim=1).unsqueeze(dim=1)

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = - alpha * torch.pow((1. - probs + eps), gamma) * target * torch.log(probs + eps) \
               - (1 - alpha) * torch.pow(probs + eps, gamma) * (1. - target) * torch.log(1. - probs + eps)

    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2017focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, 1, *)`.
        - Target: :math:`(N, 1, *)`.
    Examples:
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, self.eps)


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=1.0, gamma=0.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


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
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=1,
                 gamma=3,
                 logits=True,
                 reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self,
                inputs,
                targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(
                inputs, targets, reduce=False)
        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


import torch.distributed as dist


class DistributedWeightedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(range(len(self.dataset)))
        # if self.shuffle:
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # else:
        #     indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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
        image_list_train = pickle.load(f)
    with open('../process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f)
    with open('../lung_localization/split2/bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f)
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))

    # hyperparameters
    learning_rate = 0.0004
    batch_size = 32
    image_size = 576
    num_epoch = 3

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = seresnext50()
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    num_train_steps = int(len(image_list_train)/(batch_size*4)*num_epoch)   # 4 GPUs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    # criterion = FocalLoss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    # iterator for training
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train,
                        target_size=image_size, transform=train_transform)
    sampler = DistributedWeightedSampler(datagen)
    generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=5, pin_memory=True)

    for ep in tqdm(range(num_epoch)):
        losses = AverageMeter()
        model.train()
        for j,(images,labels) in enumerate(tqdm(generator)):
            images = images.to(args.device)
            labels = labels.float().to(args.device)

            logits = model(images)
            loss = criterion(logits.view(-1), labels)
            # print(f'the loss value is {loss}')
            # losses.update(loss.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'loss:{losses.avg}')

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

        if args.local_rank == 0:
            out_dir = 'weights/new_sampler/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save(model.module.state_dict(), out_dir+'new_sampler_epoch{}'.format(ep))


if __name__ == "__main__":
    main()
