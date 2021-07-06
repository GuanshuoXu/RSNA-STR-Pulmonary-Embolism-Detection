import logging
import math
from numpy.lib.type_check import _imag_dispatcher
from torchvision.transforms.transforms import RandomAffine, RandomCrop
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
#from  fixmatch.FixMatch.dataset.randaugment import RandAugmentMC 
from randaugment import RandAugmentMC 
from torch.utils.data import Dataset, DataLoader
import albumentations
import pydicom
import cv2
import gdcm
import random
import torch

logger = logging.getLogger(__name__)
cifar10_mean = (0.456, 0.456, 0.456) #(0.4914, 0.4822, 0.4465)
cifar10_std = (0.224, 0.224, 0.224)#(0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


seed = 2001
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

class PE_SSL(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, targets, transform=None, pil=False, three=True, win=False):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list###[:1000]
        self.target_size=target_size
        self.transform=transform
        #self.images=images
        self.targets=targets
        self.pil=pil
        self.three=three
        self.win=win
    def __len__(self):
        return self.image_list.shape[0]
    def __getitem__(self,index):
        path='../../input/train/'
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        if self.three:
            data1 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
            data3 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
            x1 = data1.pixel_array       
            x3 = data3.pixel_array
            x1 = x1*data1.RescaleSlope+data1.RescaleIntercept        
            x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
            if self.win:
                x1 = np.expand_dims(window(x1, WL=-600, WW=1500), axis=2)        
                x3 = np.expand_dims(window(x3, WL=50, WW=400), axis=2)
            else:
                x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)        
                x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        data2 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        x2 = data2.pixel_array
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        if self.three:
            x = np.concatenate([x1, x2, x3], axis=2)
        else:
            x=x2#np.expand_dims(x2, axis=0)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        #x = x[bbox[1]+20:bbox[3]-20,bbox[0]+20:bbox[2]-20,:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        ##img = Image.fromarray(x)
        ##if self.transform is not None:
            ##img = self.transform(img)
        if self.transform is None:
            x = transforms.ToTensor()(x)
            x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
            #print('test')
        elif not self.pil:
           
            x = self.transform(image=x)['image']
            if self.three:
                x = x.transpose(2, 0, 1)
            else:
                x=np.expand_dims(x, axis=0)
                #print("trans0",x.shape)
            #print('22222',x.shape)
            #print('labled')
        else:
            #np.expand_dims(x, axis=2)
            img = Image.fromarray(x)
             
            x=self.transform(img)
            #print("trans0",x[0].shape)
            if not self.three:
                x=(x[0][1,:,:].unsqueeze(axis=0),x[1][1,:,:].unsqueeze(axis=0))
            #print("trans1",x[0].shape)
            #print('unlabeld')
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        y_true=self.targets[index]

        # if y_true!=y:
        #     print('yyyyyy', y, y_true)
        return x, y_true


def get_data(args):
# prepare input
    
    path='../process_input/split2/'
    path2='../lung_localization/split2/'
 
    with open(path +'image_list_train.pickle', 'rb') as f:
        image_list_train = pickle.load(f)#[:1000]
    with open(path+'image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    
    
    with open(path2+'bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f) 
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))
    print('AAAAAAAAAAAAA')
    with open(path+'image_list_valid.pickle', 'rb') as f:
        image_list_valid = pickle.load(f)#[:1000] 
    #with open('../process_input/split2/image_dict.pickle', 'rb') as f:
    #    image_dict = pickle.load(f) 
    with open(path2+'bbox_dict_valid.pickle', 'rb') as f:
        bbox_dict_valid = pickle.load(f)
    print(len(image_list_valid), len(image_dict), len(bbox_dict_valid))
    with open(path +'series_list_train.pickle', 'rb') as f:
        series_list_train = pickle.load(f)
    with open(path + 'series_list_valid.pickle', 'rb') as f:
        series_list_valid = pickle.load(f) 
    with open(path+'series_dict.pickle', 'rb') as f:
        series_dict = pickle.load(f)
    ###########
    #fix match
    # transform_labeled_fix = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32,
    #                           padding=int(32*0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    # transform_val = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    #base_dataset= PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size)#, transform=train_transform)
    #kaggle
    image_size = args.size
    # if args.three<=0:
    #     transform_labeled = albumentations.Compose([
    #     albumentations.RandomContrast(limit=0.2, p=1.0),
    #     albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    #     albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
    #     albumentations.Normalize(mean=0.456, std=0.224, max_pixel_value=255.0, p=1.0)
    #     ])
    # else:
    transform_labeled = albumentations.Compose([
    #albumentations.Flip(d=1),
    albumentations.RandomContrast(limit=0.2, p=1.0),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
    albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)])

    
    ############
    gt_z=[]
    gt_position=[]
    gt_ser=[]
    data_ratio=0.2
    count_pos=0

    def reduce_data(data_ratio, series_list):
        image_list_train=[]
        gt_ser=[]
        count_pos=0
        num_series= round(data_ratio* len(series_list))
        ser_idx= np.random.choice(len(series_list), size=num_series, replace=False)
        series_list_train=np.array(series_list)[ser_idx]
        
        for series_id in series_list_train:
            tmp_list=list(series_dict[series_id]['sorted_image_list'])
            #gt_ser.append(series_dict[series_id]['negative_exam_for_pe'])
            image_list_train += tmp_list
        #     for idx,img in enumerate(tmp_list):
        #         count_pos+= image_dict[img]['pe_present_on_image']
                # gt_position.append(idx/len(tmp_list))
                # gt_z.append(image_dict[img]['z_pos'])          
        #np.save('gt_z',np.array(gt_z))
        #def get_sets(series_list_train, series_dict, image_dict )
        
        return  series_list_train, ser_idx, np.array(image_list_train)
    ser_list_train, ser_idx, train_images=reduce_data(data_ratio, series_list_train)
    
    #type='pe'
    # targets_20, train_labeled_img20, targets_ser=get_labels(ser_list_train, series_dict, image_dict, type)
    # np.save('images_20',train_labeled_img20 )
    # np.save('targets_20',targets_20 )
    # targets_20, train_labeled_img20, targets_ser=get_labels(ser_list_train, series_dict, image_dict, type)
    # np.save('images_val10',image_list_valid[:10000])
    # np.save('targets_val10',targets_v )
    
    data_ratio=0.1
    if data_ratio<1:
        s_list_train, s_idx,_=reduce_data(data_ratio, ser_list_train)
        # image_list_train=[]
        # num_series= round(data_ratio* len(series_list_train))
        # ser_idx= np.random.choice(len(series_list_train), size=num_series, replace=False)
        # series_list_train=np.array(series_list_train)[ser_idx]
        
        # for series_id in series_list_train:
        #     tmp_list=list(series_dict[series_id]['sorted_image_list'])
        #     gt_ser.append(series_dict[series_id]['negative_exam_for_pe'])
        #     image_list_train += tmp_list
        #     for idx,img in enumerate(tmp_list):
        #         count_pos+= image_dict[img]['pe_present_on_image']
        #         gt_position.append(idx/len(tmp_list))
        #         gt_z.append(image_dict[img]['z_pos'])
                

    
    #num_labeled = len(ser_list_train) // 10
    if args.pos>0:
        three=(args.three>0)#True#False
        type='pos'

    elif args.z>0:
        three=False
        type='pos'
    else:
        type='pe'
        three=True#(args.three>0)#True
    win=(args.win>0)
    
    
    
    #else:
        #train_labeled_img, train_unlabeled_idxs, targets_labeled = x_u_split(num_labeled, targets, targets_ser,ser_list_train, series_dict, image_dict)
    if args.pos>0:
        targets_labeled, train_labeled_img, targets_ser=get_labels(ser_list_train, series_dict, image_dict, type)
        print('ttt', len(targets_labeled), len(train_labeled_img))#, len(train_images))
   
        #train_labeled_img=train_images
        #targets_labeled=targets
        train_unlabeled_dataset=None
    else:
        if args.resume>=0:
             print('loading data')
             train_labeled_img=np.load('images_02.npy')
             targets_labeled=np.load('targets_02.npy')
        else:
            targets_labeled, train_labeled_img, targets_ser=get_labels(s_list_train, series_dict, image_dict, type)
            print('ttt', len(targets_labeled), len(train_labeled_img), len(train_images))
            #np.save("images_02", train_labeled_img)
            #np.save('targets_02', targets_labeled)
        
        #dummy targets for unlabeld
        targets=np.arange(len(train_images))#np.array(targets_labeled)
        train_unlabeled_idxs = x_u_split(s_idx, ser_list_train)# np.arange(len(ser_list_train)) #12:17
        targets, unlbl_img, _= get_labels(ser_list_train[train_unlabeled_idxs], series_dict, image_dict)
        print(unlbl_img.shape[0])
        train_unlabeled_dataset = PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=unlbl_img, target_size=image_size, targets=targets,transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std,image_size=args.size), pil=True,win=win)#, three=three)
     #### cifar mean
    if args.up> 0.05:
        train_labeled_img, targets_labeled= x_u_split_equal(args.up, np.array(targets_labeled), train_labeled_img)#, series_dict, image_dict)#targets_ser,s_list_train,
    
    
    train_labeled_dataset = PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=train_labeled_img, target_size=image_size, targets=targets_labeled,transform=transform_labeled,three=three,win=win)
    print('bbbb unnnn')
    
    val_targets, val_images,_=get_labels(series_list_valid[0:300], series_dict, image_dict, type)
    print('pe num test', sum(val_targets[:10000]))
    test_dataset=PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=val_images[:10000], target_size=image_size, targets=val_targets[:20000],three=three,win=win)#,transform=transform_val)
    ##### validation set (not test)
    #st=np.random.get_state()
    #idx_val=np.random.choice(1000, size=500, replace=False) ###np.arange(700)#
    #np.random.set_state(st)
    val_pool=np.array(image_list_valid[10000:11000])
    #val_pool=pool[idx_val]
    y_val=[]
    for i in range(val_pool.shape[0]):
        y_val.append(image_dict[val_pool[i]]['pe_present_on_image'])
    print('validation: ', sum(y_val)/val_pool.shape[0])
    val_set = PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=val_pool, target_size=image_size, targets=y_val, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std,image_size=args.size), pil=True)
        
    return train_labeled_dataset, train_unlabeled_dataset, val_set

def get_labels(series_list, series_dict, image_dict, type='pe'):
    gt_list=[]
    gt_ser=[]
    image_lst=[] 
    count_pe=0
    #np.random.shuffle(labeled_idx)
    for n in tqdm(range(len(series_list))):
        gt_ser.append(series_dict[series_list[n]]['negative_exam_for_pe'])
        image_list_ser = series_dict[series_list[n]]['sorted_image_list']
        image_lst+=image_list_ser
        for m in range(len(image_list_ser)):
            if type=='pe':
                gt_list.append(image_dict[image_list_ser[m]]['pe_present_on_image'])
                count_pe+=gt_list[-1]
            elif type=='pos':
                gt_list.append(m/len(image_list_ser))
               
            else:
                gt_list.append(image_dict[image_list_ser[m]]['z_pos'])
    print('reduced data: ',len(series_list),'# lables: ',len(image_lst), 'pe ratio: ', count_pe/len(image_lst))
    return gt_list, np.array(image_lst), gt_ser

def x_u_split_equal(ratio,targets, img_list):#,labels_ser, series_list, series_dict, image_dict):
        total=0
        num_classes=2 ###########
        
        # num_pe=int(ratio*len(labels_ser))
        # label_per_class = [num_pe, len(labels_ser)-num_pe]#]np.array(, (1-ratio)*num_labeled], dtype=int)
        # print(label_per_class)
        # #labels = np.array(labels)
        # labels_ser = np.array(labels_ser)
        # unlabeled_idx = []

        img_pe=[]
        img_non_pe=[]
        labeled_img_list=[img_non_pe, img_pe]

        labeled_pe=[]
        labeled_non_pe=[]
        gt_labeled=[labeled_non_pe, labeled_pe]
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        #unlabeled_idx = np.array(range(len(labels)))
        
        # for i in range(num_classes):
        #     idx = np.where(labels_ser == i)[0]
        #     print('lllll', label_per_class[i], idx.shape)
        #     idx_lbl = np.random.choice(idx, label_per_class[i], replace=(i==0))
        #     #unlabeled_idx+=[x for x in idx if not x in idx_lbl]
           
        #     for j in idx_lbl:
        #         #images for series j
        #         images=np.array(series_dict[series_list[j]]['sorted_image_list'])
        #         total+=len(images)
        #         #labeled_img_list+=images
        #         images_lbls=np.zeros(len(images), dtype=int)
        #         for m in range(len(images)):
        #             images_lbls[m]=image_dict[images[m]]['pe_present_on_image']
                     
        #         #idx_class=np.where(images_lbls==i)[0]
        #         idx_class=(images_lbls==i)
        #        # print('444',idx_class.shape, images_lbls.shape, images_lbls[~idx_class].shape,images_lbls[idx_class].shape)
        #         gt_labeled[i].append(images_lbls[idx_class])
        #         labeled_img_list[i].append(images[idx_class])
        #         gt_labeled[(i+1)%2].append(images_lbls[~idx_class])
        #         #print('nu', np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
        #         labeled_img_list[(i+1)%2].append(images[~idx_class])
        #             #gt_labeled.append(ser_lbls)

        #
            #labeled_idx.extend(idx)
        
        #assert len(labeled_idx) == num_labeled
        #unlabeled_idx = np.array(unlabeled_idx)
        #print('unlabel',unlabeled_idx.shape)
        total=len(targets)
        slice4class=[total-int(total*ratio), int(total*ratio)]
        lst_img=[]
        lst_lbl=[]
        for i in [0,1]:
            #print(num_pe, np.concatenate(gt_labeled[0]).shape, np.concatenate(gt_labeled[1]).shape)
            idx = np.where(targets == i)[0]
            print('lllll', slice4class[i], idx.shape)
            idx_lbl = np.random.choice(idx, slice4class[i], replace=(i==1))
            # gt_labeled[i]=np.concatenate(gt_labeled[i])
            
            # labeled_img_list[i]=np.concatenate(labeled_img_list[i])
            # idx=np.random.choice(gt_labeled[i].size, num_slices[i], replace=(i==1))
            print("3333", idx.shape, idx_lbl.shape)#, num_pe, gt_labeled[i].size)
            lst_lbl.append(targets[idx_lbl])
            lst_img.append(img_list[idx_lbl])
        #print(lst_img[0].shape, lst_img[1].shape)    
        labeled_img_arr=np.concatenate(lst_img)
        gt_labeled=np.concatenate(lst_lbl)
        # labeled_idx = np.array(labeled_idx)
        print("bbbbb", gt_labeled.shape, gt_labeled.sum(), labeled_img_arr.shape)
        return labeled_img_arr, gt_labeled#, unlabeled_idx
def x_u_split(lbl_ser_idx, ser_list):
     
    unlabeled_idx=[x for x in range(len(ser_list)) if not x in lbl_ser_idx]
    print("unlabeld ",len(unlabeled_idx))
    return np.array(unlabeled_idx)

def x_u_split_old(num_labeled, labels, labels_ser, series_list, series_dict, image_dict):
    num_classes=2 ###########
    label_per_class =num_labeled // num_classes
    #labels = np.array(labels)
    labels_ser = np.array(labels_ser)
    labeled_idx = []
    labeled_img_list=[]
    gt_labeled=[]
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = []#np.array(range(len(labels)))
    
    for i in range(num_classes):
        idx = np.where(labels_ser == i)[0]
        print('lllll', labels_ser.shape, idx.shape)
        idx_lbl = np.random.choice(idx, label_per_class, False)
        unlabeled_idx+=[x for x in idx if not x in idx_lbl]
        labeled_idx.extend(idx_lbl)
    
    #assert len(labeled_idx) == args.num_labeled
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    
    np.random.shuffle(labeled_idx)
    for j in labeled_idx:
            images=series_dict[series_list[j]]['sorted_image_list']
            labeled_img_list+=images
            for m in range(len(images)):
                gt_labeled.append(image_dict[images[m]]['pe_present_on_image'])

    #
    print('# of labels',len(labeled_img_list))
    labeled_img_list= np.array(labeled_img_list)
    gt_labeled=np.array(gt_labeled)
    print('sss', gt_labeled.shape, labeled_img_list.shape, gt_labeled.mean())
    # if args.expand_labels or args.num_labeled < args.batch_size:
    #     num_expand_x = math.ceil(
    #         args.batch_size * args.eval_step / (num_labeled*200))
    #     labeled_img_list= np.hstack([labeled_img_list for _ in range(num_expand_x)])
    #     gt_labeled=np.hstack([gt_labeled for _ in range(num_expand_x)])
    print('sss22', gt_labeled.shape, labeled_img_list.shape, 'sum: ',gt_labeled.sum())
    return labeled_img_list, unlabeled_idx, gt_labeled


    
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

        
class TransformFixMatch(object):
    def __init__(self, mean, std, image_size=576):
        self.weak = transforms.Compose([
            #transforms.RandomVerticalFlip(), # new 
            transforms.RandomHorizontalFlip()])# 87 horizontal
            # transforms.RandomCrop(size=576,
            #                       padding=int(576*0.125),
            #                       padding_mode='reflect')])
            #transforms.RandomAffine(10, translate=(0.1,0.1))]) ## new
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(), #87= horizontak
           # transforms.RandomVerticalFlip(), ## new
            transforms.RandomAffine(20, translate=(0.2,0.2)),
            transforms.RandomResizedCrop(size=image_size),
            #transforms.RandomVerticalFlip(), ### new
            #transforms.CenterCrop(size=image_size*0.75),
           # transforms.Resize((image_size, image_size)),
            # transforms.RandomCrop(size=image_size,
            #                       padding=int(image_size*0.125),
            #                       padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


