import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from efficientnet_pytorch import EfficientNet
import pickle
import pydicom
import glob

def window(x, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

class BboxDataset(Dataset):
    def __init__(self, series_list):
        self.series_list = series_list
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        return index

class BboxCollator(object):
    def __init__(self, series_list):
        self.series_list = series_list
    def _load_dicom_array(self, f):
        dicom_files = glob.glob(os.path.join(f, '*.dcm'))
        dicoms = [pydicom.dcmread(d) for d in dicom_files]
        M = np.float32(dicoms[0].RescaleSlope)
        B = np.float32(dicoms[0].RescaleIntercept)
        z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
        sorted_idx = np.argsort(z_pos)
        dicom_files = np.asarray(dicom_files)[sorted_idx]
        dicoms = np.asarray(dicoms)[sorted_idx]
        selected_idx = [int(0.2*len(dicom_files)), int(0.3*len(dicom_files)), int(0.4*len(dicom_files)), int(0.5*len(dicom_files))]
        selected_dicom_files = dicom_files[selected_idx]
        selected_dicoms = dicoms[selected_idx]
        dicoms = np.asarray([d.pixel_array.astype(np.float32) for d in selected_dicoms])
        dicoms = dicoms * M
        dicoms = dicoms + B
        dicoms = window(dicoms, WL=100, WW=700)
        return dicoms, dicom_files, selected_dicom_files
    def __call__(self, batch_idx):
        study_id = self.series_list[batch_idx[0]].split('_')[0]
        series_id = self.series_list[batch_idx[0]].split('_')[1]
        series_dir = '../../../input/train/' + study_id + '/'+ series_id
        dicoms, dicom_files, selected_dicom_files = self._load_dicom_array(series_dir)
        image_list = []
        for i in range(len(dicom_files)):
            name = dicom_files[i][-16:-4]
            image_list.append(name)
        selected_image_list = []
        for i in range(len(selected_dicom_files)):
            name = selected_dicom_files[i][-16:-4]
            selected_image_list.append(name)
        x = np.zeros((4, 3, dicoms.shape[1], dicoms.shape[2]), dtype=np.float32)
        for i in range(4):
            x[i,0] = dicoms[i]
            x[i,1] = dicoms[i]
            x[i,2] = dicoms[i]
        return torch.from_numpy(x), image_list, selected_image_list, self.series_list[batch_idx[0]]

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

    # prepare input
    import pickle
    with open('../../process_input/split2/series_list_valid.pickle', 'rb') as f:
        series_list = pickle.load(f) 
    df = pd.read_csv('../lung_bbox.csv')
    bbox_image_id_list = df['Image'].values
    bbox_Xmin_list = df['Xmin'].values
    bbox_Ymin_list = df['Ymin'].values
    bbox_Xmax_list = df['Xmax'].values
    bbox_Ymax_list = df['Ymax'].values
    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        bbox_dict[bbox_image_id_list[i]] = [max(0.0, bbox_Xmin_list[i]), max(0.0, bbox_Ymin_list[i]), min(1.0, bbox_Xmax_list[i]), min(1.0, bbox_Ymax_list[i])]

    # build model
    model = efficientnet()
    model.load_state_dict(torch.load('weights/epoch34_polyak'))
    model = model.cuda()
    model.eval()

    pred_bbox = np.zeros((len(series_list)*4,4),dtype=np.float32)
    bbox_dict_valid = {}
    selected_image_list_valid = []

    # iterator for validation
    datagen = BboxDataset(series_list=series_list)
    collate_fn = BboxCollator(series_list=series_list)
    generator = DataLoader(dataset=datagen, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
    total_steps = len(generator)
    for i, (images, image_list, selected_image_list, series_id) in tqdm(enumerate(generator), total=total_steps):
        with torch.no_grad():
            start = i*4
            end = start+4
            if i == len(generator)-1:
                end = len(generator.dataset)*4
            images = images.cuda()
            logits = model(images)
            bbox = np.squeeze(logits.cpu().data.numpy())
            pred_bbox[start:end] = bbox
            selected_image_list_valid += list(selected_image_list)
            xmin = np.round(min([bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]])*512)
            ymin = np.round(min([bbox[0,1], bbox[1,1], bbox[2,1], bbox[3,1]])*512)
            xmax = np.round(max([bbox[0,2], bbox[1,2], bbox[2,2], bbox[3,2]])*512)
            ymax = np.round(max([bbox[0,3], bbox[1,3], bbox[2,3], bbox[3,3]])*512)
            bbox_dict_valid[series_id] = [int(max(0, xmin)), int(max(0, ymin)), int(min(512, xmax)), int(min(512, ymax))]

    total_loss = 0
    for i in range(len(series_list)*4):
        for j in range(4):
            total_loss += abs(pred_bbox[i,j]-bbox_dict[selected_image_list_valid[i]][j])
    total_loss = total_loss / len(series_list) / 4 / 4
    print("total loss: ", total_loss)

    with open('bbox_dict_valid.pickle', 'wb') as f:
        pickle.dump(bbox_dict_valid, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
