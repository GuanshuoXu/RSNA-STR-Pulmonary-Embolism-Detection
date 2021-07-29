import numpy as np
import streamlit as st

 
pred= np.load('pred_prob_list_seresnext50_128.npy')
gt= np.load('gt_list_seresnext50_128.npy')
ids= np.load('id_list.npy')
lens= np.load('series_len_list.npy')
tmp = list(map(lambda x: x.split('_',1), ids))#np.array([x.split('_',1) for x in ids])

studies=np.ndarray(len(tmp), dtype=object)
types=np.ndarray(len(tmp), dtype=object)
images=np.ndarray(len(tmp), dtype=object)
prev_study=0
for l in lens:#[0:2]:#-1]:
    item= tmp[prev_study]
    #if len(item)>1:
    studies[prev_study:l] = item[0]#'np.repeat(item[0], l-prev_study)
    images[prev_study:prev_study+9] = 'study' 
    types[prev_study:prev_study+9]=np.array([x[1] for x in tmp[prev_study:prev_study+9]])
    #else:
    images[prev_study+9:l]=np.concatenate(tmp[prev_study+9:l])#np.squeeze(np.array(tmp[prev_study+9:l]))
    types[prev_study+9:l]='pe_slice' 
    prev_study=l

import pickle
with open('../../process_input/split2/series_list_valid.pickle', 'rb') as f:
    series_list_valid = pickle.load(f)
tmp_ser= list(map(lambda x: tuple(x.split('_',1)), series_list_valid))
dict_series = {x[0]:x[1] for x in tmp_ser}


import pandas as pd

preds_tbl = pd.DataFrame({'studies':studies,'slices':images,'types':types,'labels':gt,'predicted':pred})#, columns=['a','b','c'])
preds_tbl['dif']=preds_tbl.labels-preds_tbl.predicted
err_tbl=preds_tbl[abs(preds_tbl['dif'].values)>0.5] 
err_tbl=err_tbl.sort_values(by=['studies','dif'], ascending=False)

def slices4label(label, thresh_lbl=-1, thresh_pe=-1):
    df_lbl= err_tbl[(err_tbl.types==label) & (err_tbl.dif>=thresh_lbl)].sort_values(by='dif', ascending=False)
    stud_lbl = df_lbl.studies.values
    lbl_full=err_tbl[err_tbl.studies.isin(stud_lbl)]
    df_slices=lbl_full[(lbl_full.types=='pe_slice') & (lbl_full.dif >= thresh_pe)].sort_values(by=['studies','dif'], ascending=False)
    df_slices.reset_index(drop=True, inplace=True)
    df_slices.set_index('studies', inplace=True)
    dict_slices={x:df_slices.loc[[x]].slices.values for x in df_slices.index.unique()}
    return dict_slices, df_lbl, df_slices

dict_chronic, df_chronic, slices_chronic=slices4label('chronic_pe')

from matplotlib import pyplot as plt 
import os
import pydicom

def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    # convert ouside pixel-values to air:
    # I'm using <= -1000 to be sure that other defaults are captured as well
    images[images <= -1000] = 0
    
    # convert to HU
    for n in range(len(slices)):
        
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
            
        images[n] += np.int16(intercept)
    
    return np.array(images, dtype=np.int16)
def show_image2(series, image, title):
# Call the local dicom file 
    folder_path = r"../../../input/train"
    file_path = os.path.join(folder_path,series)
    file_name = image +'.dcm'
    file_path = os.path.join(file_path,file_name)
    print(file_path)
    fig = plt.figure(figsize=(20,20), dpi=300)
    #fig, axs = plt.subplots(2, 3, figsize=(10, 3))
    #fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    #print(axs.shape)
    #for i in range(4):
    ds = pydicom.dcmread(file_path)
    first_patient_pixels = transform_to_hu([ds])
    plt.title(title)
       # fig.add_subplot(2, 2, i+1)
    plt.imshow(first_patient_pixels[0],cmap=plt.cm.bone)
    #st.image(first_patient_pixels[0], clamp=True)
    st.pyplot(fig,dpi=300)
       # ax[1].imshow(first_patient_pixels[0],cmap=plt.cm.bone)
    #plt.show()

def show_study2(dict_slices,df_lbl,df_slice,study="",threshold=-9,num_studies=5, num_slices=12, n_col=4):
        if threshold>0:
            studies=df_lbl[df_lbl.dif>=threshold].studies.values
        else:
            studies=df_lbl.studies.values
        if study=="":    
            #study_idx= st.slider('Image (study)', 0, len(studies), 0)  # min: 0h, max: 23h, default: 17h
            #study=studies[study_idx]
            study=st.text_input('study name', 'bc60cab42620')
            print(study)
        try:
            
            slices=dict_slices[study]
        except:
            st.write(study)
            st.write(len(dict_slices))
            print('no erronious slices to show')
            return
        
        slice_idx=st.slider('Slice', 0, len(slices),0)
        slice=slices[slice_idx]
        #file_name = slice +'.dcm'    
        path=study+'//'+ dict_series[study]
        #file_path = os.path.join(path,file_name)
        
        diff_lbl=round(df_lbl[df_lbl['studies']==study].dif.values[0],3)
        diff_pe=round(df_slice[df_slice['slices']==slice].dif.values[0],3)
        title=f'diff label={diff_lbl}, diff pe={diff_pe}'
        show_image2(path,slice, title)

def show(label,thresh_lbl=-1, thresh_pe=-1):
    dict_lbl, df_lbl, slices_lbl=slices4label(label)
    #print(slices_lbl)
    #if slide:
    study_name=""
    #if st.button('Show slices by study name'):#, ['Name','errors'])=='Name':
    #study_name=st.text_input('study name', 'bc60cab42620')
        
    show_study2(dict_lbl,df_lbl,slices_lbl,study_name,thresh_lbl)
    #else:
    #   show_study(dict_lbl,df_lbl,slices_lbl,thresh_lbl)
#show_study2(dict_chronic,df_chronic,slices_chronic)
show('negative_exam_for_pe')