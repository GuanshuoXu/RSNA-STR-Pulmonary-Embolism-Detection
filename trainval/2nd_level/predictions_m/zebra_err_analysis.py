import numpy as np
import streamlit as st
import imageio
from PIL import Image
from matplotlib import pyplot as plt
import os
import pydicom
import cv2

# getting the predictions and corresponding data
pred= np.load('pred_prob_list_seresnext50_128.npy')
gt= np.load('gt_list_seresnext50_128.npy')
ids= np.load('id_list.npy')
lens= np.load('series_len_list.npy')
tmp = list(map(lambda x: x.split('_',1), ids))#np.array([x.split('_',1) for x in ids]) 




#process the data into a data frame 
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
up_path='../../'
with open(up_path+'process_input/split2/series_list_valid.pickle', 'rb') as f:
    series_list_valid = pickle.load(f)
with open(up_path+'process_input/split2/image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
with open(up_path+'lung_localization/split2/bbox_dict_valid.pickle', 'rb') as f:
        bbox_dict = pickle.load(f)
tmp_ser= list(map(lambda x: tuple(x.split('_',1)), series_list_valid))
dict_series = {x[0]:x[1] for x in tmp_ser}

 
import pandas as pd

preds_tbl = pd.DataFrame({'studies':studies,'slices':images,'types':types,'labels':gt,'predicted':pred})#, columns=['a','b','c'])
preds_tbl['dif']=preds_tbl.labels-preds_tbl.predicted
preds_tbl['err']=  (0.5 <=(preds_tbl.labels.values+preds_tbl.predicted.values )) & ((preds_tbl.labels.values+preds_tbl.predicted.values)< 1.5)
preds_tbl['err']=preds_tbl['err'].astype('int')
preds_tbl.loc[preds_tbl.types=='negative_exam_for_pe', 'labels']=1 -preds_tbl[preds_tbl.types=='negative_exam_for_pe'].labels.values
preds_tbl.loc[preds_tbl.types=='negative_exam_for_pe', 'predicted']=1 -preds_tbl[preds_tbl.types=='negative_exam_for_pe'].predicted.values

 


#get the slices associated with the label
def slices4label(tbl,label, df_lbl=None,thresh_lbl=0.7, thresh_pe=0.7):
    if df_lbl is None:
        df_lbl= tbl[(tbl.types==label) & (tbl.dif>=thresh_lbl)].sort_values(by='dif', ascending=False)
    stud_lbl = df_lbl.studies.values
    indeter=preds_tbl[(preds_tbl.types=='indeterminate') & (preds_tbl.labels==1.0)].studies.values
     
    stud_lbl=np.setdiff1d(stud_lbl, indeter)
    
    lbl_full=tbl[tbl.studies.isin(stud_lbl)]
    df_lbl=lbl_full[(lbl_full.types==label)] 
    df_slices=lbl_full[(lbl_full.types=='pe_slice') & (lbl_full.dif >= thresh_pe)]
    df_slices.reset_index(drop=True, inplace=True)
    df_slices.set_index('studies', inplace=True)
    dict_slices={x:df_slices.loc[[x]].slices.values for x in df_slices.index.unique()}
    return dict_slices, df_lbl, df_slices


#Main function - gets the label to show
def show_errors(label):
    df_pos=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==1.0)]
    df_neg=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==0.0)]
    #dict_fn ,df_fn, slices_fn = slices4label(preds_tbl, label, df_pos, -9, -9)
    
    st.write('positive = show the slices with PE in PE-positive images')
    st.write('false positive = slices for which PE predicted > 0.5 in non-PE images')
    if st.radio('Choose', ['positive', 'false positive'])=='positive':
        df_err=df_pos
        types='positive'
        slider_txt="PE probabilty predicted for image is smaller or equal to"
    else:
        df_err=df_neg
        types='false positive'
        slider_txt='PE probabilty predicted for image is bigger than'
    
    #filtering indeterminate studies
    indeter=preds_tbl[(preds_tbl.types=='indeterminate') & (preds_tbl.labels==1.0)].studies.values
    df_err=df_err[~df_err.studies.isin(indeter)]#stud_lbl=np.setdiff1d(stud_lbl, indeter)
    
    dict_err ,df,slices_err = slices4label(preds_tbl, label, df_err, -9, -9)
    prob=st.slider(slider_txt,0.0,1.0,0.05)
    if types=='positive':
        df_sort = df_err[df_err.predicted <= prob].sort_values(by='predicted', ascending=False)
    else:
        df_sort = df_err[df_err.predicted >= prob].sort_values(by='predicted', ascending=True)
    #print(df_sort)
    studies=df_sort.studies.values
    study_idx= st.slider('Image (study) #', 0, len(studies)-1, 0)  # min: 0h, max: 23h, default: 17h
    if study_idx<0:
        print('no studies to show')
        return
    study=studies[study_idx]
        #study = st.text_input('study name', 'bc60cab42620')
    path = study + '//' + dict_series[study]
        
    folder_path = up_path +'../input/train' ###r"../../../input/train"
    file_path = os.path.join(folder_path,path)
    try:

        slices = dict_err[study]
        #slices_tmp = [(s,pydicom.read_file(file_path + '/' + s+'.dcm')) for s in slices]
        #slices_tmp.sort(key = lambda x: float(x[1].ImagePositionPatient[2]))
        dict_idx={x:idx for idx, x in enumerate(slices)}#_tmp)}
        

        if types=='positive':
            slices2show=slices_err[(slices_err.slices.isin(slices))& (slices_err.labels==1.0)].slices.values #(slices_err.predicted<0.5)&
        else:
            slices2show=slices_err[(slices_err.slices.isin(slices))& (slices_err.predicted>0.5)& (slices_err.labels==0.0)].slices.values #
    except Exception as e: 
        print(e)
        st.write(study)

        st.write('no erronious slices to show')
        return
    if slices2show.size==0:
        st.write('no erronious slices to show')
        return
    prob_lbl = df_err[df_err['studies'] == study].predicted.values[0]
    st.write(study + ' has ', len(slices2show),  ' ' + types + ' slices out of ', len(slices), ' slices')
    option=st.radio('show',['gif', 'every fifth slice', 'specific','grid'])
    if option=='gif':#bGif
        
        file_path = os.path.join(folder_path,path)

        first_patient= load_slice(file_path,slices2show)
        first_patient_pixels = transform_to_hu(first_patient)
        
        imageio.mimsave("/tmp/gif.gif", first_patient_pixels, duration=0.1)
         
        st.image("/tmp/gif.gif")
    
    
    elif option=='every fifth slice':
        slice_idx=0
       # if st.button('next one'):

             
        while slice_idx<len(slices2show):
            
            slice = slices2show[slice_idx]
            idx=dict_idx[slice]
        

            prob_pe = slices_err[slices_err['slices'] == slice].predicted.values[0]
        
            pe_label=slices_err[slices_err['slices'] == slice].labels.values[0]
            title = f'index={idx}, pe_slice={pe_label}, prob pe 4 image={prob_lbl:.3f}, prob pe 4 slice={prob_pe:.3f}'
        # st.write(title)
            slice_idx=slice_idx+5
            show_image2(path, slice, title)
    elif option=='specific':
        idx=st.number_input('slice index:', 0)
        #path = study + '//' + dict_series[study]
        slice=slices[idx]
        prob_pe = slices_err[slices_err['slices'] == slice].predicted.values[0]
        
        pe_label=slices_err[slices_err['slices'] == slice].labels.values[0]
        title = f'index={idx}, pe_slice={pe_label}, prob pe 4 image={prob_lbl:.3f}, prob pe 4 slice={prob_pe:.3f}'
        show_image2(path, slice, title)
    else:
        plot_multi(study,slices2show, slices_err,prob,dict_idx)





def window(x, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

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


def show_image2(series, image, title, size=576):
    # Call the local dicom file
    folder_path = r"../../../input/train"
    file_path = os.path.join(folder_path, series)
    file_name = image + '.dcm'
    file_path = os.path.join(file_path, file_name)
    print(file_path)
    fig = plt.figure(figsize=(20, 20), dpi=300)
    
    ds = pydicom.dcmread(file_path)
    x1 = ds.pixel_array.astype(np.float32)
    x1 = x1*ds.RescaleSlope+ds.RescaleIntercept
    x = window(x1, WL=100, WW=700)
    #first_patient_pixels = transform_to_hu([ds])
    bbox = bbox_dict[image_dict[image]['series_id']]
    #x = x[bbox[1]+20:bbox[3]-20,bbox[0]+20:bbox[2]-20]
    x = x[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    x = cv2.resize(x, (size,size))
    plt.title(title, fontdict = {'fontsize' : 20})
    # fig.add_subplot(2, 2, i+1)
    plt.imshow(x, cmap=plt.cm.bone) #first_patient_pixels[0],
    # st.image(first_patient_pixels[0], clamp=True)
    st.pyplot(fig, dpi=300)
   
   
def prepare_image(series, image):
# Call the local dicom file 
    folder_path = r"../../../input/train"
    file_path = os.path.join(folder_path,series)
    file_name = image +'.dcm'
    file_path = os.path.join(file_path,file_name)
    #print(file_path)
    ds = pydicom.dcmread(file_path)
    #first_patient_pixels = transform_to_hu([ds])
    x1 = ds.pixel_array.astype(np.float32)
    x1 = x1*ds.RescaleSlope+ds.RescaleIntercept
    x = window(x1, WL=100, WW=700)
    
    return x#first_patient_pixels[0]

#grid plot
def plot_multi(study,slices, df_slices,prob_lbl, dict_index,n_col=4):
    n_col=min(2,len(slices))
    step=1+int((len(slices)-16)/16)
    num_slices=min([16,len(slices)])
    n_row=int(num_slices/n_col)
    st.write('step: ', step)#num of positive slices - ', len(slices))
    fig=plt.figure(figsize=(10 * n_col, 10 * n_row), dpi=300)
    #fig,ax = plt.subplots(n_row,n_col,figsize=(18,20))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.2)
    j=1
    
    for i in slices[0:step*n_col*n_row:step]:
        path=study+'//'+ dict_series[study]
        #diff_lbl=df_lbl[df_lbl['studies']==s].dif.values[0]
        #diff_pe=df_slice[df_slice['slices']==i].dif.values[0]
        prob_pe = df_slices[df_slices['slices'] == i].predicted.values[0]
        index=dict_index[i]
        pe_label=df_slices[df_slices['slices'] == i].labels.values[0]
        title = f'pe_label={pe_label}, prob pe 4 image={prob_lbl:.3f}, prob pe 4 slice={prob_pe:.3f} , index={index}'
        #st.write(title)
        plt.subplot(n_row, n_col, j)
        plt.title(title,  fontdict = {'fontsize' : 20})
        image=prepare_image(path,i)
        plt.imshow(image,cmap=plt.cm.bone)
        #ax[int(i/n_col),int(i % n_row)].imshow(image,cmap='bone')
        #ax[int(i/n_col),int(i % n_row)].axis('off')
        j=j+1
    st.pyplot(fig, dpi=300)

def load_slice(path,slices2show):
    slices = [pydicom.read_file(path + '/' + s+'.dcm') for s in slices2show]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
show_errors('negative_exam_for_pe')#chronic_pe') #acute_and_chronic_pe negative_exam_for_pe
    
    
    
    
    
    
    
    