import numpy as np
import streamlit as st
import imageio
from PIL import Image

# getting the predictions
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
preds_tbl['err']=  (0.5 <=(preds_tbl.labels.values+preds_tbl.predicted.values )) & ((preds_tbl.labels.values+preds_tbl.predicted.values)< 1.5)
preds_tbl['err']=preds_tbl['err'].astype('int')
preds_tbl.loc[preds_tbl.types=='negative_exam_for_pe', 'labels']=1 -preds_tbl[preds_tbl.types=='negative_exam_for_pe'].labels.values
preds_tbl.loc[preds_tbl.types=='negative_exam_for_pe', 'predicted']=1 -preds_tbl[preds_tbl.types=='negative_exam_for_pe'].predicted.values

 

from sklearn.metrics import precision_score, recall_score,roc_auc_score, auc, precision_recall_curve, roc_curve, fbeta_score
def scores(label, thresh=0.5, verbose=True):
    pred_pos=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==1.0)].predicted.mean()
    total=preds_tbl[preds_tbl.types==label].shape[0]
    neg_pes=preds_tbl[(preds_tbl.types==label)]# & (err_tbl.diff<0.5) ]
    neg_pes_preds=neg_pes.predicted.values
    neg_pes_lbls=neg_pes.predicted.values >= thresh
    neg_pes_gt=neg_pes.labels.values
    num_pos=neg_pes_gt.sum()
    num_pred=neg_pes_lbls.sum()
    roc_auc=roc_auc_score(neg_pes_gt,neg_pes_preds, average='micro')
    recall=recall_score(neg_pes_gt,neg_pes_lbls)
    precision=precision_score(neg_pes_gt,neg_pes_lbls)
    precision1, recall1, _ = precision_recall_curve(neg_pes_gt,neg_pes_preds)
    auc2=auc(recall1, precision1)
    f2=fbeta_score(neg_pes_gt,neg_pes_lbls, beta=2)
    if verbose:
        print(f'Number of positive series - {num_pos} out of total {total}')
        print('auc for roc:', roc_auc)
        print('recall:', recall, ' predicted ', neg_pes_lbls.sum(), ' avg. pos prediction ', pred_pos)
        print('precision:', precision)
        #
        #recall, precision, _ = roc_curve(neg_pes_gt,neg_pes_preds)
        print('f_bta', f2)
        print('auc precision-recall', auc2)
    return [total, num_pos, roc_auc, precision, recall, num_pred,pred_pos,f2,auc2]


# In[211]:


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


# In[212]:


def fn_fp(label):
    df_fn=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==1.0) & (preds_tbl.predicted<0.5)]
    df_fp=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==0.0) & (preds_tbl.predicted>=0.5)]
    return df_fn, df_fp


# In[213]:


def analyze_label(label, thresh=0.5):
   
    df_fn, df_fp = fn_fp(label)
    #print(df_fp)
    _,_, slices_fn=slices4label(preds_tbl,label,df_fn,-9,-9)
    _,_, slices_fp=slices4label(preds_tbl,label,df_fp,-9,-9)
    scores(label, thresh)
   # fn_pred, fp_pred= fn_fp('pe_slice')
    #print(fp_pred)
    print(f'\nstats for {df_fn.shape[0]} false negative series')
    dict_fn=stats_label(slices_fn, label, df_fn)#,df_fn.shape[0],'blue')
    dict_pred=stats_label(preds_tbl[(preds_tbl.types=='pe_slice')],'pe_slice',preds_tbl[(preds_tbl.types==label)])
    plot_stat(dict_fn,dict_pred)
    print(f'stats for {df_fp.shape[0]} false positive series')
    stats_label(slices_fp, df_fp.shape[0])
    dict_fp=stats_label(slices_fp, label, df_fp)#, df_fp.shape[0],'red')
    #plot_stat(dict_fp,dict_pred)
    #_, df_lbl, slices_lbl= slices4label(preds_tbl, label,preds_tbl[(preds_tbl.types==label) &(preds_tbl.labels==0.0) ],-8,-9)
    #dict_lbl=stats_label(slices_lbl,label,df_lbl)#,preds_tbl[(preds_tbl.types==label) &(preds_tbl.labels==1.0) ])
    #plot_scatter(dict_lbl)
    return slices_fn, slices_fp#, dict_fn,dict_fp




def stats_label(slices_lbl, label, df_lbl=None,num_series=0):
    probs=None
    if not df_lbl is None:
        probs =df_lbl.predicted 
    ser_size=slices_lbl.groupby('studies').labels.size()#.mean()
    pe_freq=slices_lbl.groupby('studies').labels.mean()#.mean()
    
    d,=np.where(pe_freq==0)
    #print(slices_lbl.groupby('studies').index[d])
    #print(pe_freq.index.get_level_values(0)[d])
    #plt.plot(range(num_series),pe_freq)
    pe_max=slices_lbl.groupby('studies').predicted.max()
    #pe_freq.hist()
    err_freq=slices_lbl.groupby('studies').err.mean()#.mean()
    #sns.histplot(err_freq,stat='probability', color=color).set_title('mean error across studies')#' density=True, histtype='step')
    
    err_freq1=slices_lbl[slices_lbl.labels==1.0].groupby('studies').err.mean()#.mean()
    err_freq0=slices_lbl[slices_lbl.labels==0.0].groupby('studies').err.mean()#.mean()
    df=preds_tbl[preds_tbl.studies.isin(err_freq1.index.get_level_values(0))]
    
    #print(1-slices_lbl.groupby('studies').err.mean(), slices_lbl.groupby('studies').err.mean())
    pe_correct_freq=1-err_freq1#slices_lbl.groupby('studies').err.mean()
    prob_lbl = preds_tbl[(preds_tbl.types==label) &(preds_tbl.labels==1.0) ].predicted
    dict_stat={'pe_freq':pe_freq,'pe err_freq':err_freq,'err_freq for pe=0':err_freq0, 'err_freq for pe=1':err_freq1, 'max prob':pe_max}
    dict_stat['probs']=df[df.types==label].predicted#probs
    dict_stat['label']=label
    dict_stat['label predictions']=prob_lbl
    dict_stat['correct pe %'] = pe_correct_freq
    return dict_stat

def show_errors(label, bGif = False):
    df_pos=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==1.0)]
    df_neg=preds_tbl[(preds_tbl.types==label)&(preds_tbl.labels==0.0)]
    #dict_fn ,df_fn, slices_fn = slices4label(preds_tbl, label, df_pos, -9, -9)
    
    st.write('positive = show the slices with PE in PE-positive images')
    st.write('false positive = slices for which PE predicted > 0.5 in non-PE images')
    if st.radio('Choose', ['positive', 'false positive'])=='positive':
        df_err=df_pos
        types='positive'
    else:
        df_err=df_neg
        types='false positive'
    
    #filtering indeterminate studies
    indeter=preds_tbl[(preds_tbl.types=='indeterminate') & (preds_tbl.labels==1.0)].studies.values
    df_err=df_err[~df_err.studies.isin(indeter)]#stud_lbl=np.setdiff1d(stud_lbl, indeter)
    
    dict_err ,df,slices_err = slices4label(preds_tbl, label, df_err, -9, -9)
    prob=st.slider('PE probabilty predicted for image is smaller or equal to',0.0,1.0,0.05)
    df_sort = df_err[df_err.predicted <= prob].sort_values(by='predicted', ascending=False)
    #print(df_sort)
    studies=df_sort.studies.values
    study_idx= st.slider('Image (study) #', 0, len(studies), 0)  # min: 0h, max: 23h, default: 17h
    study=studies[study_idx]
        #study = st.text_input('study name', 'bc60cab42620')
    path = study + '//' + dict_series[study]
        
    folder_path = r"../../../input/train"
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
        #path = study + '//' + dict_series[study]
        
        #folder_path = r"../../../input/train"
        file_path = os.path.join(folder_path,path)

        first_patient= load_slice(file_path,slices2show)
        first_patient_pixels = transform_to_hu(first_patient)
        
        imageio.mimsave("/tmp/gif.gif", first_patient_pixels, duration=0.1)
        #image=Image.open("/tmp/gif.gif")
        st.image("/tmp/gif.gif")
    
    
    elif option=='every fifth slice':
        slice_idx=0
       # if st.button('next one'):

             
        while slice_idx<len(slices2show):
            
            slice = slices2show[slice_idx]
            idx=dict_idx[slice]
        # # file_name = slice +'.dcm'
            #path = study + '//' + dict_series[study]
        # # file_path = os.path.join(path,file_name)
        # prob_lbl = round(df_err[df_err['studies'] == study].predicted.values[0], 3)

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
        title=f'index:{idx}'
        show_image2(path, slice, title)
    else:
        plot_multi(study,slices2show, slices_err,prob,dict_idx)



from matplotlib import pyplot as plt
import os
import pydicom

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


def show_image2(series, image, title):
    # Call the local dicom file
    folder_path = r"../../../input/train"
    file_path = os.path.join(folder_path, series)
    file_name = image + '.dcm'
    file_path = os.path.join(file_path, file_name)
    print(file_path)
    fig = plt.figure(figsize=(20, 20), dpi=300)
    # fig, axs = plt.subplots(2, 3, figsize=(10, 3))
    # fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    # print(axs.shape)
    # for i in range(4):
    ds = pydicom.dcmread(file_path)
    x1 = ds.pixel_array.astype(np.float32)
    x1 = x1*ds.RescaleSlope+ds.RescaleIntercept
    img = window(x1, WL=100, WW=700)
    #first_patient_pixels = transform_to_hu([ds])
    plt.title(title, fontdict = {'fontsize' : 20})
    # fig.add_subplot(2, 2, i+1)
    plt.imshow(img, cmap=plt.cm.bone) #first_patient_pixels[0],
    # st.image(first_patient_pixels[0], clamp=True)
    st.pyplot(fig, dpi=300)
    # ax[1].imshow(first_patient_pixels[0],cmap=plt.cm.bone)
    # plt.show()
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
    img = window(x1, WL=100, WW=700)
    return img#first_patient_pixels[0]

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
    
# def show_study2(dict_slices, df_lbl, df_slice, study="", threshold=-9, num_studies=5, num_slices=12, n_col=4):
#     if threshold > 0:
#         studies = df_lbl[df_lbl.dif >= threshold].studies.values
#     else:
#         studies = df_lbl.studies.values
#     if study == "":
#         # study_idx= st.slider('Image (study)', 0, len(studies), 0)  # min: 0h, max: 23h, default: 17h
#         # study=studies[study_idx]
#         study = st.text_input('study name', 'bc60cab42620')
#         print(study)
#     try:

#         slices = dict_slices[study]
#     except:
#         st.write(study)

#         st.write('no erronious slices to show')
#         return

#     slice_idx = st.slider('Slice', 0, len(slices), 0)
#     slice = slices[slice_idx]
#     # file_name = slice +'.dcm'
#     path = study + '//' + dict_series[study]
#     # file_path = os.path.join(path,file_name)

#     diff_lbl = round(df_lbl[df_lbl['studies'] == study].dif.values[0], 3)
#     diff_pe = round(df_slice[df_slice['slices'] == slice].dif.values[0], 3)
#     title = f'diff label={diff_lbl}, diff pe={diff_pe}'
#     show_image2(path, slice, title)
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
show_errors('negative_exam_for_pe') #acute_and_chronic_pe
    
    
    
    
    
    
    
    