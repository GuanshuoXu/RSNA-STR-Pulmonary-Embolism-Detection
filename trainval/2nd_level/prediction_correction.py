import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm

def correct_predictions(pred_prob_list, series_len_list):
    eps = 0.000001
    pred_prob_list_corrected = np.zeros(pred_prob_list.shape, dtype=np.float32)
    start = 0
    for i in tqdm(range(len(series_len_list))):
        end = series_len_list[i]
        negative_exam_for_pe = pred_prob_list[start+0]
        indeterminate = pred_prob_list[start+1]
        chronic_pe = pred_prob_list[start+2]
        acute_and_chronic_pe = pred_prob_list[start+3]
        central_pe = pred_prob_list[start+4]
        leftsided_pe = pred_prob_list[start+5]
        rightsided_pe = pred_prob_list[start+6]
        rv_lv_ratio_gte_1 = pred_prob_list[start+7]
        rv_lv_ratio_lt_1 = pred_prob_list[start+8]
        image_pe = pred_prob_list[start+9:end]

        loss_weight_list = np.zeros(pred_prob_list[start:end].shape, dtype=np.float32)
        loss_weight_list[0] = 0.0736196319
        loss_weight_list[1] = 0.09202453988
        loss_weight_list[2] = 0.1042944785
        loss_weight_list[3] = 0.1042944785
        loss_weight_list[4] = 0.1877300613
        loss_weight_list[5] = 0.06257668712
        loss_weight_list[6] = 0.06257668712
        loss_weight_list[7] = 0.2346625767
        loss_weight_list[8] = 0.0782208589
        loss_weight_list[9:] = 0.07361963*0.005
        
        if (np.amax(image_pe)<=0.5) and (int(negative_exam_for_pe>0.5)+int(indeterminate>0.5)==1) and (int(chronic_pe>0.5)+int(acute_and_chronic_pe>0.5)==0) and (int(central_pe>0.5)+int(leftsided_pe>0.5)+int(rightsided_pe>0.5)==0) and (int(rv_lv_ratio_gte_1>0.5)+int(rv_lv_ratio_lt_1>0.5)==0):
            pred_prob_list_corrected[start:end] = pred_prob_list[start:end]
        elif (np.amax(image_pe)>0.5) and (int(negative_exam_for_pe>0.5)+int(indeterminate>0.5)==0) and (int(chronic_pe>0.5)+int(acute_and_chronic_pe>0.5)<2) and (int(central_pe>0.5)+int(leftsided_pe>0.5)+int(rightsided_pe>0.5)>0) and (int(rv_lv_ratio_gte_1>0.5)+int(rv_lv_ratio_lt_1>0.5)==1):
            pred_prob_list_corrected[start:end] = pred_prob_list[start:end]
        else:
            to_neg = pred_prob_list[start:end].copy()
            for n in range(len(image_pe)):
                if image_pe[n]>0.5:
                    to_neg[9+n] = 0.5
            if negative_exam_for_pe>0.5 and indeterminate>0.5:
                if negative_exam_for_pe>indeterminate:
                    to_neg[1] = 0.5
                else:
                    to_neg[0] = 0.5
            elif negative_exam_for_pe<=0.5 and indeterminate<=0.5:
                if negative_exam_for_pe>indeterminate:
                    to_neg[0] = 0.5+eps
                else:
                    to_neg[1] = 0.5+eps
            if chronic_pe>0.5:
                to_neg[2] = 0.5
            if acute_and_chronic_pe>0.5:
                to_neg[3] = 0.5
            if central_pe>0.5:
                to_neg[4] = 0.5
            if leftsided_pe>0.5:
                to_neg[5] = 0.5
            if rightsided_pe>0.5:
                to_neg[6] = 0.5
            if rv_lv_ratio_gte_1>0.5:
                to_neg[7] = 0.5
            if rv_lv_ratio_lt_1>0.5:
                to_neg[8] = 0.5

            to_pos = pred_prob_list[start:end].copy()
            if np.amax(image_pe)<=0.5:
                max_idx = np.argmax(image_pe)
                to_pos[9+max_idx] = 0.5+eps
            if negative_exam_for_pe>0.5:
                to_pos[0] = 0.5
            if indeterminate>0.5:
                to_pos[1] = 0.5
            if chronic_pe>0.5 and acute_and_chronic_pe>0.5:
                if chronic_pe>acute_and_chronic_pe:
                    to_pos[3] = 0.5
                else:
                    to_pos[2] = 0.5
            if central_pe<=0.5 and leftsided_pe<=0.5 and rightsided_pe<=0.5:
                if central_pe>leftsided_pe and central_pe>rightsided_pe:
                    to_pos[4] = 0.5+eps
                if leftsided_pe>central_pe and leftsided_pe>rightsided_pe:
                    to_pos[5] = 0.5+eps
                if rightsided_pe>central_pe and rightsided_pe>leftsided_pe:
                    to_pos[6] = 0.5+eps
            if rv_lv_ratio_gte_1>0.5 and rv_lv_ratio_lt_1>0.5:
                if rv_lv_ratio_gte_1>rv_lv_ratio_lt_1:
                    to_pos[8] = 0.5
                else:
                    to_pos[7] = 0.5
            elif rv_lv_ratio_gte_1<=0.5 and rv_lv_ratio_lt_1<=0.5:
                if rv_lv_ratio_gte_1>rv_lv_ratio_lt_1:
                    to_pos[7] = 0.5+eps
                else:
                    to_pos[8] = 0.5+eps

            loss_weight_list1 = torch.tensor(loss_weight_list, dtype=torch.float32)
            pred_prob_list1 = torch.tensor(pred_prob_list[start:end], dtype=torch.float32)
            pred_prob_list_neg = torch.tensor(to_neg, dtype=torch.float32)
            pred_prob_list_pos = torch.tensor(to_pos, dtype=torch.float32)
            #print(loss_weight_list1.shape, pred_prob_list1.shape, pred_prob_list_neg.shape, pred_prob_list_pos.shape)
            to_neg_loss = ((torch.nn.BCELoss(reduction='none')(pred_prob_list1, pred_prob_list_neg)*loss_weight_list1).sum() / loss_weight_list1.sum()).numpy()
            to_pos_loss = ((torch.nn.BCELoss(reduction='none')(pred_prob_list1, pred_prob_list_pos)*loss_weight_list1).sum() / loss_weight_list1.sum()).numpy()

            if to_neg_loss>to_pos_loss:
                pred_prob_list_corrected[start:end] = to_pos
            else:
                pred_prob_list_corrected[start:end] = to_neg

        start = series_len_list[i]
    return pred_prob_list_corrected

def check_consistency(sub, test):
    
    '''
    Checks label consistency and returns the errors
    
    Args:
    sub   = submission dataframe (pandas)
    test  = test.csv dataframe (pandas)
    '''
    
    # EXAM LEVEL
    for i in test['StudyInstanceUID'].unique():
        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)
        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]
        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))
        del df_tmp['id']
        if i == test['StudyInstanceUID'].unique()[0]:
            df = df_tmp.copy()
        else:
            df = pd.concat([df, df_tmp], axis = 0)
    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')
    
    # IMAGE LEVEL
    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)
    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)
    del df_image['id']
    
    # MERGER
    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')
    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]
    
    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())
    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'
    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'
    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'
    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    # OUTPUT
    print('Found', len(errors), 'inconsistent predictions')
    return errors

gt_list_seresnext50_128 = np.load('predictions/gt_list_seresnext50_128.npy')
loss_weight_list_seresnext50_128 = np.load('predictions/loss_weight_list_seresnext50_128.npy')
pred_prob_list_seresnext50_128 = np.load('predictions/pred_prob_list_seresnext50_128.npy')
# gt_list_seresnext101_128 = np.load('predictions/gt_list_seresnext101_128.npy')
# loss_weight_list_seresnext101_128 = np.load('predictions/loss_weight_list_seresnext101_128.npy')
# pred_prob_list_seresnext101_128 = np.load('predictions/pred_prob_list_seresnext101_128.npy')

# gt_list_seresnext50_192 = np.load('predictions/gt_list_seresnext50_192.npy')
# loss_weight_list_seresnext50_192 = np.load('predictions/loss_weight_list_seresnext50_192.npy')
# pred_prob_list_seresnext50_192 = np.load('predictions/pred_prob_list_seresnext50_192.npy')
# gt_list_seresnext101_192 = np.load('predictions/gt_list_seresnext101_192.npy')
# loss_weight_list_seresnext101_192 = np.load('predictions/loss_weight_list_seresnext101_192.npy')
# pred_prob_list_seresnext101_192 = np.load('predictions/pred_prob_list_seresnext101_192.npy')

gt_list = gt_list_seresnext50_128
loss_weight_list = loss_weight_list_seresnext50_128
pred_prob_list = (pred_prob_list_seresnext50_128 )#+ pred_prob_list_seresnext101_128) / 2.0

gt_list1 = torch.tensor(gt_list, dtype=torch.float32)
loss_weight_list1 = torch.tensor(loss_weight_list, dtype=torch.float32)
pred_prob_list1 = torch.tensor(pred_prob_list, dtype=torch.float32)
kaggle_loss = torch.nn.BCELoss(reduction='none')(pred_prob_list1, gt_list1)
kaggle_loss = (kaggle_loss*loss_weight_list1).sum() / loss_weight_list1.sum()
print(kaggle_loss.numpy())

series_len_list = np.load('predictions/series_len_list.npy')
pred_prob_list = correct_predictions(pred_prob_list, series_len_list)

gt_list1 = torch.tensor(gt_list, dtype=torch.float32)
loss_weight_list1 = torch.tensor(loss_weight_list, dtype=torch.float32)
pred_prob_list1 = torch.tensor(pred_prob_list, dtype=torch.float32)
kaggle_loss = torch.nn.BCELoss(reduction='none')(pred_prob_list1, gt_list1)
kaggle_loss = (kaggle_loss*loss_weight_list1).sum() / loss_weight_list1.sum()
print(kaggle_loss.numpy())

id_list = np.load('predictions/id_list.npy')
train_df = pd.read_csv('../../input/train.csv')
study_id_list = train_df['StudyInstanceUID'].values
series_id_list = train_df['SeriesInstanceUID'].values
image_id_list = train_df['SOPInstanceUID'].values
with open('../process_input/split2/series_list_valid.pickle', 'rb') as f:
    series_list_valid = pickle.load(f)
for i in range(len(series_list_valid)):
    series_list_valid[i] = series_list_valid[i].split('_')[0]
series_list_valid = set(series_list_valid)
valid_study_id_list = []
valid_series_id_list = []
valid_image_id_list = []
for i in range(len(study_id_list)):
    if study_id_list[i] in series_list_valid:
        valid_study_id_list.append(study_id_list[i])
        valid_series_id_list.append(series_id_list[i])
        valid_image_id_list.append(image_id_list[i])
valid_df = pd.DataFrame(data={'StudyInstanceUID': valid_study_id_list, 'SeriesInstanceUID': valid_series_id_list, 'SOPInstanceUID': valid_image_id_list})
sub_df = pd.DataFrame(data={'id': id_list, 'label': pred_prob_list})
print(len(id_list), len(series_len_list), len(sub_df), len(valid_df))

errors = check_consistency(sub_df, valid_df)
