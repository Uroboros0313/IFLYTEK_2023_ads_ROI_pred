import os
import pathlib
import warnings
from itertools import product

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from models import LgbmEstimator, CgbEstimator
from fe import *
from const import *

warnings.filterwarnings('ignore')


def ensemble(save_dir, weight_dict=None):
    save_dir = pathlib.Path(save_dir)
    if weight_dict is None:
        weight_dict = {'CgbEstimator': 0.5, 
                       'LgbmEstimator': 0.5}
    
    fs = os.listdir(save_dir)
    f_pts = [save_dir / f for f in fs]
    
    uuid = pd.read_csv(f_pts[0])['uuid'].values
    preds_dict = {k: [] for k in weight_dict.keys()}
    
    for pt in f_pts:
        est_name = pt.stem.split('_')[0].split('-')[1]
        preds = pd.read_csv(pt)['roi'].values
        preds_dict[est_name].append(preds)
    
    wpreds_list = []
    for k in preds_dict.keys():
        wpreds_list.append(weight_dict[k] * np.mean(preds_dict[k], axis=0))
    final_preds = np.sum(wpreds_list, axis=0)
    
    submit = pd.DataFrame({
        'uuid': uuid,
        'roi': final_preds,
    })
    
    submit.to_csv(save_dir / 'ensemble.csv', index=False)
    
               
def get_k_fold_ests(all_df, fea_cols):
    ad_id_list = np.asarray(list(set(all_df['ad_id'])))
    kf = KFold(n_splits=10, random_state=SEED, shuffle=True)

    est_list = []
    for i, (trn_idxs, val_idxs) in enumerate(kf.split(ad_id_list)):
        train_ad_id_list = ad_id_list[trn_idxs]
        valid_ad_id_list = ad_id_list[val_idxs]
        train_df = all_df[all_df['ad_id'].isin(train_ad_id_list)].reset_index(drop=True)
        valid_df = all_df[all_df['ad_id'].isin(valid_ad_id_list)].reset_index(drop=True)

        train_x, train_y = train_df[fea_cols], train_df[LABEL]
        valid_x, valid_y = valid_df[fea_cols], valid_df[LABEL]
        
        lgbm_est = LgbmEstimator({'random_state': SEED+i})
        lgbm_est.train(train_x, train_y, valid_x, valid_y, shuffle_params=True)
        cgb_est = CgbEstimator({'random_state': SEED+i})
        cgb_est.train(train_x, train_y, valid_x, valid_y, shuffle_params=True)
        
        est_list.extend([lgbm_est, cgb_est])
    
    return est_list
        

def get_all_train_and_valid(all_df, fea_cols):
    ad_id_list = list(set(all_df['ad_id']))
    num_ads = len(ad_id_list)
    train_ad_id_list = ad_id_list[: int(num_ads*0.9)]
    valid_ad_id_list = ad_id_list[int(num_ads*0.9): ]

    train_df = all_df[all_df['ad_id'].isin(train_ad_id_list)].reset_index(drop=True)
    valid_df = all_df[all_df['ad_id'].isin(valid_ad_id_list)].reset_index(drop=True)
    
    train_x, train_y = train_df[fea_cols], train_df[LABEL]
    valid_x, valid_y = valid_df[fea_cols], valid_df[LABEL]
    
    return train_x, train_y, valid_x, valid_y

def get_test_result(test_df, fea_cols, model, save_path=None):
    test_df['uuid'] = test_df['uuid'].astype(int)
    test_df = test_df.sort_values('uuid').reset_index(drop=True)
    
    test_x = test_df[fea_cols]
    test_preds = model.predict(test_x)
    
    submit = pd.DataFrame({
        'uuid': test_df['uuid'].values,
        'roi': np.where(test_preds < 0, 0, test_preds),
    })
    
    if save_path is not None:
        submit.to_csv(save_path, index=False)
        
    return submit

def make_feats():
    if os.path.exists(ROOT / 'data/processed/user_featureB.pkl'):
        all_df = pd.read_pickle(ROOT / 'data/processed/user_featureB.pkl')
    else:
        df = pd.read_csv(ROOT / 'data/raw/train.csv')
        tdf = pd.read_csv(ROOT / 'data/raw/testB.csv')
        all_df = pd.concat([df, tdf]).reset_index(drop=True)
        
        all_df = split_date_part(all_df)
        all_df = revise_cum_col(all_df, True)
        all_df = all_df.sort_values(['ad_id', 'datetime']).reset_index(drop=True)
        
        # split countries list
        all_df = split_seq_cal_len(all_df, 'countries')
        # calculate online duration
        all_df = make_cumsum_day(all_df)
        # numeric combination feature
        all_df = numeric_comb_fast(all_df)
        # category combo target encoding
        combs = list(product(['account_id'], ['post_type', 'gender']))
        all_df = category_combo_enc(all_df, combs)
        # fake roi of last day
        #all_df = make_last_day_half_roi(all_df, True)
        # diff feats
        all_df = make_lag_feats(all_df, LAG_COLS)
        all_df = cal_price(all_df)
        # category target encoding
        #for col in TG_ENC_COLS:
        #    all_df = target_enc(all_df, col)
        
        all_df.to_pickle(ROOT / 'data/processed/user_featureB.pkl')
        
    return all_df


if __name__ == '__main__':
    all_df = make_feats()
    
    test_df = all_df[all_df[LABEL].isna()].reset_index(drop=True)
    all_df = all_df[all_df[LABEL].notna()]
    
    fea_cols = [col for col in all_df.columns.to_list() if col not in USELESS_COLS + [LABEL]]
    
    train_x, train_y, valid_x, valid_y = get_all_train_and_valid(all_df, fea_cols)
    
    lgbm_est = LgbmEstimator({'random_state': SEED})
    lgbm_est.train(train_x, train_y, valid_x, valid_y)
    
    est_list = [lgbm_est]
    for est in est_list:
        save_path = ROOT / 'data/submit/model-{}_seed-{}_B-test.csv'.format(est.__class__.__name__, est.params['random_state'])
        get_test_result(test_df, fea_cols, est, save_path)
        
    #ensemble(ROOT / 'data/submit')