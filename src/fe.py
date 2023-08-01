import pandas as pd
import numpy as np

from const import CUM_PREFIX_COLS


def revise_cum_col(df, make_grp_sum=True):
    df_cols = df.columns.to_list()
    for col in CUM_PREFIX_COLS:
        if f'{col}_norm' not in df_cols:
            col_name = col[4:]
            df[f'{col}_norm'] = df.groupby(['ad_id', 'date'])[
                col_name].cumsum()
            print(f'End Norm Feture: {col}')
    if make_grp_sum:
        for col in CUM_PREFIX_COLS:
            if f'{col}_norm_lk' not in df_cols:
                col_name = col[4:]
                mp_ = df.groupby(['ad_id', 'date'])[col_name].sum().to_dict()
                df[f'{col}_norm_lk'] = pd.Series(
                    zip(df['ad_id'], df['date'])).map(mp_)
                print(f'End Norm LKFeture: {col}')

    return df


def make_cumsum_day(df):
    df_cols = df.columns.to_list()
    if 'ad_online_days' not in df_cols:
        key_col = 'ad_id'
        min_date_dict = df.groupby(key_col)['date'].agg('min').to_dict()
        min_date_ss = df[key_col].map(min_date_dict)
        df['ad_online_days'] = (df['date'] - min_date_ss).dt.days

    if 'ad_online_days_refine' not in df_cols:
        df['ad_online_days_refine'] = 1
        df['ad_online_hours_refine'] = df.groupby(
            'ad_id')['ad_online_days_refine'].cumsum()
        df['ad_online_hours_refine'] = df.groupby(
            ['ad_id', 'hour'])['ad_online_days_refine'].cumsum()

    return df


def global_count_enc(df, col, prefix='COUNT_ENC_'):
    if f'{prefix}{col}' not in df.columns.to_list():
        map_ = df.groupby(col)['roi'].count().to_dict()
        df[f'{prefix}{col}'] = df[col].map(map_)
    return df


def category_combo_enc(df, pairs, prefix='COMB_ENC_'):
    df_cols = df.columns.to_list()
    for col1, col2 in pairs:
        name_lb = f'{prefix}{col1}_{col2}_LB'
        name_tg = f'{prefix}{col1}_{col2}_TG'

        if (name_lb in df_cols) and (name_tg in df_cols):
            continue
        else:
            ss = pd.Series(zip(df[col1], df[col2]))
            label_map_ = dict(zip(ss.unique(), range(ss.nunique())))
            target_map_ = df.groupby(ss)['roi'].mean().to_dict()
            df[f'{prefix}{col1}_{col2}_LB'] = ss.map(label_map_)
            df[f'{prefix}{col1}_{col2}_TG'] = ss.map(target_map_)

    return df


def target_enc(df, col, target='roi', prefix='TARGET_ENC_'):
    df_cols = df.columns.to_list()
    if f'{prefix}{col}' not in df_cols:
        df[col] = df[col].fillna("UNK")
        map_ = df.groupby(col)[target].agg('mean').to_dict()
        df[f'{prefix}{col}_{target}'] = df[col].map(map_).fillna(0)

    return df


def label_enc(df, col, prefix='LABEL_ENC_'):
    df_cols = df.columns.to_list()
    if f'{prefix}{col}' not in df_cols:
        df[col] = df[col].fillna("UNK")
        map_ = dict(zip(df[col].unique(), range(df[col].nunique())))
        df[f'{prefix}{col}'] = df[col].map(map_)

    return df


def split_seq_cal_len(df, col, sep=',', prefix='SEQ_CNT_'):
    df_cols = df.columns.to_list()
    if f'{prefix}{col}' not in df_cols:
        df[col] = df[col].fillna("UNK")
        df[f'{prefix}{col}'] = df[col].str.split(sep).apply(len)

    return df


def split_date_part(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.day_of_week
    df['decade_in_month'] = df['datetime'].dt.day // 10

    return df


def make_lag_feats(df, lag_cols, prefix='LAG_'):
    def lag_feats_op(df, col):
        for lag in [1, 2, 3]:
            df[f'{prefix}{col}_DIFF_{lag}'] = df[col].diff(lag)
            df[f'{prefix}{col}_SHIFT_{lag}'] = df[col].shift(lag)
        return df

    for col in lag_cols:
        df = df.groupby('ad_id').apply(lag_feats_op, col)

    return df


def make_last_day_half_roi(df, drop_last_day=False):
    df_cols = df.columns.to_list()
    if 'LAST_DAY_HALF_ROI' not in df_cols:
        tmp_df = df.copy()
        tmp_df = tmp_df[tmp_df['hour'] == 12].reset_index(drop=True)

        def last_day_roi_half_op(df):
            df['LAST_DAY_HALF_ROI'] =\
                (df['cum_income_1_norm'] + df['cum_income_2_norm']) / (0.001 + df['cum_spend_norm'])
            df['LAST_DAY_HALF_ROI_SHIFT_1'] = df['LAST_DAY_HALF_ROI'].shift(1)
            df['LAST_DAY_HALF_ROI_SHIFT_2'] = df['LAST_DAY_HALF_ROI'].shift(2)
            df['LAST_DAY_HALF_ROI_SHIFT_3'] = df['LAST_DAY_HALF_ROI'].shift(3)
            df['LAST_DAY_HALF_ROI_SHIFT_4'] = df['LAST_DAY_HALF_ROI'].shift(4)
            return df

        cols = ['ad_id', 'date', 'cum_income_1_norm',
                'cum_income_2_norm',  'cum_spend_norm']
        last_day_half_roi = tmp_df[cols].groupby(
            'ad_id').apply(last_day_roi_half_op)
        last_day_half_roi = last_day_half_roi[['ad_id',
                                               'date',
                                               'LAST_DAY_HALF_ROI',
                                               'LAST_DAY_HALF_ROI_SHIFT_1',
                                               'LAST_DAY_HALF_ROI_SHIFT_2',
                                               'LAST_DAY_HALF_ROI_SHIFT_3',
                                               'LAST_DAY_HALF_ROI_SHIFT_4']]
        df = df.merge(last_day_half_roi, on=['ad_id', 'date'], how='left')
        
        if drop_last_day:
            df = df.drop(['LAST_DAY_HALF_ROI'], axis=1)
    return df


def numeric_comb_fast(df):
    df['income'] = df['income_1'] + df['income_2']
    df['cum_income'] = df['cum_income_1'] + df['cum_income_2']
    df['cum_income_norm'] = df['cum_income_1_norm'] + df['cum_income_2_norm']

    df[f'income_cp_roi'] = df['income'] / (0.001 + df['spend'])
    df[f'cum_income_cp_roi_norm'] = df[f'cum_income_norm'] / (0.001 + df['cum_spend_norm'])
    df[f'cum_income_cp_roi'] = df[f'cum_income'] / (0.001 + df['cum_spend'])

    convert_cols = ['clicks', 'watches', 'purchase']
    for col in convert_cols:
        df[f'cum_{col}_tr_roi_norm'] = df[f'cum_{col}_norm'] / (1 + df['cum_impressions_norm'])

    for col in ['clicks', 'watches', 'purchase', 'spend', 'income']:
        df[f'avg_{col}_hour'] = df[f'cum_{col}_norm'] / (1 + df['hour'])
    df['avg_cum_income_cp_roi_norm_hour'] = df['cum_income_cp_roi_norm'] / (1 + df['hour'])
    return df


def cal_price(df):
    df['price'] = df['cum_income_1'] / (1 + df['cum_purchase'])
    price_dict = df.groupby('ad_id')['price'].max().to_dict()
    df['price'] = df['ad_id'].map(price_dict)

    return df
