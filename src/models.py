from copy import deepcopy
import random

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class LgbmEstimator():
    PARAMS = {
        'boosting_type': 'gbdt',
        'num_leaves': 2**7-1,
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'objective': 'mae',
        'subsample_freq': 1,
        'colsample_bytree': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.01,
        'importance_type': 'gain',
        'n_jobs': -1,
    }
    
    def __init__(self, params):
        self.params = deepcopy(self.PARAMS)
        self.params.update(params)
        
        self.model = None
        self.feat_imp = None
        
    def train(self, train_x, train_y, valid_x, valid_y, shuffle_params=False):
        if shuffle_params:
            self.params_shuffle(self.params['random_state'])
            
        sample_weights = (1 + train_x['hour'] / 13).values
        model = LGBMRegressor(**self.params)
        model.fit(train_x,
                  train_y,
                  sample_weight=sample_weights,
                  eval_set=(valid_x, valid_y),
                  eval_metric='mae',
                  early_stopping_rounds=50,
                  verbose=20)

        feat_imp = dict(zip(model.feature_name_, 
                            model.feature_importances_))
        feat_imp = dict(sorted(feat_imp.items(), 
                               key=lambda x: x[1], reverse=True))
        
        valid_preds = model.predict(valid_x, num_iteration=model._best_iteration)
        refine_valid_preds = np.where(valid_preds < 0, 0, valid_preds)
        valid_wmae = np.abs(valid_preds - valid_y) * (1 + valid_x['hour']) / 13
        refine_valid_wmae = np.abs(refine_valid_preds - valid_y) * (1 + valid_x['hour']) / 13

        print('valid wmae :', np.mean(valid_wmae))
        print('refine valid wmae :', np.mean(refine_valid_wmae))
        
        self.model = model
        self.feat_imp = feat_imp
    
    def predict(self, test_x):
        preds = self.model.predict(test_x, num_iteration=self.model._best_iteration)
        return preds
    
    def params_shuffle(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        shuffle_params = {
            'learning_rate': np.random.uniform(0.08, 0.15),
            'reg_alpha': np.random.uniform(0.08, 0.15),
            'max_depth': np.random.randint(7, 9),
            'colsample_bytree': np.random.uniform(0.8, 1)
        }
        
        self.params.update(shuffle_params)
        


class CgbEstimator():
    PARAMS =  {
        'depth': 8,
        'learning_rate': 0.12,
        'iterations': 1000,
        'objective': 'MAE',
        'eval_metric': 'MAE',
        'l2_leaf_reg': 0.1,
        'grow_policy': 'Lossguide'
    }
    
    def __init__(self, params):
        self.params = deepcopy(self.PARAMS)
        self.params.update(params)
        
        self.model = None
        self.feat_imp = None
        
    def train(self, train_x, train_y, valid_x, valid_y, shuffle_params=False):
        if shuffle_params:
            self.params_shuffle(self.params['random_state'])
        
        sample_weights = (1 + train_x['hour'] / 13).values
        model = CatBoostRegressor(**self.params)
        model.fit(train_x,
                  train_y,
                  sample_weight=sample_weights,
                  eval_set=(valid_x, valid_y),
                  early_stopping_rounds=50,
                  use_best_model=True,
                  verbose=20,)

        feat_imp = dict(zip(model.feature_names_, 
                            model.get_feature_importance()))
        feat_imp = dict(sorted(feat_imp.items(), 
                               key=lambda x: x[1], reverse=True))
        
        valid_preds = model.predict(valid_x)
        refine_valid_preds = np.where(valid_preds < 0, 0, valid_preds)
        valid_wmae = np.abs(valid_preds - valid_y) * (1 + valid_x['hour']) / 13
        refine_valid_wmae = np.abs(refine_valid_preds - valid_y) * (1 + valid_x['hour']) / 13

        print('valid wmae :', np.mean(valid_wmae))
        print('refine valid wmae :', np.mean(refine_valid_wmae))
        
        self.model = model
        self.feat_imp = feat_imp
    
    def predict(self, test_x):
        preds = self.model.predict(test_x)
        return preds
    
    def params_shuffle(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        shuffle_params = {
            'learning_rate': np.random.uniform(0.08, 0.15),
            'l2_leaf_reg': np.random.uniform(0.08, 0.15),
            'depth': np.random.randint(7, 9),
            'rsm': np.random.uniform(0.8, 1),
            'subsample': np.random.uniform(0.9, 1),
        }
        
        self.params.update(shuffle_params)
        
