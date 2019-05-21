import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def remove_null_importance(feature_importance_df):
    to_drop=feature_importance_df.groupby('feature')['importance'].mean()<=1
    to_drop=to_drop[to_drop].index.tolist()
    return to_drop

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def factorize(train,test):
    train=train.select_dtypes(exclude='<M8[ns]')
    test=test.select_dtypes(exclude='<M8[ns]')
    
    cat_cols=list(train.select_dtypes(include='object').columns)
    cat_cols.remove('card_id')
    
    for col in cat_cols:
        train[col]=pd.factorize(train[col])[0]
        test[col]=pd.factorize(test[col])[0]
        print(col)
    return train,test 
    

def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    target=train_df['target']
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()
    
    print('Average rmse for '+str(num_folds)+' folds = '+str(rmse(target, oof_preds)))
    # display importances
    display_importances(feature_importance_df)
    
    # save submission file
    test_df.loc[:,'target'] = sub_preds
    test_df = test_df.reset_index()
    test_df[['card_id', 'target']].to_csv('submission.csv', index=False)
    
    return feature_importance_df

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')



debug= 0
num_rows = 1000 if debug else None
hist_df=pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv',nrows=num_rows)

hist_df['purchase_date']=pd.to_datetime(hist_df['purchase_date'])
hist_df=hist_df.sort_values(by='purchase_date')
hist_df['purchase_date_shift']=hist_df.groupby('card_id')['purchase_date'].shift()
hist_df['days_bw_visits']=(hist_df['purchase_date']-hist_df['purchase_date_shift']).dt.days
hist_df['hours_bw_visits']=((hist_df['purchase_date']-hist_df['purchase_date_shift']).dt.seconds)//3600

aggs={}
aggs['days_bw_visits']=['mean','min','max','sum','var','skew']
aggs['hours_bw_visits']=['mean','min','max','sum','var','skew']

groups=hist_df.groupby('card_id').agg(aggs)
groups.columns=[ 'hist_'+col[0]+'_'+col[1] for col in groups.columns]

print('loading train and test set')
train_df=pd.read_csv('../input/simple-lightgbm-without-blending/train.csv',nrows=num_rows)
test_df=pd.read_csv('../input/simple-lightgbm-without-blending/test.csv',nrows=num_rows)

train_df=train_df[train_df['outliers']==0]
    

train_df=train_df.merge(right=groups,how='left',on='card_id')
test_df=test_df.merge(right=groups,how='left',on='card_id')
del groups
gc.collect()


#additional features
train_df['authorized_flag_thresold']=train_df['hist_authorized_flag_mean'].apply(lambda x:1 if x>0.82 else 0)
test_df['authorized_flag_thresold']=test_df['hist_authorized_flag_mean'].apply(lambda x:1 if x>0.82 else 0)
train_df['hist_month_nunique_thresold']=train_df['hist_month_nunique'].apply(lambda x:1 if x>=4 else 0)
test_df['hist_month_nunique_thresold']=test_df['hist_month_nunique'].apply(lambda x:1 if x>=4 else 0)


cols=[
 'card_id',
 'feature_1',
 'feature_2',
 'feature_3',
 'first_active_month',
 'outliers',
 'target',
 'quarter',
 'elapsed_time',
 'days_feature1',
 'days_feature2',
 'days_feature3',
 'days_feature1_ratio',
 'days_feature2_ratio',
 'days_feature3_ratio',
 'feature_sum',
 'feature_max',
 'feature_min',
 'feature_var',
 'hist_subsector_id_nunique',
 'hist_merchant_id_nunique',
 'hist_month_nunique',
 'hist_month_mean',
 'hist_month_min',
 'hist_month_max',
 'hist_hour_nunique',
 'hist_hour_mean',
 'hist_hour_min',
 'hist_hour_max',
 'hist_weekofyear_nunique',
 'hist_weekday_mean',
 'hist_weekday_min',
 'hist_day_nunique',
 'hist_day_mean',
 'hist_day_min',
 'hist_day_max',
 'hist_weekend_mean',
 'hist_authorized_flag_mean',
 'hist_authorized_flag_sum',
 'hist_purchase_amount_max',
 'hist_purchase_amount_min',
 'hist_purchase_amount_mean',
 'hist_purchase_amount_var',
 'hist_purchase_amount_skew',
 'hist_installments_sum',
 'hist_installments_max',
 'hist_installments_mean',
 'hist_installments_var',
 'hist_installments_skew',
 'hist_purchase_date_min',
 'hist_month_lag_max',
 'hist_month_lag_min',
 'hist_month_lag_mean',
 'hist_month_lag_var',
 'hist_month_lag_skew',
 'hist_month_diff_var',
 'hist_month_diff_skew',
 'hist_category_1_mean',
 'hist_category_1_sum',
 'hist_category_2_mean',
 'hist_category_2_sum',
 'hist_category_3_mean',
 'hist_price_max',
 'hist_price_min',
 'hist_price_var',
 'hist_Christmas_Day_2017_mean',
 'hist_Children_day_2017_mean',
 'hist_Black_Friday_2017_mean',
 'hist_Mothers_Day_2018_mean',
 'hist_duration_mean',
 'hist_duration_var',
 'hist_amount_month_ratio_min',
 'hist_city_id_nunique',
 'hist_category_2_mean_mean',
 'hist_category_3_mean_mean',
 'hist_purchase_date_average',
 'first_purchase_amount',
 'last_purchase_amount',
 'last_month_subsector_id_nunique',
 'last_month_merchant_id_nunique',
 'last_month_month_nunique',
 'last_month_month_mean',
 'last_month_month_min',
 'last_month_month_max',
 'last_month_hour_nunique',
 'last_month_hour_mean',
 'last_month_hour_min',
 'last_month_hour_max',
 'last_month_weekofyear_nunique',
 'last_month_weekday_nunique',
 'last_month_weekday_mean',
 'last_month_weekday_min',
 'last_month_weekday_max',
 'last_month_day_mean',
 'last_month_day_min',
 'last_month_day_max',
 'last_month_weekend_nunique',
 'last_month_weekend_mean',
 'last_month_authorized_flag_mean',
 'last_month_authorized_flag_sum',
 'last_month_purchase_amount_max',
 'last_month_purchase_amount_min',
 'last_month_purchase_amount_mean',
 'last_month_purchase_amount_var',
 'last_month_purchase_amount_skew',
 'last_month_installments_sum',
 'last_month_installments_max',
 'last_month_installments_mean',
 'last_month_installments_var',
 'last_month_installments_skew',
 'last_month_month_lag_min',
 'last_month_month_lag_mean',
 'last_month_month_lag_var',
 'last_month_month_lag_skew',
 'last_month_month_diff_var',
 'last_month_month_diff_skew',
 'last_month_category_1_sum',
 'last_month_category_2_sum',
 'last_month_price_max',
 'last_month_price_min',
 'last_month_price_var',
 'last_month_Christmas_Day_2017_mean',
 'last_month_Children_day_2017_mean',
 'last_month_Black_Friday_2017_mean',
 'last_month_Mothers_Day_2018_mean',
 'last_month_duration_min',
 'last_month_amount_month_ratio_min',
 'last_month_city_id_nunique',
 'last_month_category_3_mean_mean',
 'last_month_hist_days_elapsed',
 'last_month_unique_city_to_size',
 'last_month_unique_merchant_to_size',
 'new_subsector_id_nunique',
 'new_merchant_id_nunique',
 'new_month_nunique',
 'new_month_mean',
 'new_hour_mean',
 'new_hour_min',
 'new_hour_max',
 'new_weekofyear_nunique',
 'new_weekday_nunique',
 'new_weekday_mean',
 'new_weekday_min',
 'new_weekday_max',
 'new_day_mean',
 'new_day_min',
 'new_day_max',
 'new_weekend_nunique',
 'new_weekend_mean',
 'new_purchase_amount_max',
 'new_purchase_amount_min',
 'new_purchase_amount_mean',
 'new_purchase_amount_var',
 'new_purchase_amount_skew',
 'new_installments_sum',
 'new_installments_max',
 'new_installments_mean',
 'new_installments_var',
 'new_installments_skew',
 'new_purchase_date_max',
 'new_month_lag_max',
 'new_month_lag_min',
 'new_month_lag_mean',
 'new_month_lag_var',
 'new_month_lag_skew',
 'new_month_diff_var',
 'new_month_diff_skew',
 'new_category_1_mean',
 'new_category_1_sum',
 'new_category_2_sum',
 'new_price_mean',
 'new_price_max',
 'new_price_min',
 'new_price_var',
 'new_Christmas_Day_2017_mean',
 'new_Children_day_2017_mean',
 'new_Black_Friday_2017_mean',
 'new_Mothers_Day_2018_mean',
 'new_duration_mean',
 'new_duration_min',
 'new_amount_month_ratio_min',
 'new_amount_month_ratio_skew',
 'new_city_id_nunique',
 'new_category_2_mean_mean',
 'new_category_3_mean_mean',
 'new_purchase_date_diff',
 'new_purchase_date_average',
 'card_id_cnt_ratio',
 'purchase_amount_max',
 'purchase_amount_ratio',
 'month_diff_ratio',
 'month_lag_max',
 'category_1_mean',
 'installments_ratio',
 'price_total',
 'price_max',
 'new_CLV',
 'hist_CLV',
 'CLV_ratio',
 'hist_days_bw_visits_max',
 'hist_days_bw_visits_var',
 'hist_days_bw_visits_skew',
 'hist_hours_bw_visits_mean',
 'hist_hours_bw_visits_max',
 'hist_hours_bw_visits_var',
 'authorized_flag_thresold',
 'hist_month_nunique_thresold']
 


# train_df=train_df[cols]
# test_df=test_df[cols]

category_feats_train=pd.read_csv('../input/category-aggregation/train.csv',nrows=num_rows)
category_feats_test=pd.read_csv('../input/category-aggregation/test.csv',nrows=num_rows)

train_df= train_df.merge(right=category_feats_train,how='left',on='card_id')
test_df= test_df.merge(right=category_feats_test,how='left',on='card_id')

del category_feats_train,category_feats_test
gc.collect()

# threshold = 0.9
# corr_matrix = train_df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#print('There are %d columns to remove.' % (len(to_drop)))

cols=['card_id',
 'feature_1',
 'feature_2',
 'feature_3',
 'first_active_month',
 'outliers',
 'target',
 'quarter',
 'elapsed_time',
 'days_feature1',
 'days_feature2',
 'days_feature3',
 'days_feature1_ratio',
 'days_feature2_ratio',
 'days_feature3_ratio',
 'feature_sum',
 'feature_max',
 'feature_min',
 'feature_var',
 'hist_subsector_id_nunique',
 'hist_merchant_id_nunique',
 'hist_month_nunique',
 'hist_month_mean',
 'hist_month_min',
 'hist_month_max',
 'hist_hour_nunique',
 'hist_hour_mean',
 'hist_hour_min',
 'hist_hour_max',
 'hist_weekofyear_nunique',
 'hist_weekday_nunique',
 'hist_weekday_mean',
 'hist_weekday_min',
 'hist_weekday_max',
 'hist_day_nunique',
 'hist_day_mean',
 'hist_day_min',
 'hist_day_max',
 'hist_weekend_nunique',
 'hist_weekend_mean',
 'hist_weekend_min',
 'hist_authorized_flag_mean',
 'hist_authorized_flag_sum',
 'hist_purchase_amount_max',
 'hist_purchase_amount_min',
 'hist_purchase_amount_mean',
 'hist_purchase_amount_var',
 'hist_purchase_amount_skew',
 'hist_installments_sum',
 'hist_installments_max',
 'hist_installments_mean',
 'hist_installments_var',
 'hist_installments_skew',
 'hist_purchase_date_min',
 'hist_month_lag_max',
 'hist_month_lag_min',
 'hist_month_lag_mean',
 'hist_month_lag_var',
 'hist_month_lag_skew',
 'hist_month_diff_var',
 'hist_month_diff_skew',
 'hist_category_1_mean',
 'hist_category_1_sum',
 'hist_category_2_mean',
 'hist_category_2_sum',
 'hist_category_3_mean',
 'hist_price_max',
 'hist_price_min',
 'hist_price_var',
 'hist_Christmas_Day_2017_mean',
 'hist_Children_day_2017_mean',
 'hist_Black_Friday_2017_mean',
 'hist_Mothers_Day_2018_mean',
 'hist_duration_mean',
 'hist_duration_var',
 'hist_amount_month_ratio_min',
 'hist_city_id_nunique',
 'hist_category_2_mean_mean',
 'hist_category_3_mean_mean',
 'hist_purchase_date_average',
 'first_purchase_amount',
 'last_purchase_amount',
 'last_month_subsector_id_nunique',
 'last_month_merchant_id_nunique',
 'last_month_month_nunique',
 'last_month_month_mean',
 'last_month_month_min',
 'last_month_month_max',
 'last_month_hour_nunique',
 'last_month_hour_mean',
 'last_month_hour_min',
 'last_month_hour_max',
 'last_month_weekofyear_nunique',
 'last_month_weekday_nunique',
 'last_month_weekday_mean',
 'last_month_weekday_min',
 'last_month_weekday_max',
 'last_month_day_mean',
 'last_month_day_min',
 'last_month_day_max',
 'last_month_weekend_nunique',
 'last_month_weekend_mean',
 'last_month_weekend_min',
 'last_month_authorized_flag_mean',
 'last_month_authorized_flag_sum',
 'last_month_purchase_amount_max',
 'last_month_purchase_amount_min',
 'last_month_purchase_amount_mean',
 'last_month_purchase_amount_var',
 'last_month_purchase_amount_skew',
 'last_month_installments_sum',
 'last_month_installments_max',
 'last_month_installments_mean',
 'last_month_installments_var',
 'last_month_installments_skew',
 'last_month_month_lag_min',
 'last_month_month_lag_mean',
 'last_month_month_lag_var',
 'last_month_month_lag_skew',
 'last_month_month_diff_var',
 'last_month_month_diff_skew',
 'last_month_category_1_sum',
 'last_month_category_2_sum',
 'last_month_price_max',
 'last_month_price_min',
 'last_month_price_var',
 'last_month_Christmas_Day_2017_mean',
 'last_month_Children_day_2017_mean',
 'last_month_Black_Friday_2017_mean',
 'last_month_Mothers_Day_2018_mean',
 'last_month_duration_min',
 'last_month_amount_month_ratio_min',
 'last_month_city_id_nunique',
 'last_month_category_3_mean_mean',
 'last_month_hist_days_elapsed',
 'last_month_unique_city_to_size',
 'last_month_unique_merchant_to_size',
 'new_subsector_id_nunique',
 'new_merchant_id_nunique',
 'new_month_nunique',
 'new_month_mean',
 'new_hour_mean',
 'new_hour_min',
 'new_hour_max',
 'new_weekofyear_nunique',
 'new_weekday_nunique',
 'new_weekday_mean',
 'new_weekday_min',
 'new_weekday_max',
 'new_day_mean',
 'new_day_min',
 'new_day_max',
 'new_weekend_nunique',
 'new_weekend_mean',
 'new_weekend_min',
 'new_weekend_max',
 'new_authorized_flag_mean',
 'new_purchase_amount_max',
 'new_purchase_amount_min',
 'new_purchase_amount_mean',
 'new_purchase_amount_var',
 'new_purchase_amount_skew',
 'new_installments_sum',
 'new_installments_max',
 'new_installments_mean',
 'new_installments_var',
 'new_installments_skew',
 'new_purchase_date_max',
 'new_month_lag_max',
 'new_month_lag_min',
 'new_month_lag_mean',
 'new_month_lag_var',
 'new_month_lag_skew',
 'new_month_diff_var',
 'new_month_diff_skew',
 'new_category_1_mean',
 'new_category_1_sum',
 'new_category_2_mean',
 'new_category_2_sum',
 'new_price_mean',
 'new_price_max',
 'new_price_min',
 'new_price_var',
 'new_Christmas_Day_2017_mean',
 'new_Children_day_2017_mean',
 'new_Black_Friday_2017_mean',
 'new_Mothers_Day_2018_mean',
 'new_duration_mean',
 'new_duration_min',
 'new_amount_month_ratio_min',
 'new_amount_month_ratio_skew',
 'new_city_id_nunique',
 'new_category_2_mean_mean',
 'new_category_3_mean_mean',
 'new_purchase_date_diff',
 'new_purchase_date_average',
 'card_id_cnt_ratio',
 'purchase_amount_max',
 'purchase_amount_ratio',
 'month_diff_ratio',
 'month_lag_max',
 'category_1_mean',
 'installments_ratio',
 'price_total',
 'price_max',
 'new_CLV',
 'hist_CLV',
 'CLV_ratio',
 'category_1_Ypurchase_amount_mean',
 'category_1_Ypurchase_amount_min',
 'category_1_Npurchase_amount_mean',
 'category_1_Npurchase_amount_min',
 'category_2_1_purchase_amount_min',
 'category_2_2_purchase_amount_mean',
 'category_2_2_purchase_amount_sum',
 'category_2_2_purchase_amount_min',
 'category_2_3_purchase_amount_sum',
 'category_2_3_purchase_amount_min',
 'category_2_4_purchase_amount_mean',
 'category_2_4_purchase_amount_sum',
 'category_2_4_purchase_amount_min',
 'category_2_5_purchase_amount_mean',
 'category_2_5_purchase_amount_sum',
 'category_2_5_purchase_amount_min',
 'category_3_A_purchase_amount_min',
 'category_3_B_purchase_amount_mean',
 'category_3_B_purchase_amount_sum',
 'category_3_B_purchase_amount_min',
 'l_category_1_Ypurchase_amount_mean',
 'l_category_1_Ypurchase_amount_sum',
 'l_category_1_Ypurchase_amount_min',
 'l_category_1_Ypurchase_amount_max',
 'l_category_1_Ypurchase_amount_var',
 'l_category_1_Npurchase_amount_mean',
 'l_category_1_Npurchase_amount_sum',
 'l_category_1_Npurchase_amount_min',
 'l_category_2_1_purchase_amount_min',
 'l_category_2_2_purchase_amount_sum',
 'l_category_2_2_purchase_amount_min',
 'l_category_2_3_purchase_amount_min',
 'l_category_2_4_purchase_amount_mean',
 'l_category_2_4_purchase_amount_sum',
 'l_category_2_4_purchase_amount_min',
 'l_category_2_4_purchase_amount_var',
 'l_category_2_5_purchase_amount_mean',
 'l_category_2_5_purchase_amount_sum',
 'l_category_2_5_purchase_amount_min',
 'l_category_2_5_purchase_amount_var',
 'l_category_3_A_purchase_amount_sum',
 'l_category_3_B_purchase_amount_min',
 'l_category_3_C_purchase_amount_mean',
 'l_category_3_C_purchase_amount_sum',
 'l_category_3_C_purchase_amount_min',
 'l_category_3_C_purchase_amount_max',
 'l_category_3_C_purchase_amount_var',
 'm_category_1_Ypurchase_amount_mean',
 'm_category_1_Ypurchase_amount_var',
 'm_category_1_Npurchase_amount_mean',
 'm_category_1_Npurchase_amount_min',
 'm_category_1_Npurchase_amount_max',
 'm_category_1_Npurchase_amount_var',
 'm_category_2_1_purchase_amount_mean',
 'm_category_2_1_purchase_amount_sum',
 'm_category_2_1_purchase_amount_min',
 'm_category_2_1_purchase_amount_max',
 'm_category_2_2_purchase_amount_mean',
 'm_category_2_2_purchase_amount_sum',
 'm_category_2_2_purchase_amount_max',
 'm_category_2_2_purchase_amount_var',
 'm_category_2_3_purchase_amount_mean',
 'm_category_2_3_purchase_amount_sum',
 'm_category_2_3_purchase_amount_min',
 'm_category_2_3_purchase_amount_max',
 'm_category_2_3_purchase_amount_var',
 'm_category_2_4_purchase_amount_mean',
 'm_category_2_4_purchase_amount_sum',
 'm_category_2_4_purchase_amount_min',
 'm_category_2_4_purchase_amount_max',
 'm_category_2_4_purchase_amount_var',
 'm_category_2_5_purchase_amount_mean',
 'm_category_2_5_purchase_amount_sum',
 'm_category_2_5_purchase_amount_min',
 'm_category_2_5_purchase_amount_max',
 'm_category_2_5_purchase_amount_var',
 'm_category_3_A_purchase_amount_mean',
 'm_category_3_A_purchase_amount_max',
 'm_category_3_B_purchase_amount_mean',
 'm_category_3_B_purchase_amount_max',
 'm_category_3_B_purchase_amount_var',
 'm_category_3_C_purchase_amount_mean',
 'm_category_3_C_purchase_amount_sum',
 'm_category_3_C_purchase_amount_min',
 'm_category_3_C_purchase_amount_var']

train_df=train_df[cols]
test_df=test_df[cols]

feat_imp=kfold_lightgbm(train_df, test_df, num_folds=5, stratified=False, debug=debug)

print('writing train.csv')
train_df.to_csv('train.csv')
print('writing test.csv')
test_df.to_csv('test.csv')

    



