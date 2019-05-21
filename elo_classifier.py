import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.listdir('../input'))
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error,roc_auc_score,roc_curve
from sklearn.model_selection import KFold, StratifiedKFold


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def remove_null_importance(feature_importance_df):
    to_drop=feature_importance_df.groupby('feature')['importance'].mean()<=1
    to_drop=to_drop[to_drop].index.tolist()
    return to_drop

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

def business_feats(hist_df):
    
    business_df=pd.DataFrame()
    value_counts=hist_df.groupby('month_lag')['card_id'].value_counts()
    nunique=hist_df.groupby('month_lag')['card_id'].nunique()
    total_count=hist_df['month_lag'].value_counts()
    business_df['LCR']=(value_counts>1).groupby(level=0).sum()/nunique
    business_df['repurchase_rate']=(value_counts>1).groupby(level=0).sum()/total_count
    business_df['purchase_rate']=total_count/nunique
    
    return business_df
    
def train_test(num_rows=None):

    # load csv
    train_df = pd.read_csv('../input/train.csv', index_col=['card_id'], nrows=num_rows)
    test_df = pd.read_csv('../input/test.csv', index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1

    # set target as nan
    test_df['target'] = np.nan

    # merge
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # to datetime
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']

    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    for f in ['feature_1','feature_2','feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    return df

def historical_transactions(num_rows=None):
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)
    
    # reduce memory usage
    #hist_df = reduce_mem_usage(hist_df)

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)
    
    

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    #Christmas : December 25 2017
    hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #2018
    #Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']
    
    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)
    
    ###########################
    hist_df=hist_df.merge(right=business_feats(hist_df),how='left',on='month_lag')
    print('Business features done')
    # group=business_feats(hist_df)
    # hist_df.set_index('month_lag',inplace=True)
    # hist_df[['LCR','repurchase_rate','purchase_rate']]=group
    # del group
    # gc.collect()
    # hist_df.reset_index(inplace=True)
    ###########################
    
    
    # first=pd.DataFrame()
    # first[['card_id','first_purchase_amount']]=hist_df[hist_df.groupby(['card_id'])['purchase_date'].transform(min) == hist_df['purchase_date']][['card_id','purchase_amount']]
    # first=first.groupby('card_id',as_index=False)['first_purchase_amount'].mean()
    # #first.set_index('card_id',inplace=True)
    # print('first done')

    # last=pd.DataFrame()
    # last[['card_id','last_purchase_amount']]=hist_df[hist_df.groupby(['card_id'])['purchase_date'].transform(max) == hist_df['purchase_date']][['card_id','purchase_amount']]
    # last=last.groupby('card_id',as_index=False)['last_purchase_amount'].mean()
    
    # print('last done')
    #last.set_index('card_id',inplace=True)
    


    
    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']
    
    aggs['days_bw_visits']=['mean','min','max','sum','var','skew']
    aggs['hours_bw_visits']=['mean','min','max','sum','var','skew']
    aggs['LCR']=['sum','max','min','mean','var','skew']
    aggs['repurchase_rate']=['sum','max','min','mean','var','skew']
    aggs['purchase_rate']=['sum','max','min','mean','var','skew']
    # aggs['first_purchase_amount']=['mean']
    # aggs['last_purchase_amount']=['mean']
    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['installments'] = ['sum','max','mean','var','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['max','min','mean','var','skew']
    aggs['authorized_flag'] = ['mean','sum']
    aggs['weekend'] = ['mean'] # overwrite
    aggs['weekday'] = ['mean'] # overwrite
    aggs['day'] = ['nunique', 'mean', 'min'] # overwrite
    aggs['category_1'] = ['mean','sum']
    aggs['category_2'] = ['mean','sum']
    aggs['category_3'] = ['mean','sum']
    aggs['card_id'] = ['size','count']
    aggs['price'] = ['sum','mean','max','min','var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']
    
    aggs_category={ 'purchase_amount': ['mean','min','max','sum'] }
    group=hist_df.groupby('cateogory_2').agg(aggs_category)
    group.columns=[col[0]+'_'+col[1] for col in group.columns]
    for col in ['category_2','category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    
    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)
    print('historical aggregation done')    
    ##################################################################
    

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_'+ c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today()-hist_df['hist_purchase_date_min']).dt.days
    
    return hist_df


def additional_features(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    # df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    # df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    # date_features=['hist_purchase_date_max','hist_purchase_date_min',
    #               'new_purchase_date_max', 'new_purchase_date_min']

    # for f in date_features:
    #     df[f] = df[f].astype(np.int64) * 1e-9

    # df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
    # df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']
    # df['card_id_cnt_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
    # df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    # df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    # df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
    # df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']
    # df['purchase_amount_ratio'] = df['new_purchase_amount_sum']/df['hist_purchase_amount_sum']
    # df['month_diff_mean'] = df['new_month_diff_mean']+df['hist_month_diff_mean']
    # df['month_diff_ratio'] = df['new_month_diff_mean']/df['hist_month_diff_mean']
    # df['month_lag_mean'] = df['new_month_lag_mean']+df['hist_month_lag_mean']
    # df['month_lag_max'] = df['new_month_lag_max']+df['hist_month_lag_max']
    # df['month_lag_min'] = df['new_month_lag_min']+df['hist_month_lag_min']
    # df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    # df['installments_total'] = df['new_installments_sum']+df['hist_installments_sum']
    # df['installments_mean'] = df['new_installments_mean']+df['hist_installments_mean']
    # df['installments_max'] = df['new_installments_max']+df['hist_installments_max']
    # df['installments_ratio'] = df['new_installments_sum']/df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean']+df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min']+df['hist_duration_min']
    df['duration_max'] = df['new_duration_max']+df['hist_duration_max']
    # df['amount_month_ratio_mean']=df['new_amount_month_ratio_mean']+df['hist_amount_month_ratio_mean']
    # df['amount_month_ratio_min']=df['new_amount_month_ratio_min']+df['hist_amount_month_ratio_min']
    # df['amount_month_ratio_max']=df['new_amount_month_ratio_max']+df['hist_amount_month_ratio_max']
    # df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df

def model(df_train,target,df_test,folds=5,rounds=100):
    param = {'num_leaves': 55,
             'min_data_in_leaf': 30, 
             'objective':'binary',
             'max_depth': 6,
             'learning_rate': 0.01,
             "boosting": "rf",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             #"metric": 'binary_logloss',
             "metric": 'auc',
             "lambda_l1": 0.1,
             "verbosity": -1,
             "random_state": 2333}
    
    print('Starting training with train shape = {}'.format(df_train.shape))
    print('Starting training with test shape = {}'.format(df_test.shape))
    
    folds = KFold(n_splits=folds, shuffle=True, random_state=15)
    oof = np.zeros(len(df_train))
    predictions = np.zeros(len(df_test))
    feature_importance_df = pd.DataFrame()
    
    features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','outliers']]
    df_train=df_train[features]
    df_test=df_test[features]
    categorical_feats = [c for c in features if 'feature_' in c]

    start = time.time()
    
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
        fold_+=1
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])
    
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds =rounds)
        oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits
    
    print("CV score: {:<8.5f}".format(roc_auc_score(target,oof)))
    
    
    return oof,predictions,feature_importance_df
    

#def main(debug=False):

debug=0
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
del hist_df
gc.collect()

train_df=pd.read_csv('../input/simple-lightgbm-without-blending/train.csv',nrows=num_rows)
test_df=pd.read_csv('../input/simple-lightgbm-without-blending/test.csv',nrows=num_rows)
target=train_df['outliers']
    
to_drop=['target','hist_purchase_date_max','hist_purchase_date_min','Unnamed: 0']
train_df.drop(to_drop,axis=1,inplace=True)
test_df.drop(to_drop,axis=1,inplace=True)

train_df=train_df.merge(right=groups,how='left',on='card_id')
test_df=test_df.merge(right=groups,how='left',on='card_id')
del groups
gc.collect()

#additional features
train_df['authorized_flag_thresold']=train_df['hist_authorized_flag_mean'].apply(lambda x:1 if x>0.82 else 0)
test_df['authorized_flag_thresold']=test_df['hist_authorized_flag_mean'].apply(lambda x:1 if x>0.82 else 0)
train_df['hist_month_nunique_thresold']=train_df['hist_month_nunique'].apply(lambda x:1 if x>=4 else 0)
test_df['hist_month_nunique_thresold']=test_df['hist_month_nunique'].apply(lambda x:1 if x>=4 else 0)

    
#with timer("Run LightGBM with kfold"):

to_drop=['feature_1',
 'feature_3',
 'feature_max',
 'feature_mean',
 'hist_Mothers_Day_2018_mean',
 'hist_days_bw_visits_min',
 'hist_hours_bw_visits_max',
 'hist_hours_bw_visits_min',
 'hist_month_lag_max',
 'hist_month_max',
 'hist_month_min',
 'hist_month_nunique_thresold',
 'hist_weekday_max',
 'hist_weekday_min',
 'hist_weekend_max',
 'hist_weekend_min',
 'hist_weekend_nunique',
 'last_month_hist_count_',
 'last_month_month_lag_max',
 'last_month_month_lag_min',
 'last_month_month_max',
 'last_month_month_nunique',
 'last_month_purchase_amount_max',
 'last_month_weekday_max',
 'last_month_weekend_max',
 'last_month_weekend_min',
 'last_month_weekend_nunique',
 'new_authorized_flag_mean',
 'new_authorized_flag_sum',
 'new_card_id_count',
 'new_card_id_size',
 'new_day_nunique',
 'new_merchant_id_nunique',
 'new_month_diff_var',
 'new_month_nunique',
 'new_weekday_nunique',
 'new_weekend_max',
 'new_weekend_mean',
 'new_weekend_min',
 'purchase_amount_mean',
 'hist_weekday_nunique',
 'authorized_flag_thresold',
 'new_city_id_nunique',
 'last_month_day_max',
 'hist_hour_min',
 'feature_2',
 'last_month_installments_max',
 'hist_purchase_amount_max',
 'last_month_weekday_min',
 'last_month_day_min',
 'new_category_2_sum',
 'new_installments_max',
 'hist_duration_var',
 'last_month_installments_var',
 'purchase_amount_max',
 'last_month_hour_min',
 'hist_hour_max',
 'new_month_lag_min',
 'new_category_2_mean_mean',
 'feature_min',
 'new_weekend_nunique',
 'hist_day_max',
 'new_month_diff_skew']

# train_df.drop(to_drop,axis=1,inplace=True)
# test_df.drop(to_drop,axis=1,inplace=True)
# print('Number of features dropped ={}'.format(len(to_drop)))
    

#  # Threshold for removing correlated variables
# threshold = 0.9
# corr_matrix = train_df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# print('There are %d columns to remove.' % (len(to_drop)))



# train_df.drop(to_drop,axis=1,inplace=True)
# test_df.drop(to_drop,axis=1,inplace=True)
# print('Number of features dropped ={}'.format(len(to_drop)))

cols=['card_id',
 'outliers',
 'first_active_month',
 'quarter',
 'elapsed_time',
 'days_feature1',
 'days_feature2',
 'days_feature3',
 'days_feature1_ratio',
 'days_feature2_ratio',
 'days_feature3_ratio',
 'feature_sum',
 'feature_var',
 'hist_subsector_id_nunique',
 'hist_merchant_id_nunique',
 'hist_month_nunique',
 'hist_month_mean',
 'hist_hour_nunique',
 'hist_hour_mean',
 'hist_weekofyear_nunique',
 'hist_weekofyear_min',
 'hist_weekofyear_max',
 'hist_weekday_mean',
 'hist_day_nunique',
 'hist_day_mean',
 'hist_day_min',
 'hist_weekend_mean',
 'hist_authorized_flag_mean',
 'hist_authorized_flag_sum',
 'hist_purchase_amount_min',
 'hist_purchase_amount_mean',
 'hist_purchase_amount_var',
 'hist_purchase_amount_skew',
 'hist_installments_sum',
 'hist_installments_max',
 'hist_installments_mean',
 'hist_installments_var',
 'hist_installments_skew',
 'hist_month_lag_min',
 'hist_month_lag_mean',
 'hist_month_lag_var',
 'hist_month_lag_skew',
 'hist_month_diff_mean',
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
 'hist_duration_mean',
 'hist_duration_max',
 'hist_amount_month_ratio_min',
 'hist_city_id_nunique',
 'hist_category_2_mean_mean',
 'hist_category_3_mean_mean',
 'hist_purchase_date_average',
 'hist_purchase_date_uptomin',
 'first_purchase_amount',
 'last_purchase_amount',
 'last_month_subsector_id_nunique',
 'last_month_merchant_id_nunique',
 'last_month_month_mean',
 'last_month_month_min',
 'last_month_hour_nunique',
 'last_month_hour_mean',
 'last_month_hour_max',
 'last_month_weekofyear_nunique',
 'last_month_weekofyear_max',
 'last_month_weekday_nunique',
 'last_month_weekday_mean',
 'last_month_day_mean',
 'last_month_weekend_mean',
 'last_month_authorized_flag_mean',
 'last_month_authorized_flag_sum',
 'last_month_purchase_amount_min',
 'last_month_purchase_amount_mean',
 'last_month_purchase_amount_var',
 'last_month_purchase_amount_skew',
 'last_month_installments_sum',
 'last_month_installments_mean',
 'last_month_installments_skew',
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
 'last_month_duration_max',
 'last_month_amount_month_ratio_min',
 'last_month_city_id_nunique',
 'last_month_category_3_mean_mean',
 'last_month_hist_days_elapsed',
 'last_month_unique_city_to_size',
 'last_month_unique_merchant_to_size',
 'new_subsector_id_nunique',
 'new_month_mean',
 'new_hour_mean',
 'new_hour_min',
 'new_hour_max',
 'new_weekofyear_nunique',
 'new_weekday_mean',
 'new_weekday_min',
 'new_weekday_max',
 'new_day_mean',
 'new_day_min',
 'new_day_max',
 'new_purchase_amount_max',
 'new_purchase_amount_min',
 'new_purchase_amount_mean',
 'new_purchase_amount_var',
 'new_purchase_amount_skew',
 'new_installments_sum',
 'new_installments_mean',
 'new_installments_var',
 'new_installments_skew',
 'new_purchase_date_max',
 'new_month_lag_max',
 'new_month_lag_mean',
 'new_month_lag_var',
 'new_month_lag_skew',
 'new_category_1_mean',
 'new_category_1_sum',
 'new_category_2_mean',
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
 'new_category_3_mean_mean',
 'new_purchase_date_diff',
 'new_purchase_date_average',
 'card_id_cnt_ratio',
 'purchase_amount_ratio',
 'month_diff_ratio',
 'month_lag_max',
 'category_1_mean',
 'installments_ratio',
 'price_total',
 'price_max',
 'duration_max',
 'new_CLV',
 'hist_CLV',
 'CLV_ratio',
 'hist_days_bw_visits_max',
 'hist_days_bw_visits_var',
 'hist_days_bw_visits_skew',
 'hist_hours_bw_visits_mean',
 'hist_hours_bw_visits_var']
 
 
train_df=train_df[cols]
test_df=test_df[cols]

category_feats_train=pd.read_csv('../input/category-aggregation/train.csv',nrows=num_rows)
category_feats_test=pd.read_csv('../input/category-aggregation/test.csv',nrows=num_rows)

train_df= train_df.merge(right=category_feats_train,how='left',on='card_id')
test_df= test_df.merge(right=category_feats_test,how='left',on='card_id')

del category_feats_train,category_feats_test
gc.collect()

 # Threshold for removing correlated variables
threshold = 0.9
corr_matrix = train_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

train_df.drop(to_drop,axis=1,inplace=True)
test_df.drop(to_drop,axis=1,inplace=True)

oof,pred,feat_imp =model(train_df,target,test_df,folds=5,rounds=200)
display_importances(feat_imp)

print('Writing train.csv')
train_df.to_csv('train.csv')
print('done train.csv')
print('Writing test.csv')
test_df.to_csv('test.csv')
print('done test.csv')
