import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

DATA_DIR = "/root/kaggle/ventilator-pressure-prediction/data"

df_train = pd.read_csv(f"{DATA_DIR}/train.csv")

def add_features(df):

    df = df.loc[df['u_out'] != 1].copy()

    df['area'] = df['time_step'] * df['u_in']
    df['delta_time'] = df['time_step'].shift(-1, fill_value=0) - df['time_step']
    #df['area'] = df['delta_time'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()

    print("Step-1...Completed")

    # rewritten calculation of lag features from this notebook: https://www.kaggle.com/patrick0302/add-lag-u-in-as-new-feat
    # some of ideas from this notebook: https://www.kaggle.com/mst8823/google-brain-lightgbm-baseline
    df['last_value_u_in'] = df.groupby('breath_id')['u_in'].transform('last')
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1, fill_value=0)
    #df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1, fill_value=0)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1, fill_value=0)
    #df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1, fill_value=0)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2, fill_value=0)
    #df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2, fill_value=0)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2, fill_value=0)
    #df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2, fill_value=0)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3, fill_value=0)
    #df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3, fill_value=0)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3, fill_value=0)
    #df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3, fill_value=0)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4, fill_value=0)
    #df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4, fill_value=0)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4, fill_value=0)
    #df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4, fill_value=0)
    df = df.fillna(0)

    print("Step-2...Completed")

    # max value of u_in and u_out for each breath
    df['u_in_max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['u_in_mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['u_in_min'] = df.groupby(['breath_id'])['u_in'].transform('min')
    df['u_in_diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['u_in_diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['u_in_diffmin'] = df.groupby(['breath_id'])['u_in'].transform('min') - df['u_in']
    #df['u_out_max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    #df['u_out_min'] = df.groupby(['breath_id'])['u_out'].transform('min')

    print("Step-3...Completed")

    # difference between consequitive values
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff_back1'] = df['u_in_lag_back1'] - df['u_in']
    #df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    #df['u_out_diff_back1'] = df['u_out_lag_back1'] - df['u_out']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff_back2'] = df['u_in_lag_back2'] - df['u_in']
    #df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    #df['u_out_diff_back2'] = df['u_out_lag_back2'] - df['u_out']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff_back3'] = df['u_in_lag_back3'] - df['u_in']
    #df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    #df['u_out_diff_back3'] = df['u_out_lag_back3'] - df['u_out']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_in_diff_back4'] = df['u_in_lag_back4'] - df['u_in']
    #df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    #df['u_out_diff_back4'] = df['u_out_lag_back4'] - df['u_out']
    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    df.fillna(0)
    #df.loc[df['time_step'] == 0, 'u_in_diff1'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff_back1'] = 0
    #df.loc[df['time_step'] == 0, 'u_out_diff1'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff2'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff_back2'] = 0
    #df.loc[df['time_step'] == 0, 'u_out_diff2'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff3'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff_back3'] = 0
    #df.loc[df['time_step'] == 0, 'u_out_diff3'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff4'] = 0
    #df.loc[df['time_step'] == 0, 'u_in_diff_back4'] = 0
    #df.loc[df['time_step'] == 0, 'u_out_diff4'] = 0

    print("Step-4...Completed")

    df['area_abs'] = df['u_in_diff_back1'] * df['delta_time']
    df['uin_in_time'] = df['u_in_diff_back1'] / df['delta_time']

    print("Step-5...Completed")

    # difference between the current value of u_in/u_out and the max value within the breath
    df['u_in_diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['u_in_diffmin'] = df.groupby(['breath_id'])['u_in'].transform('min') - df['u_in']
    df['u_in_diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['u_in_diffmean'] = df.groupby(['breath_id'])['u_in'].transform('median') - df['u_in']

    print("Step-6...Completed")

    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']

    print("Step-7...Completed")

    # OHE
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = df.merge(pd.get_dummies(df['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
    df = df.merge(pd.get_dummies(df['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
    df = df.merge(pd.get_dummies(df['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

    print("Step-8...Completed")

    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')['u_in']\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    df[["15_in_sum","15_in_min","15_in_max","15_in_mean"]] = (df\
                                                              .groupby('breath_id')['u_in']\
                                                              .rolling(window=15,min_periods=1)\
                                                              .agg({"15_in_sum":"sum",
                                                                    "15_in_min":"min",
                                                                    "15_in_max":"max",
                                                                    "15_in_mean":"mean"})\
                                                               .reset_index(level=0,drop=True))

    print("Step-9...Completed")

    # https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
    df['u_in_cumsum'] = df.groupby(['breath_id'])['u_in'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / df['count']

    print("Step-10...Completed")

    #df['cross'] = df['u_in'] * df['u_out']
    #df['cross2'] = df['time_step'] * df['u_out']

    #print("Step-11...Completed")

    return df

df_train = add_features(df_train)

print(df_train.head())
print(df_train.columns.values.tolist())
print(df_train.isnull().all())

df_train.to_csv(f"{DATA_DIR}/preprocessed_train.csv", index=False)
