import pandas as pd
import numpy as np

DATA_DIR = "/root/kaggle/ventilator-pressure-prediction/data"

df_train = pd.read_csv(f"{DATA_DIR}/train.csv")

# rewritten calculation of lag features from this notebook: https://www.kaggle.com/patrick0302/add-lag-u-in-as-new-feat
# some of ideas from this notebook: https://www.kaggle.com/mst8823/google-brain-lightgbm-baseline
df_train['last_value_u_in'] = df_train.groupby('breath_id')['u_in'].transform('last')
df_train['u_in_lag1'] = df_train.groupby('breath_id')['u_in'].shift(1)
df_train['u_out_lag1'] = df_train.groupby('breath_id')['u_out'].shift(1)
df_train['u_in_lag_back1'] = df_train.groupby('breath_id')['u_in'].shift(-1)
df_train['u_out_lag_back1'] = df_train.groupby('breath_id')['u_out'].shift(-1)
df_train['u_in_lag2'] = df_train.groupby('breath_id')['u_in'].shift(2)
df_train['u_out_lag2'] = df_train.groupby('breath_id')['u_out'].shift(2)
df_train['u_in_lag_back2'] = df_train.groupby('breath_id')['u_in'].shift(-2)
df_train['u_out_lag_back2'] = df_train.groupby('breath_id')['u_out'].shift(-2)
df_train = df_train.fillna(0)

# max value of u_in and u_out for each breath
df_train['u_in_max'] = df_train.groupby(['breath_id'])['u_in'].transform('max')
df_train['u_out_max'] = df_train.groupby(['breath_id'])['u_out'].transform('max')
df_train['u_in_min'] = df_train.groupby(['breath_id'])['u_in'].transform('min')
df_train['u_out_min'] = df_train.groupby(['breath_id'])['u_out'].transform('min')

# difference between consequitive values
df_train['u_in_diff1'] = df_train['u_in'] - df_train['u_in_lag1']
df_train['u_out_diff1'] = df_train['u_out'] - df_train['u_out_lag1']
df_train['u_in_diff2'] = df_train['u_in'] - df_train['u_in_lag2']
df_train['u_out_diff2'] = df_train['u_out'] - df_train['u_out_lag2']
# from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
df_train.loc[df_train['time_step'] == 0, 'u_in_diff'] = 0
df_train.loc[df_train['time_step'] == 0, 'u_out_diff'] = 0

# difference between the current value of u_in/u_out and the max value within the breath
df_train['u_in_diffmax'] = df_train.groupby(['breath_id'])['u_in'].transform('max') - df_train['u_in']
df_train['u_in_diffmin'] = df_train.groupby(['breath_id'])['u_in'].transform('min') - df_train['u_in']
df_train['u_in_diffmean'] = df_train.groupby(['breath_id'])['u_in'].transform('mean') - df_train['u_in']
df_train['u_in_diffmean'] = df_train.groupby(['breath_id'])['u_in'].transform('median') - df_train['u_in']

# OHE
df_train['R__C'] = df_train["R"].astype(str) + '__' + df_train["C"].astype(str)
df_train = df_train.merge(pd.get_dummies(df_train['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
df_train = df_train.merge(pd.get_dummies(df_train['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
df_train = df_train.merge(pd.get_dummies(df_train['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

# https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
df_train['u_in_cumsum'] = df_train.groupby(['breath_id'])['u_in'].cumsum()
df_train['time_step_cumsum'] = df_train.groupby(['breath_id'])['time_step'].cumsum()

print(df_train)
print(df_train.columns.values.tolist())

df_train.to_csv(f"{DATA_DIR}/preprocessed_train.csv", index=False)
