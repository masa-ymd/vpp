import pandas as pd
import numpy as np
import os
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import pickle
import warnings
#warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

# environment setup
DATA_DIR = "/root/kaggle/ventilator-pressure-prediction/data"
MODEL_DIR = "/root/kaggle/ventilator-pressure-prediction/lgbm_models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# read data
df_train = pd.read_csv(f"{DATA_DIR}/preprocessed_train.csv")
#df_test = pd.read_csv(f"{DATA_DIR}/test.csv")

X = df_train.drop(['id', 'pressure'], axis=1)
y = df_train['pressure']

print(X.head())
print("---")
print(y.head())

catecorical_features = [
    'u_out', 
    'R_5', 'R_20', 'R_50',
    'C_10', 'C_20', 'C_50',
    'R__C_20__10', 'R__C_20__20', 'R__C_20__50', 'R__C_50__10', 'R__C_50__20',
    'R__C_50__50', 'R__C_5__10', 'R__C_5__20', 'R__C_5__50']

X[catecorical_features] = X[catecorical_features].astype('category')

#with open(f"{MODEL_DIR}/lgbm_best_params.pkl", mode='rb') as f:
#    params = pickle.load(f)

seed0=2021
params = {
    #'objective': 'rmse',
    'objective': 'regression',
    'metric': 'mean_absolute_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.25,
    'max_depth': -1,
    'max_bin':100,
    'min_data_in_leaf':500,
    'learning_rate': 0.05,
    'subsample': 0.72,
    'subsample_freq': 4,
    'feature_fraction': 0.5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    #'categorical_column':[0],
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'n_jobs':-1,
    'verbose': -1}

def abs(y_true, y_pred):
    return  np.abs(y_true - y_pred)

def feval_abs(preds, lgbm_train):
    labels = lgbm_train.get_label() # 真のラベルを取得
    #print(labels)
    return 'abs', abs(y_true = labels, y_pred = preds), False

# kfold
def create_folds(data, num_splits,target):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data[target], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


#df_train = create_folds(df_train, 5, 'target')
#kf = KFold(n_splits=5, random_state=seed0, shuffle=True)
kf = GroupKFold(n_splits=5)

oof = pd.DataFrame()                 # out-of-fold result
models = []                          # models
scores = 0.0                         # validation score

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y, groups=X['breath_id'])):
#for fold in range(5):

    print("Fold :", fold+1)

    # create dataset
    #X_train, y_train = X.loc[X['kfold']!=fold].copy(), y.loc[X['kfold']!=fold].copy()
    #X_valid, y_valid = X.loc[X['kfold']==fold].copy(), y.loc[X['kfold']==fold].copy()
    X_train, y_train = X.loc[trn_idx].copy(), y[trn_idx].copy()
    X_valid, y_valid = X.loc[val_idx].copy(), y[val_idx].copy()

    X_train = X_train.drop(['breath_id'], axis=1)
    X_valid = X_valid.drop(['breath_id'], axis=1)

    #X_train['u_out'] = X_train['u_out'].astype('category')
    #X_valid['u_out'] = X_valid['u_out'].astype('category')
    
    #RMSPE weight
    #weights = 1/np.square(y_train)
    #lgbm_train = lgbm.Dataset(X_train,y_train,weight = weights, categorical_feature = catecorical_features)
    lgbm_train = lgbm.Dataset(X_train, y_train, categorical_feature = catecorical_features)

    #weights = 1/np.square(y_valid)
    #lgbm_valid = lgbm.Dataset(X_valid,y_valid,reference = lgbm_train,weight = weights, categorical_feature = catecorical_features)
    lgbm_valid = lgbm.Dataset(X_valid, y_valid, reference = lgbm_train, categorical_feature = catecorical_features)
    
    # model 
    model = lgbm.train(params=params,
                      train_set=lgbm_train,
                      valid_sets=[lgbm_train, lgbm_valid],
                      num_boost_round=1000,         
                      #feval=feval_abs,
                      verbose_eval=100,
                      categorical_feature = catecorical_features,
                      early_stopping_rounds = 50,    
                     )
    
    # validation 
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    #res_abs = abs(y_true = y_valid, y_pred = y_pred)
    #print(f'Performance of the　prediction: , ABS: {res_abs}')

    #keep scores and models
    print(f"best_score: {model.best_score['valid_1']['l1']}")
    scores += model.best_score['valid_1']['l1'] / 5
    models.append(model)
    print("*" * 100)
    importance_df = pd.DataFrame(model.feature_importance(importance_type='gain'),index=X_train.columns.values.tolist(),columns=['importance']).sort_values('importance', ascending=False)
    print(importance_df)
    print("*" * 100)

    model_name = f"{MODEL_DIR}/lgbm_model_{fold}.pkl"
    pickle.dump(model, open(model_name, 'wb'))

print(scores)

#for fold, model in enumerate(models):
for fold in range(5):
    model_name = f"{MODEL_DIR}/lgbm_model_{fold}.pkl"
    loaded_model = pickle.load(open(model_name, 'rb'))
    print(loaded_model)
    #model.save_model(model_name, num_iteration=model.best_iteration)