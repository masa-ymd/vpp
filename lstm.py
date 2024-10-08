import numpy as np
import pandas as pd
import os
import logging
import pickle

import tensorflow as tf, gc
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, GRU, Multiply, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

DATA_DIR = "/root/kaggle/ventilator-pressure-prediction/data"
MODEL_DIR = "/root/kaggle/ventilator-pressure-prediction/lstm_models"
CHECKPOINT_DIR = "/root/kaggle/ventilator-pressure-prediction/lstm_models/checkpoint"
B_SIZE = 80

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

train = pd.read_csv(f"{DATA_DIR}/preprocessed_train.csv.zip")

print(train.head())

#train = pd.read_csv('/root/kaggle/ventilator-pressure-prediction/data/train.csv')

# Detect hardware, return appropriate distribution strategy
print(tf.version.VERSION)
tf.get_logger().setLevel(logging.ERROR)
try: # detect TPU
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    BATCH_SIZE = strategy.num_replicas_in_sync * 64
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
except ValueError: # detect GPU(s) and enable mixed precision
    strategy = tf.distribute.MirroredStrategy() # works on GPU and multi-GPU
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.config.optimizer.set_jit(True) # XLA compilation
    tf.keras.mixed_precision.experimental.set_policy(policy)
    #BATCH_SIZE = 1280
    print('Mixed precision enabled')
print("REPLICAS: ", strategy.num_replicas_in_sync)

targets = train[['pressure']].to_numpy().reshape(-1, B_SIZE)
#train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
train.drop(['pressure','id', 'breath_id','one','count',
            'breath_id_lag','breath_id_lag2','breath_id_lagsame',
            'breath_id_lag2same'], axis=1, inplace=True)

# fillna
#for col in train.columns.to_list():
#    train[col] = train[col].fillna(train[col].mean())

RS = RobustScaler()
RS.fit(train)
#test = RS.transform(test)
scaler_name = f"{MODEL_DIR}/scaler.pkl"
with open(scaler_name, 'wb') as f:
    pickle.dump(RS, f)

with open(scaler_name, 'rb') as f:
    RS = pickle.load(f)
train = RS.transform(train)

train = train.reshape(-1, B_SIZE, train.shape[-1])
#test = test.reshape(-1, 80, train.shape[-1])

def dnn_model():
    
    x_input = Input(shape=(train.shape[-2:]))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z2 = Bidirectional(GRU(units=384, return_sequences=True))(x2)
    
    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = Bidirectional(GRU(units=256, return_sequences=True))(z31)
    
    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = Bidirectional(GRU(units=128, return_sequences=True))(z41)
    
    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = Bidirectional(GRU(units=64, return_sequences=True))(z51)
    
    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')

    model = keras.models.Model(inputs=x_input, outputs=x_output, name='DNN_Model')
    return model

EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

gpu_strategy = tf.distribute.get_strategy()

with strategy.scope():
#with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        K.clear_session()

        if os.path.exists(f"{MODEL_DIR}/lstm_model_{fold+1}.h5"):
            print(f"{MODEL_DIR}/lstm_model_{fold+1}.h5 exists. skip fold {fold+1}")
            continue

        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]

        checkpoint_filepath = f"{CHECKPOINT_DIR}/folds_{fold+1}.hdf5"
        #model = keras.models.Sequential([
        #    keras.layers.Input(shape=train.shape[-2:]),
        #    keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
        #    keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
        #    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
        #    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        #    keras.layers.Dense(128, activation='selu'),
        #    keras.layers.Dense(1),
        #    ])
        model = dnn_model()
        model.compile(optimizer="adam", loss="mae")

        if fold == 0:
            print(model.summary())

        if os.path.exists(checkpoint_filepath):
            print(f"load weights: {checkpoint_filepath}")
            model.load_weights(checkpoint_filepath)

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="min", restore_best_weights=True)
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es, sv])
        model.save(f"{MODEL_DIR}/lstm_model_{fold+1}.h5")

