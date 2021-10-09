import numpy as np
import pandas as pd

import tensorflow as tf, gc
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

DATA_DIR = "/root/kaggle/ventilator-pressure-prediction/data"

train = pd.read_csv(f"{DATA_DIR}/preprocessed_train.csv")
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)

RS = RobustScaler()
train = RS.fit_transform(train)
#test = RS.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
#test = test.reshape(-1, 80, train.shape[-1])

EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

gpu_strategy = tf.distribute.get_strategy()

with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        K.clear_session()

        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]

        checkpoint_filepath = f"folds{fold}.hdf5"
        model = keras.models.Sequential([
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation='selu'),
            keras.layers.Dense(1),
            ])
        model.compile(optimizer="adam", loss="mae")

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(monitor="val_loss", patience=60, verbose=1, mode="min", restore_best_weights=True)
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es, sv])
        test_preds.append(model.predict(test, batch_size=BATCH_SIZE, verbose=2).squeeze().reshape(-1, 1).squeeze())

print(np.mean(test_preds, axis=1))