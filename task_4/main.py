import pandas as pd
import os
import numpy as np

# Machine learning libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def rmse(y_pred, y_true):
    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state([y_pred, y_true])
    return m.result().numpy()

def encoder(X_train):
    model = Sequential()
    model.add(Dense(500, input_shape=X_train[0].shape))
    model.add(Activation("tanh"))
    model.add(Dropout(0.7))
    model.add(Dense(100))
    model.add(Activation("tanh"))
    model.add(Dense(21))
    model.add(Activation("tanh"))
    return model

def encoder_mk2(X_train):
    model = Sequential()
    model.add(Dense(500, input_shape=X_train[0].shape))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.2))
    model.add(keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
                    ))

    #model.add(Dense(50, input_shape=X_train[0].shape))
    #model.add(Activation("tanh"))

    return model

def lumu_reg(model):
    model.add(Dense(1))
    model.add(keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
                    ))
    return model

def honolulu_reg(inner_model):
    inner_model.trainable = False
    inner_model.add(Dense(1))
    inner_model.add(keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
                    ))
    return inner_model

def main():
    # Import
    pwd = os.getcwd()
    pretrain_features_df = pd.read_csv('data/pretrain_features.csv.zip', compression='zip')
    pretrain_labels_df = pd.read_csv('data/pretrain_labels.csv.zip', compression='zip')
    test_features_df = pd.read_csv('data/test_features.csv.zip', compression='zip')
    train_features_df = pd.read_csv('data/train_features.csv.zip', compression='zip')
    train_labels_df = pd.read_csv('data/train_labels.csv.zip', compression='zip')

    # MLP for lumo regression

    # X_train = tf.convert_to_tensor(pretrain_features_df.iloc[:, 1:], dtype=float)
    X_pretrain_ws = tf.convert_to_tensor(pretrain_features_df.iloc[:, 2:], dtype=float)
    y_pretrain = tf.convert_to_tensor(pretrain_labels_df.iloc[:, 1:], dtype=float)
    X_train_ws = tf.convert_to_tensor(train_features_df.iloc[:, 2:], dtype=float)
    y_train = tf.convert_to_tensor(train_labels_df.iloc[:, 1:], dtype=float)
    X_test_ws = tf.convert_to_tensor(test_features_df.iloc[:, 2:], dtype=float)

    id_test = pd.read_csv('data/test_features.csv.zip', compression='zip').iloc[:, 0:1]
    id_test = id_test.values.tolist()
    id_test = np.array(id_test, dtype=int)
    header = pd.read_csv('data/sample.csv', header=None).loc[0, :]  # extract test names for submission
    header = list(header)

    model = encoder_mk2(X_pretrain_ws)
    model_lumu = lumu_reg(model)

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model_lumu.compile(loss="mse",
                       optimizer=opt,
                       metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Fit model
    print(' \nTraining lumu model')
    model_lumu.fit(X_pretrain_ws, y_pretrain,
                   batch_size=100,
                   epochs=18,
                   verbose=1,
                   shuffle=True,
                   validation_split=0.1)

    print(' \nSaving model lumu')
    model.save('./model_lumu/')
    model_check = keras.models.load_model('./model_lumu/')


    model_honolulu = honolulu_reg(model_check)

    # Compile model
    model_honolulu.compile(loss="mse",
                           optimizer=opt,
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Fit model
    print(' \nTraining homo-lumu model with frozen weights')
    model_honolulu.fit(X_train_ws, y_train,
                       batch_size=1,
                       epochs=20,
                       verbose=1,
                       shuffle=True,
                       validation_split=0.1)

    print(' \nTraining homo-lumu model without frozen weights')
    model_honolulu.trainable = True
    model_honolulu.fit(X_train_ws, y_train,
                       batch_size=1,
                       epochs=20,
                       verbose=1,
                       shuffle=True,
                       validation_split=0.2)

    output = model.predict(X_test_ws)

    output_tot = id_test
    output_tot = np.append(output_tot, output, axis=1)

    df = pd.DataFrame(output_tot)
    df.iloc[:, 0] = df.iloc[:, 0].astype('int32')
    df.to_csv('new_submission.csv', index=False, header=header)

if __name__ == "__main__":
    main()