import joblib
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import pandas as pd
import numpy as np


def get_data_weights(weight_dict, y):
    if weight_dict is None:
        weights = None
    else:
        weights = np.empty_like(y)
        weights[y == 0] = weight_dict[0]
        weights[y == 1] = weight_dict[1]

    return weights


def model_evaluate(y_score, y, weight_dict):
    weights = get_data_weights(weight_dict, y)
    auc = roc_auc_score(y, y_score, sample_weight=weights)
    loss = log_loss(y, y_score, 10e-8, sample_weight=weights)

    return auc, loss


def logistic_regression(params):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=1,
              kernel_regularizer=params["regularizer"])(inputs)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=params["loss"],
                  optimizer=params["optimizer"])

    return model


def classic_mlp(params, input_dim, summary=False):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for i in range(len(params["hidden_width"])):
        x = Dense(
            units=params["hidden_width"][i],
            kernel_regularizer=params["regularizer"],
            kernel_initializer=params["initializer"])(x)
        if params["batchnorm"]:
            x = BatchNormalization()(x)
        x = Activation(params["hidden_activation"])(x)
        x = Dropout(params["dropout"][i])(x)

    x = Dense(units=1, kernel_regularizer=params["regularizer"])(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=params["loss"],
                  optimizer=params["optimizer"])

    return model


if __name__ == "__main__":
    ################################################################
    # Data
    data = np.load("../data/creditcard_train.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    input_dim = x_train.shape[1]
    df = pd.read_csv('../data/creditcard_test.csv')
    data = df.drop(['Unnamed: 0', 'Time'], 1).as_matrix()
    x_test, y_test = data[:, :-1], data[:, -1]

    ###############################################################
    nlayers = 3
    mlp_params = {
        "loss": "binary_crossentropy",
        "hidden_width": [100] * nlayers,
        "hidden_activation": "relu",
        "regularizer": l2(),
        "dropout": [0.5] * nlayers,
        "optimizer": Adam(),
        "initializer": "he_normal",
        "batchnorm": True,
        "batch_size": 1024
    }

    logistic_params = {
        "loss": "binary_crossentropy",
        "optimizer": Adam(),
        "regularizer": l2(),
    }

    mlp_best = "../mlp_model/model.auc:0.987883.lr:0.003305.reg:0.001552.hdf5"
    logistic_best = "../logistic_model/model.auc:0.982363." + \
        "lr:0.009312.reg:0.023246.hdf5"
    svm_best = "../svm_model/model.auc:0.980753.loss:0.004176.C:0.5000." + \
        "gamma:0.0010.pkl"

    print('Starting MLP predictions')
    mlp_model = classic_mlp(mlp_params, input_dim)
    mlp_model.load_weights(mlp_best)
    y_score = mlp_model.predict(x_train)
    np.save('../outputs/mlp_train', y_score)
    y_score = mlp_model.predict(x_val)
    np.save('../outputs/mlp_val', y_score)
    y_score = mlp_model.predict(x_test)
    np.save('../outputs/mlp_test', y_score)

    print('Starting Logistic Regression predictions')
    logistic_model = logistic_regression(logistic_params)
    logistic_model.load_weights(logistic_best)
    y_score = logistic_model.predict(x_train)
    np.save('../outputs/logistic_train', y_score)
    y_score = logistic_model.predict(x_val)
    np.save('../outputs/logistic_val', y_score)
    y_score = logistic_model.predict(x_test)
    np.save('../outputs/logistic_test', y_score)

    print('Starting SVM predictions')
    svm_model = joblib.load(svm_best)
    y_score = svm_model.predict_proba(x_train)[:, 1]
    np.save('../outputs/svm_train', y_score)
    y_score = svm_model.predict_proba(x_val)[:, 1]
    np.save('../outputs/svm_val', y_score)
    y_score = svm_model.predict_proba(x_test)[:, 1]
    np.save('../outputs/svm_test', y_score)
