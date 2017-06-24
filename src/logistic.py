import math
import pickle
import sys
import time

from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np

from sklearn.metrics import log_loss, roc_auc_score


def print_log(log_fp, s, newline=True, verbose=True):
    try:
        log_fp.write(s + ('\n' if newline else ''))
    except:
        print('Error appeding to file %s' % log_filename)
    if verbose:
        sys.stdout.write(s + ('\n' if newline else ''))
        sys.stdout.flush()


def clean_params(params):
    tmp = dict(params)
    del(tmp['regularizer'])
    del(tmp['optimizer'])
    del(tmp['loss'])
    del(tmp['batch_size'])
    del(tmp['epochs'])
    del(tmp['nruns'])
    del(tmp['save_history'])
    del(tmp['train_weights'])
    del(tmp['val_weights'])
    del(tmp['weight_dict'])

    return tmp


def get_model_name(auc, lr, reg):
    name = 'model.' \
        + 'auc:%.6f.' % auc \
        + 'lr:%.6f.' % lr \
           + 'reg:%.6f.' % reg
    return name


def get_data_weights(weight_dict, y):
    if weight_dict is None:
        weights = None
    else:
        weights = np.empty_like(y)
        weights[y == 0] = weight_dict[0]
        weights[y == 1] = weight_dict[1]

    return weights


def copy_model(params, model):
    copy = logistic_regression(params)
    for clayer, layer in zip(copy.layers, model.layers):
        clayer.set_weights(layer.get_weights())

    return copy


def select_hyperparameters(params, log_filename):
    train_weights = params['train_weights']
    val_weights = params['val_weights']

    log_fp = open(log_filename, 'a', 1)
    epochs = params['epochs']
    nruns = params['nruns']
    batch_size = params['batch_size']

    save_history = params['save_history']

    # SEARCH INTERVALS
    lr_int = [-2.1, -1.7]
    reg_int = [-1.8, -1.2]

    for n in range(nruns):
        start_time = time.time()
        loglr = np.random.uniform(lr_int[0], lr_int[1])
        lr = math.pow(10, loglr)

        logreg = np.random.uniform(reg_int[0], reg_int[1])
        reg = math.pow(10, logreg)

        params['reg'] = reg
        params['lr'] = lr
        params['optimizer'] = Adam(lr=lr, decay=0.)
        params['regularizer'] = l2(reg)

        print('Run %3d/%3d | Evaluating lr=%.6f, reg=%.4f'
              % (n + 1, nruns, loglr, logreg))

        if save_history:
            scores_train = {
                'loss': [],
                'auc': []
            }
            scores_val = {
                'loss': [],
                'auc': []
            }

        model = logistic_regression(params)
        best_model = None
        auc = -1
        loss = -1
        for epoch in range(epochs):
            model.fit(x_train, y_train, batch_size=batch_size,
                      epochs=1, verbose=0,
                      class_weight=params['weight_dict'])

            y_score = model.predict(x_val)
            val_auc = roc_auc_score(y_val, y_score, sample_weight=val_weights)
            if save_history:
                val_loss = log_loss(y_val, y_score, 10e-8,
                                    sample_weight=val_weights)
                scores_val['auc'].append(val_auc)
                scores_val['loss'].append(val_loss)

                y_score = model.predict(x_train)
                train_auc = roc_auc_score(
                    y_train, y_score, sample_weight=train_weights)
                train_loss = log_loss(
                    y_train, y_score, 10e-8, sample_weight=train_weights)
                scores_train['auc'].append(train_auc)
                scores_train['loss'].append(train_loss)

            if val_auc > auc:
                auc = val_auc
                best_model = copy_model(params, model)
                if save_history:
                    loss = val_loss
                else:
                    loss = log_loss(y_val, y_score, 10e-8,
                                    sample_weight=val_weights)

        model_name = get_model_name(auc, lr, reg)
        best_model.save_weights('../logistic_model/' + model_name + 'hdf5')
        if save_history:
            with open('../logistic_history/'
                      + model_name + 'pickle', 'wb') as fp:
                pickle.dump([scores_train, scores_val], fp)

        print('\telapsed (s): %f' % (time.time() - start_time))
        print('\tval_loss:    %f' % (loss))
        print('\tval_auc:     %f' % (auc))
        print_log(log_fp, 'loss: %.8f, auc: %.8f, params: %s' %
                  (loss, auc, str(clean_params(params))), verbose=False)


def logistic_regression(params):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=1,
              kernel_regularizer=params['regularizer'])(inputs)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'])

    return model


if __name__ == '__main__':
    ################################################################
    # Data
    data = np.load('../data/creditcard_train.npz')
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    input_dim = x_train.shape[1]

    print('#############################################################')
    print('Fraction of positives in training = %.6f' % y_train.mean())
    print('Number of positives in training = %d' % y_train.sum())
    print('Fraction of positives in validation = %.6f' % y_val.mean())
    print('Number of positives in validation = %d' % y_val.sum())
    print('#############################################################')

    ###############################################################
    # Hyperparameters
    params = {
        'loss': 'binary_crossentropy',
        'optimizer': Adam(),
        'regularizer': l2(0.),
        'save_history': True,
        'epochs': 100,
        'batch_size': 1024,
        'nruns': 1000
    }

    params['weight_dict'] = {
        0: len(y_train) / np.sum(1 - y_train),
        1: len(y_train) / np.sum(y_train)
    }

    params['train_weights'] = get_data_weights(params['weight_dict'], y_train)
    params['val_weights'] = get_data_weights(params['weight_dict'], y_val)

    log_filename = '../logistic_results.log'

    select_hyperparameters(params, log_filename)
