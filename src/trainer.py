import numpy as np
import math
import time

import tensorflow as tf

from mlp_model import MLPConfig, MLP

def define_flags():
    tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs.")
    tf.app.flags.DEFINE_integer("batch_size", 1024, "Batch size.")
    tf.app.flags.DEFINE_integer("nlayers", 3, "Number of layers.")
    tf.app.flags.DEFINE_integer("units", 100, "Number of units in each layer.")
    tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate.")
    tf.app.flags.DEFINE_float("reg", 0.001, "Regularization parameter.")
    tf.app.flags.DEFINE_float("decay", 0, "Learning rate decay.")
    tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate.")
    tf.app.flags.DEFINE_float("max_grad_norm", 5., "Max grad norm.")
    tf.app.flags.DEFINE_integer("auc_nthresholds", 1000, "Number of AUC thresholds.")
    tf.app.flags.DEFINE_integer("nruns", 10, "Number of searching hyperparameters")

def get_loss_weights(y, weight_dict=None):
    if weight_dict is None:
        weights = np.ones_like(y)
    else:
        weights = np.empty_like(y)
        weights[y == 0] = weight_dict[0]
        weights[y == 1] = weight_dict[1]

    return weights


def evaluate(sess, model, x, y):
    results = model.evaluate(sess, x, y, FLAGS.batch_size, weight_dict)
    return results


def select_hyperparameters():
    # SEARCH INTERVALS
    lr_int = [-3, 0]
    reg_int = [-3, 0]

    sess = tf.Session()
    config = MLPConfig(FLAGS)
    model = MLP(config, x_train.shape[1])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tf.control_dependencies(update_ops)

    for n in range(FLAGS.nruns):
        tic = time.time()
        loglr = np.random.uniform(lr_int[0], lr_int[1])
        lr = math.pow(10, loglr)

        logreg = np.random.uniform(reg_int[0], reg_int[1])
        reg = math.pow(10, logreg)

        sess.run(tf.global_variables_initializer())
        model.set_lr(sess, lr)
        model.set_reg(sess, reg)

        print('Run %d/%d | Evaluating lr=%.6f, reg=%.6f'
              % (n + 1, FLAGS.nruns, loglr, logreg))

        history = model.fit(sess, x_train, y_train, FLAGS.epochs,
                            FLAGS.batch_size, weight_dict,
                            validation_data=(x_val, y_val),
                            save_best=True,
                            ckpt_path="/tmp/model.ckpt",
                            verbose=False)
        print('\ttrain -- auc: %f, loss: %e'
              % (history["best_result"]["auc"],
                 history["best_result"]["loss"]))
        print('\tvalid -- auc: %f, loss: %e'
              % (history["best_result"]["val_auc"],
                 history["best_result"]["val_loss"]))
        print('\telapsed: %.4f s' % (time.time() - tic))
    sess.close()


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    if not FLAGS.__parsed:
        define_flags()

    #  Load Data
    data = np.load("../data/creditcard_train.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    # Initialize Variables
    weight_dict = {
        0: len(y_train) / np.sum(1 - y_train),
        1: len(y_train) / np.sum(y_train)
    }

    select_hyperparameters()
