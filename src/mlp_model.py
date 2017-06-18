import numpy as np
import time
from math import ceil

from sklearn.utils import shuffle

import tensorflow as tf


class MLPConfig(object):
    config_keys = """nlayers units lr reg decay dropout
        max_grad_norm auc_nthresholds""".split()

    def __init__(self, FLAGS, **kws):
        for key in self.config_keys:
            val = kws.get(key, getattr(FLAGS, key, None))
            setattr(self, key, val)

    def __str__(self):
        msg = ("nlayers=%d units=%d lr=%.6f reg=%.6f decay=%.6f dropout=%.2f" %
               (self.nlayers, self.units, self.lr, self.reg,
                self.decay, self.dropout))
        return msg


class MLP(object):
    def __init__(self, config, input_dim):
        self.config = config
        self.input_dim = input_dim

        # Placeholder
        self.input = tf.placeholder(
            dtype=tf.float32,
            shape=(None, input_dim),
            name="input")
        self.target = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name="target")
        self.training = tf.placeholder(
            dtype=tf.bool,
            shape=[],
            name="training")
        self.loss_weights = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name="loss_weights")

        # Learning rate and decay operation
        self.lr = tf.Variable(float(self.config.lr),
                              trainable=False, name="learning_rate")
        self.lr_decay_op = self.lr.assign(self.lr * (1 - self.config.decay))
        self.reg = tf.Variable(float(self.config.reg),
                               trainable=False, name="reg")

        with tf.variable_scope("model") as vs:
            self._construct_graph()
            self.saver = tf.train.Saver(tf.global_variables())
            tf.get_variable_scope().reuse_variables()

    def set_lr(self, sess, lr):
        self.config.lr = lr
        sess.run(self.lr.assign(lr))

    def set_reg(self, sess, reg):
        self.config.reg = reg
        sess.run(self.reg.assign(reg))

    def fit(self, sess, x_train, y_train, epochs, batch_size=32,
            weight_dict=None, validation_data=None, verbose=True,
            save_best=False, ckpt_path="/tmp/model.ckpt"):

        history = {}
        history["training"] = {}
        history["validation"] = {}
        best_auc = -1
        for epoch in range(epochs):
            if verbose:
                print("Epoch %d/%d" % (epoch + 1, epochs))

            tic = time.time()
            x_train, y_train = shuffle(x_train, y_train)
            train_results = self._epoch_run(sess, x_train, y_train, batch_size,
                                            weight_dict, do_backward=True)
            # TODO: store history
            if validation_data is not None:
                val_results = self._epoch_run(sess, validation_data[0],
                                              validation_data[1], batch_size,
                                              weight_dict)
                if save_best and val_results["auc"] > best_auc:
                    best_auc = val_results["auc"]
                    history["best_result"] = {
                        "epoch": epoch,
                        "auc": train_results["auc"],
                        "loss": train_results["loss"],
                        "val_auc": val_results["auc"],
                        "val_loss": val_results["loss"]
                    }
                    self.saver.save(sess, ckpt_path, write_meta_graph=False)

            if verbose:
                print("\ttrain -- loss: %.6e, " % (train_results["loss"])
                      + "total_loss: %.6e, " % (train_results["total_loss"])
                      + "auc: %.6f, " % (train_results["auc"])
                      + "avg_grad: %.6f." % (train_results["grad_norm"]))
                if validation_data is not None:
                    print("\tvalid -- loss: %.6e, " % (val_results["loss"])
                          + "total_loss: %.6e, " % (val_results["total_loss"])
                          + "auc: %.6f." % (val_results["auc"]))
                print("\telapsed: %.4f s" % (time.time() - tic))

        return history

    def evaluate(self, sess, x, y, batch_size=32, weight_dict=None):
        results = self._epoch_run(sess, x, y, batch_size, weight_dict)
        return results

    def step(self, sess, x, y, weight_dict=None, do_backward=False):
        loss_weights = self._get_loss_weights(y, weight_dict)

        feed_in = {}
        feed_in[self.training] = do_backward
        feed_in[self.input] = x
        feed_in[self.target] = y
        feed_in[self.loss_weights] = loss_weights

        feed_out = {}
        if do_backward:
            feed_out["grad_norm"] = self.grad_norm
            feed_out["back_update"] = self.update
        feed_out["loss"] = self.loss
        feed_out["total_loss"] = self.total_loss
        feed_out["auc"] = self.auc
        feed_out["output"] = self.output

        return sess.run(feed_out, feed_in)

    def _epoch_run(self, sess, x, y, batch_size, weight_dict=None,
                   do_backward=False):
        num_examples = x.shape[0]
        total_batches = ceil(num_examples / batch_size)

        output = np.zeros((0,))
        avg_loss = 0.
        avg_total_loss = 0.
        if do_backward:
            avg_grad = 0.

        sess.run(tf.local_variables_initializer())
        for batch in range(total_batches):
            idx_lower = batch * batch_size
            idx_upper = min(idx_lower + batch_size, num_examples)
            batch_x = x[idx_lower:idx_upper, :]
            batch_y = y[idx_lower:idx_upper]

            results = self.step(sess, batch_x, batch_y, weight_dict,
                                do_backward)

            output = np.concatenate([output, results["output"]])
            avg_loss += results["loss"] / total_batches
            avg_total_loss += results["total_loss"] / total_batches

            if do_backward:
                avg_grad += results["grad_norm"] / total_batches

        results["output"] = output
        results["loss"] = avg_loss
        results["total_loss"] = avg_total_loss
        if do_backward:
            results["grad_norm"] = avg_grad

        return results

    def _get_loss_weights(self, y, weight_dict=None):
        if weight_dict is None:
            weights = np.ones_like(y)
        else:
            weights = np.empty_like(y)
            weights[y == 0] = weight_dict[0]
            weights[y == 1] = weight_dict[1]

        return weights

    def _construct_graph(self):
        # Model
        x = self.input
        for i in range(self.config.nlayers):
            x = tf.layers.dense(
                inputs=x,
                units=self.config.units,
                name="reg/dense%d" % i)
            x = tf.layers.batch_normalization(
                inputs=x,
                training=self.training,
                name="batch_norm%d" % i)
            x = tf.nn.relu(
                features=x,
                name="relu%d" % i)
            x = tf.layers.dropout(
                inputs=x,
                rate=self.config.dropout,
                training=self.training,
                name="dropout%d" % i)
        x = tf.layers.dense(
            inputs=x,
            units=1,
            name="reg/output_dense")
        x = tf.sigmoid(
            x=x,
            name="output")
        self.output = tf.squeeze(x)

        # Loss and auc
        self.loss = tf.losses.log_loss(
            labels=self.target,
            predictions=self.output,
            weights=self.loss_weights,
            reduction=tf.losses.Reduction.MEAN)

        _, self.auc = tf.metrics.auc(
            labels=self.target,
            num_thresholds=self.config.auc_nthresholds,
            predictions=self.output,
            weights=self.loss_weights)

        # Optimizer
        adam = tf.train.AdamOptimizer(self.lr, use_locking=True)
        params = tf.trainable_variables()

        l2loss = self.reg*tf.add_n([
            tf.nn.l2_loss(v) for v in params if 'reg/' in v.name
            and 'bias' not in v.name])
        self.total_loss = self.loss + l2loss

        grads = tf.gradients(self.total_loss, params)
        grads, norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        self.grad_norm = norm
        self.update = adam.apply_gradients(zip(grads, params))
