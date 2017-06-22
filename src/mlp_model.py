from model import Config, Model

import tensorflow as tf

class MLPConfig(Config):
    _config_keys = Config._config_keys + \
        "nlayers units dropout max_grad_norm".split()


class MLP(Model):
    def __init__(self, config, input_dim):
        super().__init__(config, input_dim)

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

        l2loss = self.reg * tf.add_n([
            tf.nn.l2_loss(v) for v in params if "reg/" in v.name and
            "bias" not in v.name])
        self.total_loss = self.loss + l2loss

        grads = tf.gradients(self.total_loss, params)
        grads, norm = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        self.grad_norm = norm
        self.update = adam.apply_gradients(zip(grads, params))
