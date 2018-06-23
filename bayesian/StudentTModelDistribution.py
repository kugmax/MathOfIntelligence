import tensorflow.contrib.distributions as dist

from bayesian.StudentTParams import StudentTParams


class StudentTModelDistribution:
    def __init__(self, tf, valid_data_len, train_and_valid_data, hyper_param_dim):
        self._valid_data = train_and_valid_data[:, :valid_data_len]
        self._train_data = train_and_valid_data[:, valid_data_len:]

        self._l2_reg_strength = tf.placeholder(tf.float32, shape=[hyper_param_dim])
        self._model_lr = tf.placeholder(tf.float32)

        self._cross_entropy = None

        self._params = StudentTParams(tf)
        self._model_distribution = dist.StudentT(df=self._params.shape, loc=self._params.y, scale=self._params.scale)

    def get_model(self):
        return self._model_distribution

    def get_log_likelihoods(self):
        return self._model_distribution.log_prob(self._params.y_)

    def get_params(self):
        return self._params

    def build_train_model(self, tf):
        self._cross_entropy = -tf.reduce_mean(self.get_log_likelihoods())

        with tf.variable_scope('training_model'):
            optimizer = tf.train.AdamOptimizer(self._model_lr)

            array = tf.stack(
                [tf.nn.l2_loss(self._params.slope_scale), tf.nn.l2_loss(self._params.pow_scale - .5),
                 tf.nn.l2_loss(self._params.slope_shape), tf.nn.l2_loss(self._params.pow_shape + .5),
                 tf.nn.l2_loss(self._params.arg_min_scale), tf.nn.l2_loss(self._params.arg_min_shape)]
            )

            train = optimizer.minimize(
                self._cross_entropy + tf.reduce_sum(tf.multiply(self._l2_reg_strength, array))
            )

            return train

    def train_model(self, tf, sess, train, l2_reg_strength_val, num_steps=1000, learning_rate=0.05):

        model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
        tf.initialize_variables(model_variables).run()

        training_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='training_model')
        tf.initialize_variables(training_variables).run()

        for step in range(num_steps + 1):
            training_loss, _ = sess.run([self._cross_entropy, train],
                                        {self._params.x: self._train_data[0, :],
                                         self._params.y_: self._train_data[1, :],
                                         self._l2_reg_strength: l2_reg_strength_val,
                                         self._model_lr: learning_rate})
            # if step % 100 == 0:
            #     print("step: %d, loss: %.3f, a: %.4f, b: %.4f, min scale: %.4f, scale rate of change: %.4f, \n \
            #              min shape: %.4f shape rate of change: %.2f"
            #           % (step, training_loss, self._params.a.eval(), self._params.b.eval(),
            #              tf.exp(self._params.log_bias_scale).eval(), tf.abs(self._params.slope_scale).eval(),
            #              tf.exp(self._params.log_bias_shape).eval(), tf.abs(self._params.slope_shape).eval()))

        return self._cross_entropy.eval({self._params.x: self._valid_data[0, :],
                                         self._params.y_: self._valid_data[1, :]})
