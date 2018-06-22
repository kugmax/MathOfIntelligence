class StudentTParams:

    def __init__(self, tf):
        self.x = tf.placeholder(tf.float32, shape=[None])
        self.y_ = tf.placeholder(tf.float32, shape=[None])

        with tf.variable_scope('model'):
            self.a = tf.Variable(initial_value=tf.constant(0.), name="slope_of_mode")
            self.b = tf.Variable(initial_value=tf.constant(0.), name="x_intercept_of_mode")
            self.y = self.a * self.x + self.b

            self.log_bias_scale = tf.Variable(tf.constant(0.), name="scale_log_bias")
            self.arg_min_scale = tf.Variable(tf.constant(0, tf.float32), name="scale_arg_min")
            self.slope_scale = tf.Variable(tf.constant(1.), name="scale_log_slope")
            self.pow_scale = tf.Variable(tf.constant(.5, name='power_for_scale'))
            self.scale = tf.pow(
                tf.square(self.slope_scale * (self.x - self.arg_min_scale))
                / tf.exp(self.log_bias_scale) + 1., self.pow_scale, name="scale"
            ) * tf.exp(self.log_bias_scale)

            self.log_bias_shape = tf.Variable(tf.constant(0.), name="shape_log_bias")
            self.arg_min_shape = tf.Variable(tf.constant(0, tf.float32), name="shape_arg_min")
            self.slope_shape = tf.Variable(tf.constant(1.), name="shape_log_slope")
            self.pow_shape = tf.Variable(tf.constant(-.5, name='power_for_shape'))

            self.shape = tf.pow(
                tf.square(self.slope_shape * (self.x - self.arg_min_shape)) / tf.exp(self.log_bias_shape) + 1.,
                self.pow_shape, name="shape") * tf.exp(self.log_bias_shape)
