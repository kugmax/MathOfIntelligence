
class GaussianProcessParams:

    def __init__(self, tf, hyper_param_dim):
        with tf.variable_scope('GP'):
            self.log_length_scale = tf.tile(tf.Variable(tf.constant([1.])), [hyper_param_dim])
            self.log_sample_noise = tf.Variable(tf.constant(0.))
            self.log_kernel_scale = tf.Variable(tf.constant(0.))

            self.sampled_x = tf.placeholder(tf.float32, shape=[None, hyper_param_dim])
            self.sampled_y = tf.placeholder(tf.float32, shape=[None])

            self.new_x = tf.placeholder(tf.float32, shape=[None, hyper_param_dim])
