import numpy as np


class Kernel:

    def __init__(self):
        self._sqrt_5 = np.sqrt(5.).astype(np.float32)

    def matern_kernel(self, tf, points_1, points_2, log_length_scale, log_sample_noise=None, log_kernel_scale=0.):
        # First we compute the distances:
        distances_squared = self.distances_squared_matrix(tf, points_1, points_2, log_length_scale)
        noise_shift = 0
        if points_1 == points_2:
            # make sure the diagonal consists of zeros to compute the square root safely
            distances_squared = self.zeros_diag(tf, distances_squared)
            if log_sample_noise != None:
                # add intrisic sampling noise; it is essential that we include this since we are bootstrapping.
                noise_shift = tf.diag(tf.tile([tf.exp(log_sample_noise)], [tf.shape(points_1)[0]]))
        distances = tf.sqrt(distances_squared)

        return tf.exp(log_kernel_scale) * (1 + self._sqrt_5 * distances + 5 * distances_squared / 3.) \
               * tf.exp(-self._sqrt_5 * distances) + noise_shift

    # Here is the a helper distance function:
    def distances_squared_matrix(self, tf, points_1, points_2, log_length_scale):
        # First we rescale the points
        inv_scale = tf.exp(-tf.reshape((log_length_scale), [1, -1]))
        res_points_1 = tf.multiply(points_1, inv_scale)
        res_points_2 = tf.multiply(points_2, inv_scale)
        # Next we compute the distances,
        # we are using the formula (x-y)(x-y) = x^2-2xy+y^2
        Radial_dist_1 = tf.reduce_sum(tf.multiply(res_points_1, res_points_1), 1, keep_dims=True)
        Radial_dist_2 = tf.reduce_sum(tf.multiply(res_points_2, res_points_2), 1, keep_dims=True)
        distances_squared = Radial_dist_1 - 2 * tf.matmul(res_points_1, tf.transpose(res_points_2)) + tf.transpose(
            Radial_dist_2)
        return distances_squared

    def gaussian_kernel(self, tf, points_1, points_2, log_length_scale, log_sample_noise=None, log_kernel_scale=0.):
        # First we compute the distances:
        distances_squared = self.distances_squared_matrix(tf, points_1, points_2, log_length_scale)
        if log_sample_noise != None and points_1 == points_2:
            # make sure the diagonal consists of zeros to compute the square root safely
            distances_squared = self.zeros_diag(tf, distances_squared)
            # add intrisic sampling noise.
            noise_shift = tf.diag(tf.tile([tf.exp(log_sample_noise)], [tf.shape(points_1)[0]]))
        else:
            noise_shift = 0
        return tf.exp(log_kernel_scale) * tf.exp(-distances_squared) + noise_shift

    # we need the following to guarantee numerical stability...
    def zeros_diag(self, tf, matrix):
        # create a matrix with True along diagonal and False elsewhere (to be used as a mask)
        mask = tf.cast(tf.diag(tf.tile(tf.constant([1]), [tf.shape(matrix)[0]])), tf.bool)
        # create a zero matrix:
        zeros = tf.diag(tf.tile(tf.constant([0.]), [tf.shape(matrix)[0]]))

        return tf.where(mask, zeros, matrix)