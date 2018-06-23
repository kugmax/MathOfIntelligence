
import tensorflow as tf
import tensorflow.contrib.distributions as dist
from numpy import pi


class GaussianProcess:

    def __init__(self, gp_params, kernel):
        with tf.variable_scope('GP'):
            self._optimizer = tf.train.AdamOptimizer(.05)

        self._log_likelihood = None
        self._gp_params = gp_params
        self._kernel = kernel

    # K(x, x[i]) * K^-1[i, j] * y[j]
    def mean(self, kernel, gp_sampled_x, gp_sampled_y, new_points):
        # First compute the sample mean:

        sampled_mean_y = tf.reduce_mean(gp_sampled_y, 0)

        inv_cov_matrix = tf.matrix_inverse(kernel(gp_sampled_x, gp_sampled_x))
        # reshape the the sample values into matrices:
        gp_sampled_y_reshaped = tf.reshape(gp_sampled_y, [-1, 1])
        # return the expected mean
        return sampled_mean_y + tf.matmul(kernel(new_points, gp_sampled_x),
                                          tf.matmul(inv_cov_matrix, gp_sampled_y_reshaped -sampled_mean_y))

    def kernel_lambda(self):
        return lambda l2x, l2y: \
            self._kernel.matern_kernel(tf=tf, points_1=l2x, points_2=l2y,
                                       log_length_scale=self._gp_params.log_length_scale,
                                       log_sample_noise=self._gp_params.log_sample_noise,
                                       log_kernel_scale=self._gp_params.log_kernel_scale)


    # k(x, x) - k(x, x[i]) * k^-1[i, j] * k(x[j], x)
    def cov(self, kernel, gp_sampled_x, new_points):
        inv_cov_matrix = tf.matrix_inverse(kernel(gp_sampled_x, gp_sampled_x))
        k = kernel(gp_sampled_x, new_points)
        return kernel(new_points, new_points) - tf.matmul(tf.transpose(k), tf.matmul(inv_cov_matrix, k))

    # new_points is a (num_points,amb_dim) shape matrix of prospective points at which to
    # measure the cross-entropy (on the validation set)
    def expected_improv(self, kernel, GP_sampled_x, GP_sampled_y, new_points):
        # compute the mean of the bayesian process at new_points
        mu = tf.reshape(self.mean(kernel, GP_sampled_x, GP_sampled_y, new_points), [-1])
        # compute the standard deviation across those new_points
        sigma = tf.diag_part(self.cov(kernel, GP_sampled_x, new_points))
        # check that the standard deviation is positive (and fill in a dummy value of 1 otherwise)
        non_zero_variance = tf.greater(sigma, 0., name="variance_Control_Op")
        sigma_safe = tf.where(non_zero_variance, sigma, tf.tile(tf.constant([1.]), tf.shape(sigma)))
        # model our expected cross-entropy at those new points using the bayesian process
        normal_distribution = dist.Normal(loc=mu, scale=sigma_safe)
        # compare our model with the current minimum
        min_sampled_y = tf.reshape(tf.reduce_min(GP_sampled_y), [-1])
        # compute the expected value of max(min_sampled_y - mu)
        result = (min_sampled_y - mu) * normal_distribution.cdf(min_sampled_y) + sigma * normal_distribution.log_prob(
            min_sampled_y)
        return tf.where(non_zero_variance, result, tf.tile(tf.constant([0.]), tf.shape(non_zero_variance)))

    def log_prob_of_improv(self, kernel, gp_sampled_x, gp_sampled_y, new_points):
        mu = tf.reshape(self.mean(kernel, gp_sampled_x, gp_sampled_y, new_points), [-1])
        sigma = tf.diag_part(self.cov(kernel, gp_sampled_x, new_points))
        non_zero_variance = tf.greater(sigma, 0., name="variance_Control_Op")
        sigma_safe = tf.where(non_zero_variance, sigma, tf.tile(tf.constant([1.]), tf.shape(sigma)))
        normal_distribution = dist.Normal(loc=mu, scale=sigma_safe)
        min_sampled_y = tf.reshape(tf.reduce_min(gp_sampled_y), [-1])
        return tf.where(non_zero_variance,
                         normal_distribution.log_cdf(min_sampled_y),
                         tf.tile(tf.constant([0.]), tf.shape(non_zero_variance)))

    def get_optimizer(self):
        return self._optimizer

    def build_train(self):
        gp_kernel = self.kernel_lambda()

        cov_matrix = gp_kernel(self._gp_params.sampled_x, self._gp_params.sampled_x)
        cov_matrix_inv = tf.matrix_inverse(cov_matrix)

        # GP_mean_zero_sampled_y is the vector {y_i-\bar y}, with mean zero.
        mean_zero_sampled_y = tf.reshape((self._gp_params.sampled_y - tf.reduce_mean(self._gp_params.sampled_y)), [-1, 1])
        self._log_likelihood = tf.matmul(tf.matmul(tf.transpose(mean_zero_sampled_y), cov_matrix_inv),
                                      mean_zero_sampled_y) \
                            + self.log_determinant_for_PD_matrices(cov_matrix) \
                            + 0.5 * tf.log(2 * pi) * tf.to_float(tf.shape(self._gp_params.sampled_y)[0])

        with tf.variable_scope('GP'):
            return self._optimizer.minimize(self._log_likelihood)

    # We need a numerically stable means to compute the log-determinant of a matrix when fitting the Gaussian process.
    def log_determinant_for_PD_matrices(self, tensor):
        # first compute the cholesky decomposition M=R*R^T
        cholesy_root = tf.cholesky(tensor)
        # now det(M)=det(R)^2=(Prod diag_entries(R))^2
        # so log(det(M))=2*sum(log(diag_entries(R)))
        return 2 * tf.reduce_sum(tf.log(tf.diag_part(cholesy_root)))

    def get_log_likelihood(self):
        return self._log_likelihood
