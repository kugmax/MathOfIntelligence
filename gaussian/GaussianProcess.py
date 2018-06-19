
import tensorflow as tf
import tensorflow.contrib.distributions as dist


# K(x, x[i]) * K^-1[i, j] * y[j]
def mean(kernel, gp_sampled_x, gp_sampled_y, new_points):
    # First compute the sample mean:

    sampled_mean_y = tf.reduce_mean(gp_sampled_y, 0)

    inv_cov_matrix = tf.matrix_inverse(kernel(gp_sampled_x, gp_sampled_x))
    # reshape the the sample values into matrices:
    gp_sampled_y_reshaped = tf.reshape(gp_sampled_y, [-1, 1])
    # return the expected mean
    return sampled_mean_y + tf.matmul(kernel(new_points, gp_sampled_x),
                                      tf.matmul(inv_cov_matrix, gp_sampled_y_reshaped -sampled_mean_y))


# k(x, x) - k(x, x[i]) * k^-1[i, j] * k(x[j], x)
def cov(kernel, gp_sampled_x, new_points):
    inv_cov_matrix = tf.matrix_inverse(kernel(gp_sampled_x, gp_sampled_x))
    k = kernel(gp_sampled_x, new_points)
    return kernel(new_points, new_points) - tf.matmul(tf.transpose(k), tf.matmul(inv_cov_matrix, k))


# new_points is a (num_points,amb_dim) shape matrix of prospective points at which to
# measure the cross-entropy (on the validation set)
def expected_improv(kernel, GP_sampled_x, GP_sampled_y, new_points):
    # compute the mean of the gaussian process at new_points
    mu = tf.reshape(mean(kernel, GP_sampled_x, GP_sampled_y, new_points), [-1])
    # compute the standard deviation across those new_points
    sigma = tf.diag_part(cov(kernel, GP_sampled_x, new_points))
    # check that the standard deviation is positive (and fill in a dummy value of 1 otherwise)
    non_zero_variance = tf.greater(sigma, 0., name="variance_Control_Op")
    sigma_safe = tf.select(non_zero_variance, sigma, tf.tile(tf.constant([1.]), tf.shape(sigma)))
    # model our expected cross-entropy at those new points using the gaussian process
    normal_distribution = dist.Normal(mu=mu, sigma=sigma_safe)
    # compare our model with the current minimum
    min_sampled_y = tf.reshape(tf.reduce_min(GP_sampled_y), [-1])
    # compute the expected value of max(min_sampled_y - mu)
    result = (min_sampled_y - mu) * normal_distribution.cdf(min_sampled_y) + sigma * normal_distribution.pdf(
        min_sampled_y)
    return tf.select(non_zero_variance, result, tf.tile(tf.constant([0.]), tf.shape(non_zero_variance)))


def log_prob_of_improv(kernel, gp_sampled_x, gp_sampled_y, new_points):
    mu = tf.reshape(mean(kernel, gp_sampled_x, gp_sampled_y, new_points), [-1])
    sigma = tf.diag_part(cov(kernel, gp_sampled_x, new_points))
    non_zero_variance = tf.greater(sigma, 0., name="variance_Control_Op")
    sigma_safe = tf.select(non_zero_variance, sigma, tf.tile(tf.constant([1.]), tf.shape(sigma)))
    normal_distribution = dist.Normal(mu=mu, sigma=sigma_safe)
    min_sampled_y = tf.reshape(tf.reduce_min(gp_sampled_y), [-1])
    return tf.select(non_zero_variance,
                     normal_distribution.log_cdf(min_sampled_y),
                     tf.tile(tf.constant([0.]), tf.shape(non_zero_variance)))


def main():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


if __name__== "__main__":
    main()