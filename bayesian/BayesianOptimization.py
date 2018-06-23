import tensorflow as tf
import tensorflow.contrib.distributions as dist
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from bayesian.DataHolder import DataHolder
from bayesian.GaussianProcess import GaussianProcess
from bayesian.GaussianProcessParams import GaussianProcessParams
from bayesian.Kerner import Kernel
from bayesian.StudentTModelDistribution import StudentTModelDistribution

testing_proportion = 0.2
validation_proportion = 0.2

hyper_param_dim = 6
l2_reg_min, l2_reg_max = -50, 0
l2_step_size = (l2_reg_max - l2_reg_min) / 20.
sqrt_2 = np.sqrt(2.).astype(np.float32)

# The folowing are global settings used for the bayesian optimization of the hyper-parameters.
initial_GP_test_points = 5  # these are randomly chosen points with which to initialize the bayesian optimization
total_GP_test_points = 20  # total number of points used in bayesian optimization
max_feelers = 10  # number of points used in batch-gradient-descent optimization.


def load_data():
    df = pd.read_csv(filepath_or_buffer='FAO.csv', encoding='cp1252')

    rows = df[(df['Area Abbreviation'] == 'FRA') & (df['Item Code'] == 2513)]\
        .fillna(0) \
        .values
    data = rows[0:2, 10:]

    return data


def plot_model(student_t, data_holder):
    x_step_size = (data_holder.get_x_max() - data_holder.get_x_min()) / 20.
    y_step_size = (data_holder.get_y_max() - data_holder.get_y_min()) / 20.
    x_grid = np.arange(data_holder.get_x_min(), data_holder.get_x_max(), x_step_size)
    y_grid = np.arange(data_holder.get_y_min(), data_holder.get_y_max(), y_step_size)
    fig, ax = plt.subplots()

    params = student_t.get_params()

    # Plot the predicted probability density
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.exp(student_t.get_log_likelihoods().eval(
        {params.x: X.reshape(-1), params.y_: Y.reshape(-1)})
    ).reshape(len(x_grid), len(y_grid))

    ax.contourf(X, Y, Z, cmap='YlGn')
    ax.autoscale(False)

    # plot the predicted mode
    ax.plot(x_grid, params.a.eval() * x_grid + params.b.eval(), linewidth=2)

    # plot the standard deviation
    # first, restrict to where it's defined
    x_std_dev_defined = x_grid[params.shape.eval({params.x: x_grid}) > 2]
    # then compute the standard deviation
    std_dev = (params.scale * tf.sqrt((params.shape / (params.shape - 2.)))).eval({params.x: x_std_dev_defined})
    # now plot it
    ax.plot(x_std_dev_defined, params.a.eval() * x_std_dev_defined + params.b.eval() + std_dev, 'purple')
    ax.plot(x_std_dev_defined, params.a.eval() * x_std_dev_defined + params.b.eval() - std_dev, 'purple')

    ax.plot(data_holder.get_data()[0], data_holder.get_data()[1], 'ro')
    plt.show()


def prepare_model(sess):
    data = load_data()
    data_holder = DataHolder(data)
    data_holder.plot_data()

    _, data_len = data.shape
    test_data_len = int(np.floor(data_len * testing_proportion))
    valid_data_len = int(np.floor(data_len * validation_proportion))
    test_data = data[:, test_data_len:]
    train_and_valid_data = data[:, test_data_len:]

    student_t = StudentTModelDistribution(tf, valid_data_len, train_and_valid_data, hyper_param_dim)
    model_train = student_t.build_train_model(tf)

    result_cross_entropy = student_t.train_model(tf=tf, sess=sess, train=model_train,
                                                 l2_reg_strength_val=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 num_steps=10000, learning_rate=0.005)

    print("-" * 30 + "\n validation-cross-entropy estimate: %f"
          % result_cross_entropy)

    plot_model(student_t, data_holder)

    return model_train, student_t, data_holder


def fit_GP(sess, gp, sampled_x, sampled_y, gp_params, gp_train):
    model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='GP')
    tf.initialize_variables(model_variables).run()
    for i in range(500):
        loss, _ = sess.run([gp.get_log_likelihood(), gp_train],
                           {gp_params.sampled_x: sampled_x,
                            gp_params.sampled_y: sampled_y})

        # print("step: %d, loss: %.4f" % (i, loss))
        # print("length scale: %.3f, Sample Noise: %.3f, Kernel Scale: %f" %
        #       (tf.exp(gp_params.log_length_scale).eval()[0],
        #        tf.exp(gp_params.log_sample_noise).eval(),
        #        tf.exp(gp_params.log_kernel_scale).eval()))


def plot_GP_model(sampled_x, sampled_y, gp_params, gp_mean_func):
    x_rescale = max(tf.exp(-gp_params.log_length_scale).eval()[0], 1.)
    y_rescale = max(tf.exp(-gp_params.log_length_scale).eval()[1], 1.)
    log_l2_scale_grid = np.arange(x_rescale * (l2_reg_min - 5 * l2_step_size),
                                  x_rescale * (l2_reg_max + 5 * l2_step_size), x_rescale * l2_step_size)
    log_l2_shape_grid = np.arange(y_rescale * (l2_reg_min - 5 * l2_step_size),
                                  y_rescale * (l2_reg_max + 5 * l2_step_size), y_rescale * l2_step_size)
    # Make a grid of points
    X, Y = np.meshgrid(log_l2_scale_grid, log_l2_shape_grid)  # grid of points
    min_index = np.argmin(sampled_y)
    hidden0, hidden1 = sampled_x[min_index, 0], sampled_x[min_index, 1]
    hidden2, hidden3 = sampled_x[min_index, 2], sampled_x[min_index, 3]
    hidden4, hidden5 = sampled_x[min_index, 4], sampled_x[min_index, 5]
    X_Y1_zip = list(zip(X.reshape(-1), Y.reshape(-1),
                        [hidden2] * X.size, [hidden3] * X.size, [hidden4] * X.size, [hidden5] * X.size))
    X_Y2_zip = list(zip([hidden0] * X.size, [hidden1] * X.size, X.reshape(-1), Y.reshape(-1),
                        [hidden4] * X.size, [hidden5] * X.size))
    # Compute the expected value of the Gaussian process on that grid
    Z1 = gp_mean_func.eval({gp_params.sampled_x: sampled_x,
                            gp_params.sampled_y: sampled_y,
                            gp_params.new_x: X_Y1_zip})
    Z1 = np.exp(Z1.reshape(len(log_l2_scale_grid), len(log_l2_shape_grid)))
    Z2 = gp_mean_func.eval({gp_params.sampled_x: sampled_x,
                            gp_params.sampled_y: sampled_y,
                            gp_params.new_x: X_Y2_zip})
    Z2 = Z2.reshape(len(log_l2_scale_grid), len(log_l2_shape_grid))

    # adding the Contour lines with labels: First find limits for the plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set_title('Expected loss\n on validation set')
    ax1.set_xlabel('scale slope hyperparameter')
    ax1.set_ylabel('scale power hyperparameter')
    ax2.set_title('Expected loss\n on validation set')
    ax2.set_xlabel('shape slope hyperparameter')
    ax2.set_ylabel('shape power hyperparameter')

    # Next add the color-heights
    im1 = ax1.contourf(X, Y, Z1, 8, alpha=.75, cmap='jet')
    plt.colorbar(im1, ax=ax1)  # adding the colorbar on the right
    im2 = ax2.contourf(X, Y, Z2, 8, alpha=.75, cmap='jet')
    plt.colorbar(im2, ax=ax2)  # adding the colorbar on the right
    ax2.autoscale(False)  # To avoid that the scatter changes limits

    # Next add contours to the plot of the mean:
    cset1 = ax1.contour(X, Y, Z1, 4, colors='black', linewidth=.5)
    ax1.clabel(cset1, inline=True, fmt='%1.1f', fontsize=10)
    ax1.autoscale(False)  # To avoid that the scatter changes limits
    cset2 = ax2.contour(X, Y, Z2, 4, colors='black', linewidth=.5)
    ax2.clabel(cset2, inline=True, fmt='%1.1f', fontsize=10)
    ax2.autoscale(False)  # To avoid that the scatter changes limits

    # Now plot the sampled points
    ax1.scatter(sampled_x[:, 0], sampled_x[:, 1])
    ax2.scatter(sampled_x[:, 0], sampled_x[:, 1])

    plt.show()


def find_test_point(sess, gp_params, log_sample_points, log_sample_values, test_point_list,
                    gp_test_points, gp_test_point_losses, gp_test_point_train):
    model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='GP-test-point-finder')
    tf.initialize_variables(model_variables).run()
    # optimize the probability of improvement:
    for j in range(100):
        cur_points, losses, _ = sess.run([gp_test_points, gp_test_point_losses, gp_test_point_train],
                                         {gp_params.sampled_x: log_sample_points,
                                          gp_params.sampled_y: log_sample_values})
        for i in range(max_feelers):
            if not np.isfinite(cur_points[i]).all():
                print('resetting feeler')
                tf.initialize_variables([test_point_list[i]]).run()
    min_index = np.argmin(losses)
    return cur_points[min_index, :]


def find_gp_test_point(sess, gp_train, model_train, student_t, kernel, gp, gp_params, gp_mean_func,
                       log_sample_points, log_sample_values,
                       sample_points, sample_values):
    with tf.variable_scope('GP-test-point-finder'):
        test_point_list = []
        for i in range(max_feelers):
            test_point_list += [tf.Variable(tf.random_uniform([hyper_param_dim], l2_reg_min, l2_reg_max))]
        # our test point is randomly initialized
        gp_test_points = tf.stack(test_point_list)
        # For the kernel we stop gradients from back-propagating to the length scale and noise estimates.
        gp_fixed_kernel = lambda l2x, l2y: kernel.matern_kernel(tf, l2x, l2y,
                                                                tf.stop_gradient(gp_params.log_length_scale),
                                                                tf.stop_gradient(gp_params.log_sample_noise),
                                                                tf.stop_gradient(gp_params.log_kernel_scale))
        gp_test_point_losses = -gp.expected_improv(gp_fixed_kernel,
                                                   gp_params.sampled_x, gp_params.sampled_y, gp_test_points)
        gp_test_point_train = gp.get_optimizer().minimize(tf.reduce_mean(gp_test_point_losses))

    new_tests = total_GP_test_points - initial_GP_test_points + 1
    for i in range(new_tests):
        test_point = find_test_point(sess, gp_params, log_sample_points, log_sample_values, test_point_list,
                                     gp_test_points, gp_test_point_losses, gp_test_point_train)
        # print("%d'th test at: (%.3f,%.3f)"%(i,test_point[0],test_point[1]))
        if np.isfinite(test_point).all():
            # add the test point to our list of sampled points
            log_sample_points = np.append(log_sample_points, [test_point], axis=0)
            sample_points = np.append(sample_points, [np.exp(log_sample_points[-1, :])], axis=0)
            # estimate the validation-cross-entropy error at the new test point
            sample_values = np.append(sample_values,
                                      [student_t.train_model(tf=tf, sess=sess, train=model_train,
                                                             l2_reg_strength_val=sample_points[-1, :],
                                                             num_steps=1000, learning_rate=.1)],
                                      axis=0)
            log_sample_values = np.append(log_sample_values, [np.log(sample_values[-1])], axis=0)
        else:
            print('not finite')
        # Fit the Gaussian process model (after every 5 new test points)
        if i % 5 == 0:
            fit_GP(sess, gp, log_sample_points, log_sample_values, gp_params, gp_train)
        # display the results (every 20 new points)
        if i % 5 == 0:
            plot_GP_model(log_sample_points, log_sample_values, gp_params, gp_mean_func)


def find_gp_best_point(sess, kernel, gp, gp_params, log_sample_points, log_sample_values):
    with tf.variable_scope('GP-best-point-finder'):
        best_point_list = []
        for i in range(max_feelers):
            best_point_list += [tf.Variable(tf.random_uniform([hyper_param_dim], l2_reg_min, l2_reg_max))]
        # our best point is randomly initialized
        gp_best_points = tf.stack(best_point_list)
        # For the kernel we stop gradients from back-propagating to the length scale and noise estimates.
        gp_fixed_kernel = lambda l2x, l2y: kernel.matern_kernel(tf, l2x, l2y,
                                                                tf.stop_gradient(gp_params.log_length_scale),
                                                                tf.stop_gradient(gp_params.log_sample_noise),
                                                                tf.stop_gradient(gp_params.log_kernel_scale))
        gp_best_point_means = gp.mean(gp_fixed_kernel,
                                      gp_params.sampled_x, gp_params.sampled_y, gp_best_points)
        gp_best_point_train = gp.get_optimizer().minimize(tf.reduce_mean(gp_best_point_means))

    model_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='GP-best-point-finder')
    tf.initialize_variables(model_variables).run()
    # find the value of best_point which minimizes the Gaussian process:
    for j in range(1000):
        cur_points, means, _ = sess.run([gp_best_points, gp_best_point_means, gp_best_point_train],
                                        {gp_params.sampled_x: log_sample_points,
                                         gp_params.sampled_y: log_sample_values})
        for i in range(max_feelers):
            if not np.isfinite(cur_points[i]).all():
                print('resetting feeler')
                tf.initialize_variables([best_point_list[i]]).run()
    min_index = np.argmin(means)
    return cur_points[min_index, :]


def do_optimization(sess, model_train, student_t):
    kernel = Kernel()
    gp_params = GaussianProcessParams(tf, hyper_param_dim)

    gp = GaussianProcess(gp_params, kernel)
    gp_train = gp.build_train()

    gp_mean_func = gp.mean(gp.kernel_lambda(), gp_params.sampled_x, gp_params.sampled_y, gp_params.new_x)
    gp_cov_func = gp.cov(gp.kernel_lambda(), gp_params.sampled_x, gp_params.new_x)

    log_sample_points = np.random.uniform(low=l2_reg_min, high=l2_reg_max,
                                          size=[initial_GP_test_points, hyper_param_dim]).astype(np.float32)
    sample_points = np.exp(log_sample_points)

    sample_values = []
    for i in range(initial_GP_test_points):
        sample_values += [student_t.train_model(tf=tf, sess=sess, train=model_train,
                                                l2_reg_strength_val=sample_points[i, :],
                                                num_steps=1000,
                                                learning_rate=.1)
                          ]
    log_sample_values = np.log(sample_values)

    fit_GP(sess, gp, log_sample_points, log_sample_values, gp_params, gp_train)
    plot_GP_model(log_sample_points, log_sample_values, gp_params, gp_mean_func)

    find_gp_test_point(sess, gp_train, model_train, student_t,
                       kernel, gp, gp_params, gp_mean_func,
                       log_sample_points, log_sample_values, sample_points, sample_values)
    best_point = find_gp_best_point(sess, kernel, gp, gp_params, log_sample_points, log_sample_values)

    gp_min_point_est = np.exp(best_point)
    gp_min_value_est = tf.exp(gp_mean_func).eval({gp_params.sampled_x: log_sample_points,
                                                  gp_params.sampled_y: log_sample_values,
                                                  gp_params.new_x: [best_point]})
    gp_min_error_est = gp_min_value_est * tf.sqrt(tf.reduce_mean(gp_cov_func)).eval(
        {gp_params.sampled_x: log_sample_points,
         gp_params.sampled_y: log_sample_values,
         gp_params.new_x: [best_point]})
    print("Expected minimum validation-cross entropy: %.4f +- %.4f" % (gp_min_value_est, gp_min_error_est))
    print("At point: (%.4f,%.4f,%.4f,%.4f)" % (best_point[0], best_point[1], best_point[2], best_point[3]))
    print("Corresponding to hyperparameters:\n\
            prior confidence of constant scale: %.4f,\n\
            prior confidence of scale growing linearly: %.4f,\n\
            prior confidence of constant shape: %.4f,\n\
            prior confidence of shape decreasing inverse-linearly: %.4f" \
          % (np.sqrt(gp_min_point_est[0]),
             np.sqrt(gp_min_point_est[1]),
             np.sqrt(gp_min_point_est[2]),
             np.sqrt(gp_min_point_est[3])))

    return gp_min_point_est


def main():
    sess = tf.InteractiveSession()

    model_train, student_t, data_holder = prepare_model(sess)
    gp_min_point_est = do_optimization(sess, model_train, student_t)

    test_set_cross_entropy = student_t.train_model(tf=tf, sess=sess, train=model_train,
                                                   l2_reg_strength_val=gp_min_point_est,
                                                   num_steps=10000, learning_rate=0.005)
    plot_model(student_t, data_holder)
    print("Test Set Cross Entropy Error: %.4f" % test_set_cross_entropy)


if __name__ == "__main__":
    main()