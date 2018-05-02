import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrices
import warnings

np.random.seed(0)
tol = 1e-8
lam = None
max_iter = 20
r = 0.95
n = 1000
sigma = 1

beta_x, beta_z, beta_v = -4, 0.9, 1
var_x, var_z, var_v = 1, 1, 4

formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'


def catch_singularity(f):
    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Singular hessian')
            return args[0]

    return silencer


def main():
    x, z = np.random.multivariate_normal([0, 0], [[var_x, r], [r, var_z]], n).T
    v = np.random.normal(0, var_v, n) ** 3

    A = pd.DataFrame({'x': x, 'z': z, 'v': v})
    A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0, 1, n))

    A['y'] = [np.random.binomial(1, p) for p in A.log_odds]
    y, X = dmatrices(formula, A, return_type='dataframe')

    #print(y.head(10))
    #print(X.head(10))

    beta = np.zeros((len(X.columns), 1))
    iter_count = 0
    coefs_converge = False
    while not coefs_converge:
        beta_old = beta
        beta = alt_newton_step(beta, y, X, lam=lam)
        iter_count += 1
        coefs_converge = check_coefs_convergence(beta_old, beta, tol, iter_count)

    print('Iterations: {}'.format(iter_count))
    print('Beta: {}'.format(beta))


@catch_singularity
def newton_step(curr, y, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1-p))[:, 0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y - p)

    if lam:
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    beta = curr + step
    return beta


@catch_singularity
def alt_newton_step(curr, y, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1-p))[:, 0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y - p)

    if lam:
        step = np.dot(np.linalg.inv(hessian + lam * np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    beta = curr + step
    return beta


def check_coefs_convergence(beta_old, beta_new, tol, iters):
    coef_change = np.abs(beta_old - beta_new)
    return not (np.any(coef_change > tol) & (iters < max_iter))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    #my_main()
    main()