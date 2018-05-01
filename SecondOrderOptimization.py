import numpy as np
import matplotlib.pyplot as plt


def taylor2(f, x, x0):

    s1 = f(x0)
    s2 = func_df(f, x0) * (x - x0)
    s3 = (func_df2(f, x0) / 1 * 2) * ((x - x0) ** 2)

    return s1 + s2 + s3


def newtons_method2(f, x0, e=0.01):
    error = 10
    while error > e:
        l1 = f(x0)

        df = func_df(f, x0)
        df2 = func_df2(f, x0)
        x0 = x0 - df / df2

        l2 = f(x0)
        error = np.abs(l2 - l1)

        print(l1, l2, error)
    return x0, error


def main():
    X = np.arange(-10, 10, 0.1)
    Y = [func(i) for i in X]

    plt.plot(X, Y)

    xmin, error = newtons_method2(func, x0=-10)

    print("x0: %0.2f, error: %0.2f:  " % (xmin, error))

    #plt.plot(X, [func_df(func, x) for x in X], 'g')
    #plt.plot(X, [func_df2(func, x) for x in X], 'g--')

    plt.plot([xmin], [func(xmin)], 'ro')
    plt.show()


def func(x):
    return (np.e ** -x) * np.sin(x)
    #return x**2 - 20


def func_df(f, x):
    h = 0.001
    return (f(x + h) - f(x)) / h


def func_df2(f, x):
    h = 0.001
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


if __name__ == "__main__":
    main()