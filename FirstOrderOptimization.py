import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)


def init():
    return ln,


def update(frame):
    t, y = frame
    xdata.append(t)
    ydata.append(y)

    ln.set_data(xdata, ydata)
    return ln,


def main():
    X = []
    Y = []
    for i in range(-10, 10, 1):
        X.append(i)
        Y.append(func(i))

    plt.plot(X, Y)
    ani = FuncAnimation(fig, update, newtons_method(func, func_df, -10, 0.01),
                        init_func=init, blit=True)

    plt.show()


def func(x):
    return x ** 2


def func_df(x):
    h = 0.001
    return (func(x + h) - func(x)) / h


def dx(f, x):
    return abs(0 - f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0) / df(x0)
        delta = dx(f, x0)
        print(delta, e)

        yield x0, f(x0)

    print('Root is at: %0.4f' % x0)
    print("f(x) at root is: %0.4f" % f(x0))


if __name__ == "__main__":
    main()