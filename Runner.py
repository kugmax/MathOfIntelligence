import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from KMeanClustering import MyTimer

starting_b, starting_m, num_iterations = 0, 0, 10000
b_e, m_e = 0.005, 0.00000005


def main():
    my_timer = MyTimer()
    my_timer.time_it('begin')

    gradient_decrees(starting_b, starting_m, num_iterations, b_e, m_e)

    my_timer.time_it('end')


def step_gradient(b_current, m_current, points, b_e, m_e):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        t = y - ((m_current * x) + b_current)
        b_gradient += -(2 / N) * t
        m_gradient += -(2 / N) * x * t

    new_b = b_current - (b_gradient * b_e)
    new_m = m_current - (m_gradient * m_e)

    return [new_b, new_m]


def runner(points, starting_b, starting_m, num_iterations, b_e, m_e):
    b = starting_b
    m = starting_m

    best_result = dict()

    for i in range(0, num_iterations):
        err = None
        try:
            err = compute_error_for_line_given_points(b, m, points)
        except Exception:
            pass

        if err is not None:
            best_result[err] = [b, m]

        b, m = step_gradient(b, m, points, b_e, m_e)

    best_err = min(best_result.keys())
    best_b = best_result[best_err][0]
    best_m = best_result[best_err][1]

    print(len(best_result))
    print(best_result.keys())

    return [best_b, best_m]


def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def gradient_decrees(starting_b, starting_m, num_iterations, b_e, m_e):
    df = pd.read_csv(filepath_or_buffer='FAO.csv', encoding='cp1252')

    rows = df[(df['Area Abbreviation'] == 'FRA') & (df['Item Code'] == 2513)]\
        .fillna(0)\
        .values

    X = rows[0, 10:]
    Y = rows[1, 10:]

    print('X=', X)
    print('Y=', Y)

    points = [[]]
    for i in range(len(X)):
        points[i].extend([X[i], Y[i]])
        if i != len(X) - 1:
            points.append([])

    [b, m] = runner(points, starting_b, starting_m, num_iterations, b_e, m_e)
    e = compute_error_for_line_given_points(b, m, points)
    print('b =', b, 'm =', m, 'e =', e)

    # b = 31.738676103806664
    # m = -0.004120553354385847
    # e = 164.60410337853602

    yhat = np.dot(m, X) + b
    plt.plot(X, yhat, 'r')

    plt.plot(X, Y, '.')
    plt.xlabel('Feed')
    plt.ylabel('Food')
    plt.show()


if __name__ == "__main__":
    main()
