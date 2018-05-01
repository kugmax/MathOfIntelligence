import matplotlib.pyplot as plt
import numpy as np

from Fao import Fao, FaoYearsColumnInfo


def main():
    gradient_decrees()


def step_gradient(b_current, m_current, points):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        t = y - ((m_current * x) + b_current)
        b_gradient += -(2 / N) * t
        m_gradient += -(2 / N) * x * t

    new_b = b_current - (b_gradient * 0.005)
    new_m = m_current - (m_gradient * 0.00000005)

    return [new_b, new_m]


def runner(points, starting_b, starting_m, num_iterations):
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

        b, m = step_gradient(b, m, points)

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


def gradient_decrees():
    fao = Fao('FAO.csv')
    rows = fao.get_all_country_supply(country="USA", product=2513)

    X = map(lambda string: float(string) if string.strip() else 0.0,
            rows[0][
            FaoYearsColumnInfo.BEGIN_INDEX: int(FaoYearsColumnInfo.BEGIN_INDEX) + int(FaoYearsColumnInfo.COUNT)])

    Y = map(lambda string: float(string) if string.strip() else 0.0,
            rows[1][
            FaoYearsColumnInfo.BEGIN_INDEX: int(FaoYearsColumnInfo.BEGIN_INDEX) + int(FaoYearsColumnInfo.COUNT)])

    points = [[]]
    for i in range(len(X)):
        points[i].extend([X[i], Y[i]])
        if i != len(X) - 1:
            points.append([])

    [b, m] = runner(points, 0, 0, 100000)
    e = compute_error_for_line_given_points(b, m, points)
    print(b, m)
    print(e)

    yhat = np.dot(m, X) + b
    plt.plot(X, yhat, 'r')

    plt.plot(X, Y, '.')
    plt.xlabel('Feed')
    plt.ylabel('Food')
    plt.show()


if __name__ == "__main__":
    main()
