import matplotlib.pyplot as plt
from base64 import b64decode
from json import loads
import numpy as np


def generate_points():
    #np.random.seed(7)
    result = []

    k1 = np.random.multivariate_normal([0, 10], [[1, 0], [1, 10]], 100)
    k2 = np.random.multivariate_normal([5, 0], [[1, 0], [1, 10]], 100)
    k3 = np.random.multivariate_normal([15, 15], [[1, 0], [1, 10]], 100)

    result.extend(k1)
    result.extend(k2)
    result.extend(k3)

    return result


def main():
    points = generate_points()

    k = 3
    error = 0

    centroids = []
    for i in range(k):
        x = points[np.random.randint(0, 299)][0]
        y = points[np.random.randint(0, 299)][1]

        centroids.append(np.array([x, y]))

    c_X = [centroids[i][0] for i in range(k)]
    c_Y = [centroids[i][1] for i in range(k)]

    plt.plot(c_X, c_Y, 'rx')
    plt.show()

    np.random.shuffle(points)

    while True:
        clusters = []
        for p in points:
            min_dist = None
            min_ci = None
            for ci in range(0, k):
                dist = distance(p, centroids[ci])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_ci = ci

            clusters.append([min_ci, p])

        old_centroids = centroids[:]

        for ci in range(0, k):
            sum_x = 0
            sum_y = 0
            size = 0
            for p in clusters:
                if p[0] == ci:
                    sum_x += p[1][0]
                    sum_y += p[1][1]
                    size += 1

            sum_x = sum_x / size
            sum_y = sum_y / size

            centroids[ci] = np.array([sum_x, sum_y])

        exit_dist = distance(np.array(centroids), np.array(old_centroids))

        print(exit_dist)

        if exit_dist <= error:
            break

    colors = ['b', 'g', 'y']

    for c in clusters:
        plt.plot(c[1][0], c[1][1], '.' + colors[c[0]])

    c_X = [centroids[i][0] for i in range(k)]
    c_Y = [centroids[i][1] for i in range(k)]

    plt.plot(c_X, c_Y, 'rx')
    plt.show()


def parse(x):
    digit = loads(x)
    array = np.fromstring(b64decode(digit["data"]), dtype=np.ubyte)
    array = array.astype(np.float64)
    return (digit["label"], array)


with open("digits.base64.json","r") as f:
    digits = map(parse, f.readlines())


def distance(a, b):
    return np.linalg.norm(b - a)



"""
d = digits.__next__()
result = d[1].reshape(img_size, img_size)
print(result)

plt.figure()
fig = plt.imshow(result)
fig.set_cmap('gray_r')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
"""

if __name__ == "__main__":
    main()


