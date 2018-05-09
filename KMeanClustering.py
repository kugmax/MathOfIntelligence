import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures


def generate_points(k, num_in_k):
    #np.random.seed(7)
    result = []

    for i in range(k):
        points = np.random.multivariate_normal([0 + i, 50 + i], [[1, 0], [1 + i, 50 + i]], num_in_k)
        result.extend(points)

    return result


def plot(k, clusters, centroids):
    colors = []
    for c in range(k):
        r, g, b = np.random.rand(1, 3)[0]
        colors.append((r, g, b))

    for c in clusters:
        plt.plot(c[1][0], c[1][1], '.', color=colors[c[0]])

    c_X = [centroids[i][0] for i in range(k)]
    c_Y = [centroids[i][1] for i in range(k)]

    plt.plot(c_X, c_Y, 'rx')
    plt.show()


def main():
    k = 8
    num_in_k = 1000
    error = 0.001

    points = generate_points(k, num_in_k)
    len_points = len(points)

    centroids = []
    for i in range(k):
        x = points[np.random.randint(0, len_points - 1)][0]
        y = points[np.random.randint(0, len_points - 1)][1]

        centroids.append(np.array([x, y]))

    executor = ProcessPoolExecutor()

    my_timer = MyTimer()

    splitted_points = np.split(np.array(points), k)
    while True:
        clusters = None
        old_centroids = centroids[:]
        my_timer.time_it("centroids copied")

        future_mean_cluster = {executor.submit(find_clusters, k, splitted_points[ci], centroids) for ci in range(0, len(splitted_points))}
        for future in concurrent.futures.as_completed(future_mean_cluster):
            data = future.result()
            if clusters is not None:
                clusters.extend(data)
            else:
                clusters = data

        if len(clusters) != len_points:
            print('WTF: ', len(clusters))
            print(clusters)
            return

        my_timer.time_it("cluster calculated")

        for ci in range(0, k):
            mean_c = mean_cluster(ci, clusters)
            if mean_c is not None:
                centroids[ci] = mean_c

        exit_dist = distance(np.array(centroids), np.array(old_centroids))  # parralel

        print(exit_dist)

        if exit_dist <= error:
            break

    my_timer.time_it("END")

    plot(k, clusters, centroids)


def find_clusters(k, points, centroids):
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

    return clusters


def distance(a, b):
    return np.linalg.norm(b - a)


def mean_cluster(c_id, clusters):
    sum_x = 0
    sum_y = 0
    size = 0
    for p in clusters:
        if p[0] == c_id:
            sum_x += p[1][0]
            sum_y += p[1][1]
            size += 1

    if size == 0:
        return None

    sum_x = sum_x / size
    sum_y = sum_y / size

    return np.array([sum_x, sum_y])


class MyTimer:
    def __init__(self):
        self.prev = 0
        self.curr = 0
        self.current_milli_time = lambda: int(round(time.time() * 1000))

    def time_it(self, s):
        self.curr = self.current_milli_time()
        delta = self.curr - self.prev
        print(delta, ' ' + s)
        self.prev = self.curr


if __name__ == "__main__":
    main()


