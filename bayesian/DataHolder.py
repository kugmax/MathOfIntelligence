import numpy as np
import matplotlib.pyplot as plt


class DataHolder:
    def __init__(self, data):
        self._data = data
        self._x_min, self._x_max = np.min(data[0]) - 1, np.max(data[0]) + 1
        self._y_min, self._y_max = np.min(data[1]) - 1, np.max(data[1]) + 1

    def plot_data(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.set_xlim(self._x_min, self._x_max)
        ax1.set_ylim(self._y_min, self._y_max)
        ax2.set_xlim(self._x_min, self._x_max)
        ax1.plot(self._data[0], self._data[1], 'ro')
        ax2.plot(self._data[0], self._data[1], 'ro')
        plt.show()

    def get_data(self):
        return self._data

    def get_x_min(self):
        return self._x_min

    def get_x_max(self):
        return self._x_max

    def get_y_min(self):
        return self._y_min

    def get_y_max(self):
        return self._y_max

