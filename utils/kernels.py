import random

import numpy as np
from scipy.spatial.distance import cdist


class AbstractKernel(object):
    def __init__(self, sq_dist_func):
        self.sq_dist_func = sq_dist_func

    def train(self, data):
        raise NotImplementedError()

    def __call__(self, x_1, x_2):
        NotImplementedError()


class GaussianKernel(AbstractKernel):
    def __init__(self, sigma, sq_dist_func):
        AbstractKernel.__init__(self, sq_dist_func)
        self.sigma = sigma

    def train(self, data):
        pass

    def __call__(self, x, y):
        sq_dists = self.sq_dist_func(x, y) ** 2
        return np.exp((-1 / (2 * self.sigma ** 2)) * sq_dists)


class PercentileKernel(GaussianKernel):
    def __init__(self, sq_dist_func, p, max_num_train=2500):
        GaussianKernel.__init__(self, None, sq_dist_func)
        self.p = p
        self.max_num_train = max_num_train

    def train(self, x):
        n = len(x)
        if n <= self.max_num_train:
            sq_dists = self.sq_dist_func(x, x).flatten()
        else:
            idx_1 = np.random.choice(n, self.max_num_train, replace=False)
            idx_2 = np.random.choice(n, self.max_num_train, replace=False)
            sq_dists = self.sq_dist_func(x[idx_1], x[idx_2]).flatten()

        gamma = np.percentile(sq_dists.flatten(), self.p) ** -1
        self.sigma = float((0.5 / gamma) ** 0.5)


class TripleMedianKernel(AbstractKernel):
    def __init__(self, sq_dist_func, max_num_train=2500):
        AbstractKernel.__init__(self, sq_dist_func)
        self.s_1, self.s_2, self.s_3 = None, None, None
        self.max_num_train = max_num_train

    def train(self, x):
        n = len(x)
        if n <= self.max_num_train:
            sq_dists = self.sq_dist_func(x, x).flatten()
        else:
            idx_1 = np.random.choice(n, self.max_num_train, replace=False)
            idx_2 = np.random.choice(n, self.max_num_train, replace=False)
            sq_dists = self.sq_dist_func(x[idx_1], x[idx_2]).flatten()
        median_d = np.median(sq_dists) ** 0.5
        self.s_1 = median_d
        self.s_2 = 0.1 * median_d
        self.s_3 = 10.0 * median_d

    def __call__(self, x, y):
        sq_dist = self.sq_dist_func(x, y)
        k_1 = np.exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        k_2 = np.exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        k_3 = np.exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        return (k_1 + k_2 + k_3) / 3


