from collections import defaultdict
from itertools import product

import numpy as np

from benchmarks.abstract_benchmark import AbstractBenchmark


class MeanRBenchmark(AbstractBenchmark):
    def __init__(self, num_a, horizon, gamma):
        self.num_o = None
        AbstractBenchmark.__init__(self, num_a, horizon, gamma)

    def fit(self, pci_dataset):
        pass

    def estimate(self, pci_dataset, pi_e):
        est = 0
        t_range = list(range(self.horizon))[::-1]
        for t in t_range:
            est = float(np.mean(pci_dataset.get_r_t(t))) + self.gamma * est
        norm = np.sum([self.gamma ** i for i in range(self.horizon)])
        return float(est / norm)
