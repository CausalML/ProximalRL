from collections import defaultdict
from itertools import product

import numpy as np

from benchmarks.abstract_benchmark import AbstractBenchmark


class MDPTabularDirectBenchmark(AbstractBenchmark):
    def __init__(self, num_a, horizon, gamma):
        self.num_o = None
        AbstractBenchmark.__init__(self, num_a, horizon, gamma)

    def fit(self, pci_dataset):
        o_all = np.stack([pci_dataset.get_o_t(t) for t in range(self.horizon)],
                         axis=0)
        self.num_o = int(np.max(o_all)) + 1

    def estimate(self, pci_dataset, pi_e):

        # first compute final Q function at time H
        o_final = pci_dataset.get_o_t(self.horizon - 1)
        a_final = pci_dataset.get_a_t(self.horizon - 1)
        r_final = pci_dataset.get_r_t(self.horizon - 1)
        q_t = self._compute_mean_r_table(o_final, a_final, r_final)

        # next recursively compute earlier Q functions
        t_range = list(range(self.horizon - 1))[::-1]
        for t in t_range:
            o_t = pci_dataset.get_o_t(t)
            a_t = pci_dataset.get_a_t(t)
            os_t = pci_dataset.get_o_t(t + 1)
            r_t = pci_dataset.get_r_t(t)
            es_t = pci_dataset.get_e_t(t + 1)
            q_t = self._est_q_t(o_t, a_t, r_t, os_t, es_t, q_t)

        # finally, integrate over joint O,E distribution at T = 1
        o_1 = pci_dataset.get_o_t(0)
        e_1 = pci_dataset.get_e_t(0)
        oe_dist_1 = self._compute_oe_dist_unconditional(o_1, e_1)
        norm = np.sum([self.gamma ** i for i in range(self.horizon)])
        return float((q_t.flatten() * oe_dist_1).sum() / norm)

    def _compute_mean_r_table(self, o_t, a_t, r_t):
        r_lists = defaultdict(list)
        for o, a, r in zip(o_t, a_t, r_t):
            r_lists[(o, a)].append(r)
        mean_r = np.zeros((self.num_o, self.num_a))
        mean_r_default = float(r_t.mean())
        for o_i, a_i in product(range(self.num_o), range(self.num_a)):
            key = (o_i, a_i)
            if key in r_lists:
                mean_r[o_i, a_i] = float(np.mean(r_lists[key]))
            else:
                mean_r[o_i, a_i] = mean_r_default
        return mean_r

    def _compute_oe_dist(self, o_t, a_t, os_t, es_t):
        num_oa = self.num_o * self.num_a
        t_counts = np.zeros((self.num_o, self.num_a, self.num_o, self.num_a))
        for o, a, os, es in zip(o_t, a_t, os_t, es_t):
            t_counts[o, a, os, es] += 1
        t_counts = t_counts.reshape(self.num_o, self.num_a, num_oa)
        oe_dist = np.zeros((self.num_o, self.num_a, num_oa))
        default_dist = self._compute_oe_dist_unconditional(os_t, es_t)
        for o_i, a_i in product(range(self.num_o), range(self.num_a)):
            count_sum = t_counts[o_i, a_i].sum()
            if count_sum > 0:
                oe_dist[o_i, a_i] = t_counts[o_i, a_i] / float(count_sum)
            else:
                oe_dist[o_i, a_i] = default_dist
        return oe_dist.reshape(num_oa, num_oa)

    def _compute_oe_dist_unconditional(self, o_t, e_t):
        oe_counts = np.zeros((self.num_o, self.num_a))
        for o, e in zip(o_t, e_t):
            oe_counts[o, e] += 1
        dist = oe_counts.reshape(-1)
        return dist / dist.sum()

    def _est_q_t(self, o_t, a_t, r_t, os_t, es_t, q_t):
        r_table = self._compute_mean_r_table(o_t, a_t, r_t)
        oe_dist = self._compute_oe_dist(o_t, a_t, os_t, es_t)
        q_future = (oe_dist @ q_t.flatten()).reshape(self.num_o, self.num_a)
        return r_table + self.gamma * q_future
