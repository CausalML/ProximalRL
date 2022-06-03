from collections import defaultdict
from itertools import product

import numpy as np

from benchmarks.abstract_benchmark import AbstractBenchmark


class TennenholtzBenchmark(AbstractBenchmark):
    def __init__(self, num_a, horizon, gamma, ridge_alpha=1e-6):
        self.num_o = None
        AbstractBenchmark.__init__(self, num_a, horizon, gamma)
        self.p_1_list = [None for _ in range(self.horizon)]
        self.p_2_list = [None for _ in range(self.horizon - 1)]
        self.p_r_list = [None for _ in range(self.horizon)]
        self.p_0 = None
        self.ridge_alpha = ridge_alpha

    def fit(self, pci_dataset):
        o_all = np.stack([pci_dataset.get_o_t(t) for t in range(self.horizon)],
                         axis=0)
        self.num_o = int(np.max(o_all)) + 1
        # print("num_0:", self.num_o)

        # fit first set of probability tables
        for t in range(self.horizon):
            o_prev = pci_dataset.get_o_t(t-1)
            a_curr = pci_dataset.get_a_t(t)
            o_curr = pci_dataset.get_o_t(t)
            p_1_t = self.fit_p_1(o_prev, a_curr, o_curr)
            self.p_1_list[t] = p_1_t

        # fit second set of probability tables
        for t in range(self.horizon-1):
            o_prev = pci_dataset.get_o_t(t-1)
            a_curr = pci_dataset.get_a_t(t)
            o_curr = pci_dataset.get_o_t(t)
            o_next = pci_dataset.get_o_t(t+1)
            p_2_t = self.fit_p_2(o_prev, a_curr, o_curr, o_next)
            self.p_2_list[t] = p_2_t

        # fit reward tables
        for t in range(self.horizon):
            o_prev = pci_dataset.get_o_t(t-1)
            a_curr = pci_dataset.get_a_t(t)
            o_curr = pci_dataset.get_o_t(t)
            r_curr = pci_dataset.get_r_t(t)
            p_r_t = self.fit_p_r(o_prev, a_curr, o_curr, r_curr)
            self.p_r_list[t] = p_r_t

        o_0 = pci_dataset.get_o_t(0)
        self.p_0 = self.fit_p_0(o_0)

    def estimate(self, pci_dataset, pi_e):
        # iterate over all possible combinations of observations
        o_range_list = [range(self.num_o) for _ in range(self.horizon)]
        r_t_mean = np.zeros(self.horizon)
        for o_list in product(*o_range_list):
            o_list = list(o_list)

            # compute corresponding list of actions
            a_list = []
            for t in range(self.horizon):
                o_t = np.array([o_list[t]])
                prev_o_t = np.array([[0] + o_list[:t]])
                prev_a_t = np.array([a_list])
                prev_r_t = None
                a_array = pi_e.get_e_t(t=t, o_t=o_t, prev_o_t=prev_o_t,
                                       prev_a_t=prev_a_t, prev_r_t=prev_r_t)
                a_list.append(int(a_array[0]))

            # perform recursive multiplication of Omega matrices
            a_0 = a_list[0]
            p_1_0 = self.p_1_list[0]
            omega = self.solve_approx(p_1_0[:, a_0, :].T, self.p_0)

            for t in range(self.horizon):
                p_1 = self.p_1_list[t]
                p_r = self.p_r_list[t]
                o_t = o_list[t]
                a_t = a_list[t]
                mean_r = (p_r[:, a_t, o_t] * p_1[:, a_t, o_t]) @ omega
                r_t_mean[t] += mean_r

                # if not at end of horizon compute next omega
                if t < self.horizon - 1:
                    p_1_next = self.p_1_list[t+1]
                    p_2 = self.p_2_list[t]
                    a_next = a_list[t+1]
                    omega = p_2[:, a_t, o_t, :].T @ omega
                    omega = self.solve_approx(p_1_next[:, a_next, :].T, omega)

        discount_factors = np.array([self.gamma ** i
                                     for i in range(self.horizon)])
        return (discount_factors * r_t_mean).sum()

    def solve_approx(self, a, b):
        reg = np.eye(a.shape[1]) * self.ridge_alpha
        # print((a.T @ a + reg).shape)
        return np.linalg.lstsq(a.T @ a + reg, a.T @ b, rcond=None)[0]

    def fit_p_1(self, o_prev, a_curr, o_curr):
        # compute overall tuple counts
        counts = np.zeros((self.num_o, self.num_a, self.num_o))
        for o_prev_val, a_curr_val, o_curr_val in zip(o_prev, a_curr, o_curr):
            counts[o_prev_val, a_curr_val, o_curr_val] += 1

        # compute default distribution
        o_curr_counts = counts.sum((0, 1))
        default_dist = o_curr_counts / o_curr_counts.sum(keepdims=True)

        # compute p_1
        p_1 = np.zeros((self.num_o, self.num_a, self.num_o))
        for o, a in product(range(self.num_o), range(self.num_a)):
            counts_sum = counts[o, a].sum()
            if counts_sum > 0:
                p_1[o, a] = counts[o, a] / counts_sum
            else:
                p_1[o, a] = default_dist
        return p_1

    def fit_p_2(self, o_prev, a_curr, o_curr, o_next):
        # compute overall tuple counts
        counts = np.zeros((self.num_o, self.num_a, self.num_o, self.num_o))
        for o_prev_val, a_curr_val, o_curr_val, o_next_val in \
                zip(o_prev, a_curr, o_curr, o_next):
            counts[o_prev_val, a_curr_val, o_curr_val, o_next_val] += 1

        # compute default distribution
        o_curr_counts = counts.sum((0, 1))
        default_dist = o_curr_counts / o_curr_counts.sum(keepdims=True)

        # compute p_2
        p_2 = np.zeros((self.num_o, self.num_a, self.num_o, self.num_o))
        for o, a in product(range(self.num_o), range(self.num_a)):
            counts_sum = counts[o, a].sum(keepdims=True)
            if counts_sum.sum() > 0:
                p_2[o, a] = counts[o, a] / counts_sum
            else:
                p_2[o, a] = default_dist
        return p_2

    def fit_p_r(self, o_prev, a_curr, o_curr, r_curr):
        # compute overall tuple counts
        counts = np.zeros((self.num_o, self.num_a, self.num_o))
        r_sum = np.zeros((self.num_o, self.num_a, self.num_o))
        for o_prev_val, a_curr_val, o_curr_val, r_curr_val in \
                zip(o_prev, a_curr, o_curr, r_curr):
            counts[o_prev_val, a_curr_val, o_curr_val] += 1
            r_sum[o_prev_val, a_curr_val, o_curr_val] += r_curr_val

        # compute default distribution
        default_r = float(r_curr.mean())

        # compute p_r
        p_r = np.zeros((self.num_o, self.num_a, self.num_o))
        for o, a, oo in product(range(self.num_o), range(self.num_a),
                                range(self.num_o)):
            if counts[o, a, oo] > 0:
                p_r[o, a, oo] = r_sum[o, a, oo] / counts[o, a, oo]
            else:
                p_r[o, a, oo] = default_r
        return p_r

    def fit_p_0(self, o_0):
        p_0 = np.zeros(self.num_o)
        for o_val in o_0:
            p_0[o_val] += 1
        return p_0 / p_0.sum(keepdims=True)
