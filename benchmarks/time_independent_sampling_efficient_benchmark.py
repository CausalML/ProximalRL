from collections import defaultdict
from itertools import product

import numpy as np

from benchmarks.abstract_benchmark import AbstractBenchmark


class TimeIndependentSamplingEfficientBenchmark(AbstractBenchmark):
    def __init__(self, num_a, horizon, gamma, ridge_alpha=1e-6,
                 num_batches=1000, batch_size=100000):
        self.num_o = None
        AbstractBenchmark.__init__(self, num_a, horizon, gamma)
        self.rho_list = [None for _ in range(self.horizon)]
        self.p_0 = None

        self.ridge_alpha = ridge_alpha
        self.num_batches = num_batches
        self.batch_size = batch_size

    def fit(self, pci_dataset):
        o_all = np.stack([pci_dataset.get_o_t(t) for t in range(self.horizon)],
                         axis=0)
        self.num_o = int(np.max(o_all)) + 1

        # fit rho functions
        for t in range(self.horizon):
            o_prev = pci_dataset.get_o_t(t-1)
            a_curr = pci_dataset.get_a_t(t)
            o_curr = pci_dataset.get_o_t(t)
            rho_t = self.fit_rho_t(o_prev, a_curr, o_curr)
            self.rho_list[t] = rho_t

        o_0 = pci_dataset.get_o_t(0)
        self.p_0 = self.fit_p_0(o_0)

    def estimate(self, pci_dataset, pi_e):
        r_t_mean = np.zeros(self.horizon)

        for t in range(self.horizon):
            # first compute final target
            r_t = pci_dataset.get_r_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            z_t = pci_dataset.get_o_t(t-1)
            rho_t = self.rho_list[t]
            target = ((r_t * (a_t == e_t)).reshape(-1, 1)
                      * rho_t[z_t, a_t, :]).mean(0)

            # next recursively compute targets backwards
            s_range = list(range(t))[::-1]
            for s in s_range:
                x_s = pci_dataset.get_o_t(s+1)
                a_s = pci_dataset.get_a_t(s)
                e_s = pci_dataset.get_e_t(s)
                z_s = pci_dataset.get_o_t(s-1)
                rho_s = self.rho_list[s]
                phi_s = target[x_s]
                target = ((phi_s * (a_s == e_s)).reshape(-1, 1)
                          * rho_s[z_s, a_s, :]).mean(0)
            r_t_mean[t] = (target * self.p_0).sum()

        discount_factors = np.array([self.gamma ** i
                                     for i in range(self.horizon)])
        return (discount_factors * r_t_mean).sum()

    def fit_rho_t(self, o_prev, a_curr, o_curr):
        # compute overall tuple counts
        counts = np.zeros((self.num_o, self.num_a, self.num_o))
        for o_prev_val, a_curr_val, o_curr_val in zip(o_prev, a_curr, o_curr):
            counts[o_prev_val, a_curr_val, o_curr_val] += 1

        # compute default distribution
        o_curr_counts = counts.sum((0, 1))
        default_dist = o_curr_counts / o_curr_counts.sum(keepdims=True)

        # compute p_inv
        p_inv = np.zeros((self.num_o, self.num_a, self.num_o))
        for o, a in product(range(self.num_o), range(self.num_a)):
            counts_sum = counts[o, a].sum()
            if counts_sum > 0:
                p_inv[o, a] = counts[o, a] / counts_sum
            else:
                p_inv[o, a] = default_dist

        # compute regulairzed inversion for each action
        p = np.zeros((self.num_o, self.num_a, self.num_o))
        for a in range(self.num_a):
            p_a_inv = p_inv[:, a, :].T
            reg = np.eye(p_a_inv.shape[1]) * self.ridge_alpha
            p_a = np.linalg.pinv(p_a_inv.T @ p_a_inv + reg) @ p_a_inv.T
            p[:, a, :] = p_a

        # compute probability distributions of o_curr and (a_curr, o_prev)
        counts_1 = np.zeros(self.num_o)
        for o_val in o_curr:
            counts_1[o_val] += 1
        q_1 = counts_1 / counts_1.sum(keepdims=True)

        counts_2 = np.zeros((self.num_o, self.num_a))
        for o_val, a_val in zip(o_prev, a_curr):
            counts_2[o_val, a_val] += 1
        q_2 = counts_2 / counts_2.sum(keepdims=True)

        return (p * q_1.reshape(1, 1, -1)
                / q_2.reshape(self.num_o, self.num_a, 1))

    def fit_p_0(self, o_0):
        p_0 = np.zeros(self.num_o)
        for o_val in o_0:
            p_0[o_val] += 1
        return p_0 / p_0.sum(keepdims=True)
