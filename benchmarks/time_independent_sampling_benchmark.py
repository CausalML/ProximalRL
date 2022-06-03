from collections import defaultdict
from itertools import product

import numpy as np

from benchmarks.abstract_benchmark import AbstractBenchmark


class TimeIndependentSamplingBenchmark(AbstractBenchmark):
    def __init__(self, num_a, horizon, gamma, ridge_alpha=1e-6,
                 num_batches=1000, batch_size=100000):
        self.num_o = None
        AbstractBenchmark.__init__(self, num_a, horizon, gamma)
        self.rho_list = [None for _ in range(self.horizon)]

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

    def estimate(self, pci_dataset, pi_e):
        n = pci_dataset.get_n()
        # obtain observation, action, target action, and reward arrays
        o_array = np.stack([pci_dataset.get_o_t(t-1)
                            for t in range(self.horizon+1)], axis=1)
        a_array = np.stack([pci_dataset.get_a_t(t)
                            for t in range(self.horizon)], axis=1)
        r_array = np.stack([pci_dataset.get_r_t(t)
                            for t in range(self.horizon)], axis=1)

        # iterate over batches
        r_t_sum = np.zeros(self.horizon)
        for _ in range(self.num_batches):
            # compile time independent samples
            idx = np.random.randint(low=0, high=n,
                                    size=(self.batch_size, self.horizon+1))
            z_sample = np.stack([o_array[idx[:, t+1], t]
                                 for t in range(self.horizon)], axis=1)
            x_sample = np.stack([o_array[idx[:, t], t+1]
                                 for t in range(self.horizon)], axis=1)
            a_sample = np.stack([a_array[idx[:, t+1], t]
                                 for t in range(self.horizon)], axis=1)
            r_sample = np.stack([r_array[idx[:, t+1], t]
                                 for t in range(self.horizon)], axis=1)

            # compute corresponding target action sample
            e_sample_list = []
            o_all = np.stack([o_array[idx[:, t], t]
                              for t in range(self.horizon+1)], axis=1)
            for t in range(self.horizon):
                o_t = o_all[:, t+1]
                prev_o_t = np.array(o_all[:, :t+1])
                prev_a_t = np.array(a_sample[:, :t])
                prev_r_t = np.array(r_sample[:, :t])
                e_sample_list.append(pi_e.get_e_t(
                    t=t, o_t=o_t, prev_o_t=prev_o_t, prev_a_t=prev_a_t,
                    prev_r_t=prev_r_t))
            e_sample = np.stack(e_sample_list, axis=1)

            # iterate over time points
            r_factor = np.ones(self.batch_size)
            for t in range(self.horizon):
                a_t = a_sample[:, t]
                e_t = e_sample[:, t]
                r_t = r_sample[:, t]
                z_t = z_sample[:, t]
                x_t = x_sample[:, t]
                rho_t = self.rho_list[t]
                r_factor = r_factor * ((a_t == e_t) * rho_t[z_t, a_t, x_t])
                # print(t, (r_factor * r_t).mean())
                r_t_sum[t] = r_t_sum[t] + (r_factor * r_t).mean()

        discount_factors = np.array([self.gamma ** i
                                     for i in range(self.horizon)])
        return (discount_factors * r_t_sum / self.num_batches).sum()

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
