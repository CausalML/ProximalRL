import itertools
import random
from collections import defaultdict
from functools import partial

import numpy as np


class PCITrajectory(object):
    """
    Represents a single trajectory, in PCI reduction format
    """
    def __init__(self, o, z, w, x, a, r, horizon):
        """
        Z, W, A, R should all be numpy arrays whose first dimension is equal to
        horizon
        """
        assert z.shape[0] == w.shape[0] == a.shape[0] == r.shape[0] == horizon
        assert isinstance(o, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        self.o = o
        self.z = z
        self.w = w
        self.x = x
        self.a = a
        self.r = r
        self.horizon = horizon

    def get_o_t_prev(self, t):
        return self.o[:t]

    def get_a_t_prev(self, t):
        return self.a[:t]

    def get_r_t_prev(self, t):
        return self.r[:t]

    def get_o_t(self, t):
        return self.o[t+1]

    def get_z_t(self, t):
        return self.z[t]

    def get_w_t(self, t):
        return self.w[t]

    def get_x_t(self, t):
        if self.x is not None:
            return self.x[t]
        else:
            return None

    def get_a_t(self, t):
        return self.a[t]

    def get_r_t(self, t):
        return self.r[t]

    def get_horizon(self):
        return self.horizon


class PCIDataset(object):
    """
    Represents an entire dataset, made up of collection of PCI trajectories,
    togther with machinery for cross-fitting
    """
    def __init__(self, trajectory_list, pi_e, using_x=True):
        for tau in trajectory_list:
            assert isinstance(tau, PCITrajectory)
        self.trajectory_list = trajectory_list
        horizon_lens = [tau_.get_horizon() for tau_ in self.trajectory_list]
        assert len(set(horizon_lens)) == 1
        self.horizon = horizon_lens[0]
        self.pi_e = pi_e

        self.num_folds = None
        self.fold_indices = None
        self.inv_fold_index = None

        self.e_t_list = None
        self._setup_evaluation_policy(pi_e)

        self.using_x = using_x

    def make_folds(self, num_folds):
        self.num_folds = num_folds
        indices = list(range(len(self.trajectory_list)))
        random.shuffle(indices)
        index_cycle = itertools.cycle(range(num_folds))
        self.fold_indices = {f_: [] for f_ in range(num_folds)}
        self.inv_fold_index = {}
        for idx in indices:
            fold = next(index_cycle)
            self.fold_indices[fold].append(idx)
            self.inv_fold_index[idx] = fold
        for idx_list in self.fold_indices.values():
            idx_list.sort()

    def _setup_evaluation_policy(self, pi_e):
        self.e_t_list = []
        for t in range(self.horizon):
            o_t = self._get_data_t_abstract(t, PCITrajectory.get_o_t)
            prev_o_t = self._get_data_t_abstract(t, PCITrajectory.get_o_t_prev)
            prev_a_t = self._get_data_t_abstract(t, PCITrajectory.get_a_t_prev)
            prev_r_t = self._get_data_t_abstract(t, PCITrajectory.get_r_t_prev)
            e_t = pi_e.get_e_t(t, o_t, prev_o_t, prev_a_t, prev_r_t)
            self.e_t_list.append(e_t)

    def get_n(self):
        return len(self.trajectory_list)

    def get_num_folds(self):
        return self.num_folds

    def get_horizon(self):
        return self.horizon

    def get_tau_list(self):
        return self.trajectory_list

    def _get_data_t_abstract(self, t, get_data_t_method):
        get_func = partial(get_data_t_method, t=t)
        return np.stack([get_func(tau_) for tau_ in self.trajectory_list],
                        axis=0)

    def get_z_t(self, t):
        return self._get_data_t_abstract(t, PCITrajectory.get_z_t)

    def get_w_t(self, t):
        return self._get_data_t_abstract(t, PCITrajectory.get_w_t)

    def get_x_t(self, t):
        if self.using_x:
            return self._get_data_t_abstract(t, PCITrajectory.get_x_t)
        else:
            return None

    def get_a_t(self, t):
        return self._get_data_t_abstract(t, PCITrajectory.get_a_t)

    def get_r_t(self, t):
        return self._get_data_t_abstract(t, PCITrajectory.get_r_t)

    def get_o_t(self, t):
        return self._get_data_t_abstract(t, PCITrajectory.get_o_t)

    def get_e_t(self, t):
        if self.e_t_list is None:
            raise RuntimeError("Need to register evaluation policy first")
        return self.e_t_list[t]

    def get_fold_dataset(self, fold_num):
        if self.num_folds is None:
            raise RuntimeError("Need to run method 'make_folds' first")
        assert fold_num in self.fold_indices.keys()
        tau_list = [self.trajectory_list[i_]
                    for i_ in self.fold_indices[fold_num]]
        return PCIDataset(tau_list, self.pi_e, using_x=self.using_x)

    def get_ex_fold_dataset(self, fold_num):
        if self.num_folds is None:
            raise RuntimeError("Need to run method 'make_folds' first")
        assert fold_num in self.fold_indices.keys()
        ex_fold_idx = []
        for f_, idx_ in sorted(self.fold_indices.items()):
            if f_ != fold_num:
                ex_fold_idx.extend(idx_)
        tau_list = [self.trajectory_list[i_] for i_ in ex_fold_idx]
        return PCIDataset(tau_list, self.pi_e, using_x=self.using_x)
