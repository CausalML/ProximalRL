import numpy as np

from data.pci_dataset import PCITrajectory, PCIDataset
from environments.pci_reducer import AbstractPCIReducer


class AbstractEvaluationPolicy(object):
    def __init__(self, num_a):
        self.num_a = num_a

    def get_e_t(self, t, o_t, prev_o_t, prev_a_t, prev_r_t):
        raise NotImplementedError()


class AbstractEnvironment(object):
    def __init__(self, pci_reducer):
        assert isinstance(pci_reducer, AbstractPCIReducer)
        self.pci_reducer = pci_reducer

    def reset(self):
        raise NotImplementedError()

    def sample_observation(self):
        raise NotImplementedError()

    def sample_action(self):
        raise NotImplementedError()

    def transition_state(self, a):
        raise NotImplementedError()

    def embed_a(self, a):
        raise NotImplementedError()

    def embed_o(self, o):
        raise NotImplementedError()

    def get_num_a(self):
        raise NotImplementedError()

    def embed_z(self, z):
        return self.pci_reducer.embed_z(z, self)

    def embed_w(self, w):
        return self.pci_reducer.embed_w(w, self)

    def embed_x(self, x):
        return self.pci_reducer.embed_x(x, self)

    def get_z_len(self):
        return self.pci_reducer.get_z_len(self)

    def get_w_len(self):
        return self.pci_reducer.get_w_len(self)

    def sample_pci_trajectory(self, horizon, pi_e):
        assert isinstance(pi_e, AbstractEvaluationPolicy)
        o_list = []
        a_list = []
        r_list = []

        o_prior = self.reset()
        o_list.append(o_prior)
        for i in range(horizon):
            o = self.sample_observation()
            a = self.sample_action()
            r = self.transition_state(a)
            o_list.append(o)
            a_list.append(a)
            r_list.append(r)

        o = np.stack(o_list, axis=0)
        a = np.stack(a_list, axis=0)
        r = np.stack(r_list, axis=0)
        # print("o:", o)
        # print("a:", a)
        # print("r:", r)
        # print("")
        z = self.pci_reducer.compute_z(o, a, r)
        w = self.pci_reducer.compute_w(o, a, r)
        x = self.pci_reducer.compute_x(o, a, r)

        return PCITrajectory(o=o, z=z, w=w, x=x, a=a, r=r, horizon=horizon)

    def _est_pv_single(self, horizon, pi_e, gamma):
        assert isinstance(pi_e, AbstractEvaluationPolicy)
        r_list = []
        o_list = []
        a_list = []

        o_prior = self.reset()
        o_list.append(o_prior)
        for t in range(horizon):
            o = self.sample_observation()
            o_t = np.array([o])
            prev_o_t = np.array(o_list)
            prev_a_t = np.array(a_list)
            prev_r_t = np.array(r_list)
            a_array = pi_e.get_e_t(t=t, o_t=o_t, prev_o_t=prev_o_t,
                                   prev_a_t=prev_a_t, prev_r_t=prev_r_t)
            a = int(a_array[0])
            r = self.transition_state(a)
            o_list.append(o)
            a_list.append(a)
            r_list.append(r)

        discount_factors = gamma ** np.array(range(horizon))
        norm = discount_factors.sum()
        return float((np.array(r_list) * discount_factors).sum() / norm)

    def estimate_policy_value_oracle(self, horizon, pi_e, gamma, n):
        pv_estimates = [self._est_pv_single(horizon, pi_e, gamma)
                        for _ in range(n)]
        return float(np.array(pv_estimates).mean())

    def sample_pci_dataset(self, horizon, pi_e, n):
        trajectory_list = [self.sample_pci_trajectory(horizon, pi_e)
                           for _ in range(n)]
        return PCIDataset(trajectory_list, pi_e,
                          using_x=self.pci_reducer.using_x())
