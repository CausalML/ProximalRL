import numpy as np

from methods.abstract_pci_method import AbstractPCIMethod


class EfficientPCIMethod(AbstractPCIMethod):
    def __init__(self, nuisances, gamma, num_a, horizon):
        AbstractPCIMethod.__init__(self, nuisances, gamma, num_a, horizon)

    def psi_function(self, nuisances, pci_dataset):
        horizon = pci_dataset.get_horizon()
        t_range = list(range(horizon))
        n = pci_dataset.get_n()
        nu_list = []
        eta_list = []

        # first compute nu and eta values
        for t in t_range:
            z_t = pci_dataset.get_z_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            eta_prev = eta_list[-1] if t > 0 else np.ones((n, 1))

            nu_list.append(eta_prev * nuisances.q_t(t, z_t, x_t, a_t))
            eta_list.append(nu_list[-1] * (e_t == a_t).reshape(-1, 1))

        # next compute psi recursively in reverse
        psi = np.zeros((n, 1))
        for t in t_range[::-1]:
            prev_psi = psi
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            r_t = pci_dataset.get_r_t(t)
            nu_t = nu_list[t]
            eta_t = eta_list[t]
            eta_next = eta_list[t-1] if t > 0 else np.ones((n, 1))

            h_t = nuisances.h_t(t, w_t, x_t, a_t)
            psi = eta_t * r_t.reshape(-1, 1) - nu_t * h_t
            for a in range(self.num_a):
                a_const = np.array([a for _ in range(n)])
                psi = psi + eta_next * nuisances.h_t(t, w_t, x_t, a_const)
            psi = psi + self.gamma * prev_psi

        norm = float(np.sum([self.gamma ** i for i in range(self.horizon)]))
        return psi.flatten() / norm
