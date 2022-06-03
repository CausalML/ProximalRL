import numpy as np

from methods.abstract_pci_method import AbstractPCIMethod


class ImportanceSamplingPCIMethod(AbstractPCIMethod):
    def __init__(self, nuisances, gamma, num_a, horizon):
        AbstractPCIMethod.__init__(self, nuisances, gamma, num_a, horizon)

    def psi_function(self, nuisances, pci_dataset):
        horizon = pci_dataset.get_horizon()
        t_range = list(range(horizon))
        n = pci_dataset.get_n()
        eta_list = []

        # first compute nu and eta values
        for t in t_range:
            z_t = pci_dataset.get_z_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            eta_prev = eta_list[-1] if t > 0 else np.ones((n, 1))

            nu_t = eta_prev * nuisances.q_t(t, z_t, x_t, a_t)
            eta_list.append(nu_t * (e_t == a_t).reshape(-1, 1))

        # next compute psi recursively in reverse
        psi = np.zeros((n, 1))
        for t in t_range[::-1]:
            prev_psi = psi
            r_t = pci_dataset.get_r_t(t)
            eta_t = eta_list[t]
            psi = eta_t * r_t.reshape(-1, 1) + self.gamma * prev_psi

        norm = float(np.sum([self.gamma ** i for i in range(self.horizon)]))
        return psi.flatten() / norm
