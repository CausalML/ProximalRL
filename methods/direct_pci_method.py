import numpy as np

from methods.abstract_pci_method import AbstractPCIMethod


class DirectPCIMethod(AbstractPCIMethod):
    def __init__(self, nuisances, gamma, num_a, horizon):
        AbstractPCIMethod.__init__(self, nuisances, gamma, num_a, horizon)

    def psi_function(self, nuisances, pci_dataset):
        n = pci_dataset.get_n()
        w_0 = pci_dataset.get_w_t(0)
        x_0 = pci_dataset.get_x_t(0)
        psi = np.zeros((n, 1))
        for a in range(self.num_a):
            a_const = np.array([a for _ in range(n)])
            psi = psi + nuisances.h_t(0, w_0, x_0, a_const)
        norm = float(np.sum([self.gamma ** i for i in range(self.horizon)]))
        return psi.flatten() / norm
