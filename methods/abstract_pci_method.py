

class AbstractPCIMethod(object):
    def __init__(self, nuisances, gamma, num_a, horizon):
        self.nuisances = nuisances
        self.gamma = gamma
        self.num_a = num_a
        self.horizon = horizon

    def psi_function(self, nuisances, pci_dataset):
        raise NotImplementedError()

    def estimate_policy_value(self, pci_dataset):
        return self.nuisances.map_reduce(self.psi_function, pci_dataset)
