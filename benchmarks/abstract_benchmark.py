

class AbstractBenchmark(object):
    def __init__(self, num_a, horizon, gamma):
        self.num_a = num_a
        self.horizon = horizon
        self.gamma = gamma

    def fit(self, pci_dataset):
        raise NotImplementedError()

    def estimate(self, pci_dataset, pi_e):
        raise NotImplementedError()
