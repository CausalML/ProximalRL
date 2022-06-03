import numpy as np


class AbstractNuisance(object):
    def __init__(self, horizon, gamma, num_a, embed_z, embed_w, embed_x,
                 embed_a, zxa_sq_dist, wxa_sq_dist):
        self.horizon = horizon
        self.gamma = gamma
        self.num_a = num_a
        self.embed_z = embed_z
        self.embed_w = embed_w
        self.embed_x = embed_x
        self.embed_a = embed_a
        self.zxa_sq_dist = zxa_sq_dist
        self.wxa_sq_dist = wxa_sq_dist

    def fit(self, pci_dataset):
        raise NotImplementedError()

    def q_t(self, t, z_t, x_t, a_t):
        raise NotImplementedError()

    def h_t(self, t, w_t, x_t, a_t):
        raise NotImplementedError()

    def map_reduce(self, method, pci_dataset):
        """
        map-reduce method for computing mean of some function of trajectories
        """
        results = method(self, pci_dataset)
        return results.mean(0)


class CrossFitNuisance(AbstractNuisance):
    def __init__(self, horizon, gamma, num_a, embed_z, embed_w, embed_x,
                 embed_a, zxa_sq_dist, wxa_sq_dist,
                 base_nuisance_class, base_nuisance_args, num_folds):
        self.num_folds = num_folds
        assert issubclass(base_nuisance_class, AbstractNuisance)
        self.nuisance_versions = {}
        for i in range(num_folds):
            self.nuisance_versions[i] = base_nuisance_class(
                horizon=horizon, gamma=gamma, num_a=num_a, embed_z=embed_z,
                embed_w=embed_w, embed_x=embed_x, embed_a=embed_a,
                zxa_sq_dist=zxa_sq_dist, wxa_sq_dist=wxa_sq_dist,
                **base_nuisance_args)
        AbstractNuisance.__init__(self, horizon=horizon, gamma=gamma,
                                  num_a=num_a, embed_z=embed_z, embed_w=embed_w,
                                  embed_x=embed_x, embed_a=embed_a,
                                  zxa_sq_dist=zxa_sq_dist,
                                  wxa_sq_dist=wxa_sq_dist)

    def fit(self, pci_dataset):
        for i in range(self.num_folds):
            ex_fold_dataset = pci_dataset.get_ex_fold_dataset(i)
            self.nuisance_versions[i].fit(ex_fold_dataset)

    def map_reduce(self, method, pci_dataset):
        """
        re-implementation of the map-reduce method, where each nuisance
        version is only applied to out-of-fold data
        """
        results_list = []
        for i in range(self.num_folds):
            fold_dataset = pci_dataset.get_fold_dataset(i)
            fold_nuisance = self.nuisance_versions[i]
            results = method(fold_nuisance, fold_dataset)
            results_list.append(results)

        return np.concatenate(results_list, axis=0).mean(0)
