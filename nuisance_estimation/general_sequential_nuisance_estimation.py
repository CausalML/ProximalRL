import numpy as np

from nuisance_estimation.abstract_nuisance import AbstractNuisance


class GeneralSequentialNuisanceEstimation(AbstractNuisance):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, zxa_sq_dist,
                 wxa_sq_dist, horizon, gamma, num_a, q_class, q_args,
                 h_class, h_args):

        self.q_class = q_class
        self.q_args = q_args
        self.h_class = h_class
        self.h_args = h_args

        self.q_list = []
        self.h_list = []

        AbstractNuisance.__init__(
            self, horizon=horizon, gamma=gamma, num_a=num_a, embed_z=embed_z,
            embed_x=embed_x, embed_w=embed_w, embed_a=embed_a,
            zxa_sq_dist=zxa_sq_dist, wxa_sq_dist=wxa_sq_dist)

    def fit(self, pci_dataset):
        eta_list = []
        q_list = []
        min_r_list = []
        max_r_list = []
        n = pci_dataset.get_n()
        t_range = list(range(self.horizon))

        # first, fit the q functions one by one
        assert pci_dataset.get_horizon() == self.horizon
        eta_t = np.ones((n, 1))
        eta_list.append(eta_t)
        for t in t_range:
            z_t = pci_dataset.get_z_t(t)
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            e_t = pci_dataset.get_e_t(t)
            r_t = pci_dataset.get_r_t(t)
            eta_t = eta_list[t]

            # fit q
            q_estimator = self.q_class(
                embed_z=self.embed_z, embed_w=self.embed_w,
                embed_x=self.embed_x, embed_a=self.embed_a, num_a=self.num_a,
                zxa_sq_dist=self.zxa_sq_dist, wxa_sq_dist=self.wxa_sq_dist,
                **self.q_args)
            # print("t = %d, fitting q" % t)
            q_t = q_estimator.fit(eta_t=eta_t, z_t=z_t, w_t=w_t, x_t=x_t,
                                  a_t=a_t, e_t=e_t)
            self.q_list.append(q_t)

            # calculate next nu and eta
            q_t_array = q_t(z_t, x_t, a_t)
            # print(q_t_array.mean())
            eta_t = eta_t * q_t_array * (e_t == a_t).reshape(-1, 1)
            eta_list.append(eta_t)
            q_list.append(q_t_array)

            min_r_list.append(float(r_t.min()))
            max_r_list.append(float(r_t.max()))

        # next, fit the h functions backwards one by one
        reverse_h_list = []
        dfr_min = 0
        dfr_max = 0
        omega_t = np.zeros((n, 1))
        for t in t_range[::-1]:
            z_t = pci_dataset.get_z_t(t)
            w_t = pci_dataset.get_w_t(t)
            x_t = pci_dataset.get_x_t(t)
            a_t = pci_dataset.get_a_t(t)
            r_t = pci_dataset.get_r_t(t)
            e_t = pci_dataset.get_e_t(t)
            eta_t = eta_list[t]

            y_t = r_t.reshape(-1, 1) + self.gamma * omega_t
            dfr_min = min_r_list[t] + self.gamma * dfr_min
            dfr_max = max_r_list[t] + self.gamma * dfr_max

            # fit h
            h_estimator = self.h_class(
                embed_z=self.embed_z, embed_w=self.embed_w,
                embed_x=self.embed_x, embed_a=self.embed_a, num_a=self.num_a,
                zxa_sq_dist=self.zxa_sq_dist, wxa_sq_dist=self.wxa_sq_dist,
                **self.h_args)
            # print("t = %d, fitting h" % t)
            h_t = h_estimator.fit(
                eta_t=eta_t, e_t=e_t, y_t=y_t, z_t=z_t, w_t=w_t,
                x_t=x_t, a_t=a_t, dfr_min=dfr_min, dfr_max=dfr_max)
            reverse_h_list.append(h_t)
            # print(h_t(w_t, x_t, a_t).mean())

            # work out remainder for next calculation
            h_t_sum = np.zeros((n, 1))
            for a in range(self.num_a):
                a_const = np.array([a for _ in range(n)])
                h_t_sum = h_t_sum + h_t(w_t, x_t, a_const)
            mu_t = (a_t == e_t).reshape(-1, 1) * y_t

            omega_t = q_list[t] * (mu_t - h_t(w_t, x_t, a_t)) + h_t_sum
            # omega_t = q_list[t] * mu_t
            # omega_t = h_t_sum

        self.h_list = reverse_h_list[::-1]

    def q_t(self, t, z_t, x_t, a_t):
        if len(self.q_list) == 0:
            raise RuntimeError("Need to fit nuisances first")
        q_t = self.q_list[t]
        return q_t(z_t, x_t, a_t)

    def h_t(self, t, w_t, x_t, a_t):
        if len(self.h_list) == 0:
            raise RuntimeError("Need to fit nuisances first")
        h_t = self.h_list[t]
        return h_t(w_t, x_t, a_t)


class AbstractQEstimator(object):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a, zxa_sq_dist,
                 wxa_sq_dist):
        self.num_a = num_a
        self.embed_z = embed_z
        self.embed_w = embed_w
        self.embed_x = embed_x
        self.embed_a = embed_a
        self.zxa_sq_dist = zxa_sq_dist
        self.wxa_sq_dist = wxa_sq_dist

    def fit(self, eta_t, z_t, w_t, x_t, a_t, e_t):
        raise NotImplementedError()


class AbstractHEstimator(object):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a, zxa_sq_dist,
                 wxa_sq_dist):
        self.num_a = num_a
        self.embed_z = embed_z
        self.embed_w = embed_w
        self.embed_x = embed_x
        self.embed_a = embed_a
        self.zxa_sq_dist = zxa_sq_dist
        self.wxa_sq_dist = wxa_sq_dist

    def fit(self, eta_t, e_t, y_t, z_t, w_t, x_t, a_t, dfr_min, dfr_max):
        raise NotImplementedError()

