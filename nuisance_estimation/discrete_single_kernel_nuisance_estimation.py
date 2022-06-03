from functools import partial
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

from nuisance_estimation.general_sequential_nuisance_estimation import \
    AbstractQEstimator, AbstractHEstimator, GeneralSequentialNuisanceEstimation
from utils.np_utils import one_hot_embed
from utils.torch_utils import torch_to_np, np_to_tensor


class DiscreteSingleKNuisanceEstimation(GeneralSequentialNuisanceEstimation):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, zxa_sq_dist,
                 wxa_sq_dist, horizon, gamma, num_a, q_net_class, q_net_args,
                 g_kernel_class, g_kernel_args, h_net_class, h_net_args,
                 f_kernel_class, f_kernel_args, q_alpha, h_alpha,
                 q_lmbda, h_lmbda, num_rep=2):
        q_class = SingleKernelQEstimation
        h_class = SingleKernelHEstimation
        q_args = {
            "q_net_class": q_net_class,
            "q_net_args": q_net_args,
            "g_kernel_class": g_kernel_class,
            "g_kernel_args": g_kernel_args,
            "alpha": q_alpha,
            "lmbda": q_lmbda,
            "num_rep": num_rep,
        }
        h_args = {
            "h_net_class": h_net_class,
            "h_net_args": h_net_args,
            "f_kernel_class": f_kernel_class,
            "f_kernel_args": f_kernel_args,
            "alpha": h_alpha,
            "lmbda": h_lmbda,
            "num_rep": num_rep
        }
        GeneralSequentialNuisanceEstimation.__init__(
            self, embed_z=embed_z, embed_w=embed_w, embed_x=embed_x,
            embed_a=embed_a, zxa_sq_dist=zxa_sq_dist, wxa_sq_dist=wxa_sq_dist,
            horizon=horizon, gamma=gamma, num_a=num_a, q_class=q_class,
            q_args=q_args, h_class=h_class, h_args=h_args)


def hash_np(val):
    if isinstance(val, np.ndarray):
        return tuple(val.flatten())
    else:
        return val


def get_unique_vals_codes(data):
    data_hashed = [hash_np(v_) for v_ in data]
    vals = sorted(set(data_hashed))
    reverse_dict = {x_: i_ for i_, x_ in enumerate(vals)}
    codes = [reverse_dict[x_] for x_ in data_hashed]
    return np.array(vals), np.array(codes), reverse_dict


def compute_triplet_embeddings(zxa_vals, wxa_vals, z_t, w_t, x_t, num_a,
                               embed_z, embed_w, embed_x, embed_a):
    z_vals, _, z_map = get_unique_vals_codes(z_t)
    w_vals, _, w_map = get_unique_vals_codes(w_t)
    a_vals, _, a_map = get_unique_vals_codes(np.arange(num_a))
    z_vals_embed = embed_z(z_vals)
    w_vals_embed = embed_w(w_vals)
    a_vals_embed = embed_a(a_vals)

    # pre-compute all required embeddings
    if x_t is None:
        zxa_z = [z_map[z_] for z_, _ in zxa_vals]
        zxa_a = [a_map[a_] for _, a_ in zxa_vals]
        zxa_vals_embed = np.concatenate([z_vals_embed[zxa_z],
                                         a_vals_embed[zxa_a]], axis=1)
        wxa_w = [w_map[w_] for w_, _ in wxa_vals]
        wxa_a = [a_map[a_] for _, a_ in wxa_vals]
        wxa_vals_embed = np.concatenate([w_vals_embed[wxa_w],
                                         a_vals_embed[wxa_a]], axis=1)
    else:
        x_vals, _, x_map = get_unique_vals_codes(x_t)
        x_vals_embed = embed_x(x_vals)
        zxa_z = [z_map[z_] for z_, _, _ in zxa_vals]
        zxa_x = [x_map[x_] for _, x_, _ in zxa_vals]
        zxa_a = [a_map[a_] for _, _, a_ in zxa_vals]
        zxa_vals_embed = np.concatenate([z_vals_embed[zxa_z],
                                         x_vals_embed[zxa_x],
                                         a_vals_embed[zxa_a]], axis=1)
        wxa_w = [w_map[w_] for w_, _, _ in wxa_vals]
        wxa_x = [x_map[x_] for _, x_, _ in wxa_vals]
        wxa_a = [a_map[a_] for _, _, a_ in wxa_vals]
        wxa_vals_embed = np.concatenate([w_vals_embed[wxa_w],
                                         x_vals_embed[wxa_x],
                                         a_vals_embed[wxa_a]], axis=1)
    a_len = a_vals_embed.shape[1]
    return zxa_vals_embed, wxa_vals_embed, a_len


class SingleKernelQEstimation(AbstractQEstimator):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a,
                 zxa_sq_dist, wxa_sq_dist, q_net_class, q_net_args,
                 g_kernel_class, g_kernel_args, alpha, lmbda, num_rep=2,
                 cuda=False, device=None):
        AbstractQEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_x=embed_x, embed_a=embed_a,
                                    num_a=num_a, zxa_sq_dist=zxa_sq_dist,
                                    wxa_sq_dist=wxa_sq_dist)

        self.q_net_class = q_net_class
        self.q_net_args = q_net_args
        self.g_kernel = g_kernel_class(sq_dist_func=wxa_sq_dist,
                                       **g_kernel_args)

        self.alpha = alpha
        self.lmbda = lmbda
        self.num_rep = num_rep
        self.cuda = cuda
        self.device = device

    def fit(self, eta_t, z_t, w_t, x_t, a_t, e_t):
        alpha = self.alpha
        while True:
            try:
                q_net, norm = self._try_fit_internal(
                    eta_t, z_t, w_t, x_t, a_t, e_t, alpha)
                did_succeed = (norm != 0) and q_net.is_finite()
            except Exception as e:
                did_succeed = False

            if did_succeed:
                break
            elif alpha == 0:
                alpha = 1e-10
            elif alpha > 10e10:
                return self.default_q_func
            else:
                alpha *= 10

        return partial(self.q_func, q_net=q_net, norm=norm)

    def q_func(self, z_t, x_t, a_t, q_net, norm):
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            zxa_vals_embed = np.concatenate([self.embed_z(zxa_vals[:, 0]),
                                             self.embed_a(zxa_vals[:, 1])],
                                            axis=1)
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            zxa_vals_embed = np.concatenate([self.embed_z(zxa_vals[:, 0]),
                                             self.embed_x(zxa_vals[:, 1]),
                                             self.embed_a(zxa_vals[:, 2])],
                                            axis=1)

        q_net_vals = q_net(self._to_tensor(zxa_vals_embed)) / norm
        return torch_to_np(q_net_vals)[zxa_codes]

    def default_q_func(self, z_t, x_t, a_t):
        n = len(z_t)
        return np.ones((n, 1)) * self.num_a

    def _try_fit_internal(self, eta_t, z_t, w_t, x_t, a_t, e_t, alpha):
        # compute codes for each of z, w, x
        n = len(z_t)
        eta_t = eta_t / eta_t.mean()
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            wa_data = [(w_, a_) for w_ in w_t for a_ in range(self.num_a)]
            wxa_vals, _, wxa_map = get_unique_vals_codes(wa_data)
            wxa_codes = np.array([wxa_map[tuple(wa_)] for wa_ in zip(w_t, a_t)])
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            wxa_data = [(w_, x_, a_) for w_, x_ in zip(w_t, x_t)
                        for a_ in range(self.num_a)]
            wxa_vals, _, wxa_map = get_unique_vals_codes(wxa_data)
            wxa_codes = np.array([wxa_map[tuple(wxa_)]
                                  for wxa_ in zip(w_t, x_t, a_t)])

        zxa_vals_embed, wxa_vals_embed, a_len = compute_triplet_embeddings(
            zxa_vals, wxa_vals, z_t, w_t, x_t, self.num_a,
            self.embed_z, self.embed_w, self.embed_x, self.embed_a)
        zx_len = zxa_vals_embed.shape[-1] - a_len
        zxa_vals_torch = self._to_tensor(zxa_vals_embed)

        # train kernel hyper-parameters
        self.g_kernel.train(wxa_vals)

        # set up basic matrices / data structures that are constant for each
        # iteration
        k_1 = self.g_kernel(wxa_vals, wxa_vals) + 1e-3 * np.eye(len(wxa_vals))
        k_2 = k_1[wxa_codes]
        k_3 = np.zeros_like(k_2)
        for a in range(self.num_a):
            if x_t is None:
                wxa_codes_a = np.array([wxa_map[(w_, a)] for w_ in w_t])
            else:
                wxa_codes_a = np.array([wxa_map[(w_, x_, a)]
                                        for w_, x_ in zip(w_t, x_t)])
            k_3 = k_3 + k_1[wxa_codes_a]

        zxa_ind = one_hot_embed(zxa_codes, num_categories=len(zxa_vals))
        b = self._to_tensor((1/n) * k_2.T @ (eta_t * zxa_ind))
        c = self._to_tensor((1/n) * k_3.T @ eta_t.flatten())
        e_a_t = (e_t == a_t).astype(int).reshape(-1, 1)
        freqs = self._to_tensor((eta_t.reshape(-1, 1) * zxa_ind).mean(0))
        reg_freqs = self._to_tensor((eta_t.reshape(-1, 1) * e_a_t
                                     * zxa_ind).mean(0))
        obs_freqs = self._to_tensor(zxa_ind.mean(0))

        # setup q network
        q_net = self.q_net_class(**self.q_net_args, x_len=zx_len, a_len=a_len,
                                 num_a=self.num_a)

        for rep_i in range(self.num_rep):
            if rep_i > 0:
                q_tilde_vals = torch_to_np(q_net(zxa_vals_torch))
                q_tilde = q_tilde_vals[zxa_codes]
            else:
                q_tilde = np.ones((n, 1)) * self.num_a
            q_fact = eta_t * (k_2 * q_tilde - k_3)
            q = (1/n) * q_fact.T @ q_fact + alpha * k_1
            q_inv = self._to_tensor(np.linalg.inv(q))
            q_inv = (q_inv + q_inv.T) / 2

            optimizer = torch.optim.LBFGS(q_net.parameters())

            def closure():
                optimizer.zero_grad()
                q_net_vals = q_net(zxa_vals_torch).flatten()
                rho = b @ q_net_vals - c
                # reg_0 = self.lmbda[0] * (q_net_vals ** 2) @ reg_freqs
                if isinstance(self.lmbda, tuple):
                    l0, l1, l2 = self.lmbda
                else:
                    l0, l1, l2 = self.lmbda, 0, 0
                reg_0 = l0 * (q_net_vals ** 2) @ obs_freqs
                # sqb_list = []
                # for freqs in freqs_list:
                #     sqb_list.append((q_net_vals @ freqs - 1.0) ** 2)
                # reg_1 = l1 * torch.stack(sqb_list).mean()
                # reg_1 = l1 * (reg_freqs @ q_net_vals - 1.0) ** 2
                reg_1 = l1 * (freqs @ q_net_vals - self.num_a) ** 2
                reg_2 = l2 * (reg_freqs @ q_net_vals - 1.0) ** 2
                # q_net_excess = F.relu(-q_net_vals) ** 2
                # reg_1 = l1 * q_net_excess @ freqs
                loss = rho.T @ q_inv @ rho + reg_0 + reg_1 + reg_2
                loss.backward()
                return loss
            optimizer.step(closure)

        q_net_vals = q_net(zxa_vals_torch).flatten()
        rho = b @ q_net_vals - c
        loss = float(rho.T @ q_inv @ rho)

        q_vals = q_net(zxa_vals_torch).detach().flatten()
        # print(np.array(sorted(torch_to_np(q_vals))))
        mean_q = float(freqs @ q_vals)
        std_q = float(freqs @ ((q_vals - mean_q) ** 2))
        # print("mean q:", mean_q, "std q:", std_q, "loss:", loss)
        # print(float(reg_freqs @ q_vals))
        reg_mean_q = float(reg_freqs @ q_vals)
        q_abs_mean = np.abs(q_vals).mean()
        if np.isclose(q_abs_mean, 0.0):
            norm = 0
        elif np.isclose(reg_mean_q, 0.0):
            norm = 1.0
        else:
            norm = reg_mean_q
        return q_net, norm

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class SingleKernelHEstimation(AbstractHEstimator):
    def __init__(self, embed_z, embed_w, embed_x, embed_a, num_a,
                 zxa_sq_dist, wxa_sq_dist, h_net_class, h_net_args,
                 f_kernel_class, f_kernel_args, alpha, lmbda, num_rep=2,
                 cuda=False, device=None):
        AbstractHEstimator.__init__(self, embed_z=embed_z, embed_w=embed_w,
                                    embed_x=embed_x, embed_a=embed_a,
                                    num_a=num_a, zxa_sq_dist=zxa_sq_dist,
                                    wxa_sq_dist=wxa_sq_dist)

        self.h_net_class = h_net_class
        self.h_net_args = h_net_args
        self.f_kernel = f_kernel_class(sq_dist_func=zxa_sq_dist,
                                       **f_kernel_args)

        self.alpha = alpha
        self.lmbda = lmbda
        self.num_rep = num_rep
        self.cuda = cuda
        self.device = device

    def fit(self, eta_t, e_t, y_t, z_t, w_t, x_t, a_t, dfr_min, dfr_max):
        alpha = self.alpha
        while True:
            try:
                h_net, norm = self._try_fit_internal(
                    eta_t, e_t, y_t, z_t, w_t, x_t, a_t,
                    dfr_min, dfr_max, alpha)
                did_succeed = h_net.is_finite()
            except Exception as e:
                did_succeed = False

            if did_succeed:
                break
            elif alpha == 0:
                alpha = 1e-10
            elif alpha > 10e10:
                target = y_t * (e_t == a_t).reshape(-1, 1)
                target_mean = float((target * eta_t).mean())
                return partial(self.default_h_func, target_mean=target_mean)
            else:
                alpha *= 10

        return partial(self.h_func, h_net=h_net, norm=norm)

    def h_func(self, w_t, x_t, a_t, h_net, norm):
        if x_t is None:
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, a_t))
            wxa_vals_embed = np.concatenate([self.embed_w(wxa_vals[:, 0]),
                                             self.embed_a(wxa_vals[:, 1])],
                                            axis=1)
        else:
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, x_t, a_t))
            wxa_vals_embed = np.concatenate([self.embed_w(wxa_vals[:, 0]),
                                             self.embed_x(wxa_vals[:, 1]),
                                             self.embed_a(wxa_vals[:, 2])],
                                            axis=1)

        h_net_vals = h_net(self._to_tensor(wxa_vals_embed)) / norm
        return torch_to_np(h_net_vals)[wxa_codes]

    def default_h_func(self, w_t, x_t, a_t, target_mean):
        n = len(w_t)
        return np.ones((n, 1)) * target_mean

    def _try_fit_internal(self, eta_t, e_t, y_t, z_t, w_t, x_t, a_t,
                          dfr_min, dfr_max, alpha):
        # compute codes for each of z, w, x
        n = len(w_t)
        eta_t = eta_t / eta_t.mean()
        if x_t is None:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, a_t))
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, a_t))
        else:
            zxa_vals, zxa_codes, _ = get_unique_vals_codes(zip(z_t, x_t, a_t))
            wxa_vals, wxa_codes, _ = get_unique_vals_codes(zip(w_t, x_t, a_t))

        zxa_vals_embed, wxa_vals_embed, a_len = compute_triplet_embeddings(
            zxa_vals, wxa_vals, z_t, w_t, x_t, self.num_a,
            self.embed_z, self.embed_w, self.embed_x, self.embed_a)
        wx_len = wxa_vals_embed.shape[-1] - a_len
        wxa_vals_torch = self._to_tensor(wxa_vals_embed)

        # train kernel hyper-parameters
        self.f_kernel.train(zxa_vals)

        # set up basic matrices / data structures that are constant for each
        # iteration
        k_1 = self.f_kernel(zxa_vals, zxa_vals) + 1e-3 * np.eye(len(zxa_vals))
        k_2 = k_1[zxa_codes]

        wxa_ind = one_hot_embed(wxa_codes, num_categories=len(wxa_vals))
        target = y_t * (e_t == a_t).reshape(-1, 1)
        b = self._to_tensor((1/n) * k_2.T @ (eta_t * wxa_ind))
        c = self._to_tensor((1/n) * k_2.T @ (eta_t * target).flatten())
        reg_freqs = self._to_tensor((eta_t * wxa_ind).mean(0))
        reg_freqs = reg_freqs / reg_freqs.sum()
        obs_freqs = self._to_tensor(wxa_ind.mean(0))
        target_mean = float((target * eta_t).mean())

        # setup q network
        h_net = self.h_net_class(**self.h_net_args, x_len=wx_len, a_len=a_len,
                                 num_a=self.num_a)

        for rep_i in range(self.num_rep):
            if rep_i > 0:
                h_tilde_vals = torch_to_np(h_net(wxa_vals_torch))
                h_tilde = h_tilde_vals[wxa_codes]
            else:
                h_tilde = np.ones((n, 1)) * target_mean
            h_fact = eta_t * k_2 * (h_tilde - target)
            q = (1/n) * h_fact.T @ h_fact + alpha * k_1
            q_inv = self._to_tensor(np.linalg.inv(q))
            q_inv = (q_inv + q_inv.T) / 2

            optimizer = torch.optim.LBFGS(h_net.parameters())

            def closure():
                optimizer.zero_grad()
                h_net_vals = h_net(wxa_vals_torch).flatten()
                rho = b @ h_net_vals - c
                if isinstance(self.lmbda, tuple):
                    l0, l1, l2 = self.lmbda
                else:
                    l0, l1, l2 = self.lmbda, 0, 0
                reg_0 = l0 * (h_net_vals ** 2) @ obs_freqs
                reg_1 = l1 * (h_net_vals @ reg_freqs - target_mean) ** 2
                h_net_excess = (F.relu(h_net_vals - dfr_max)
                                + F.relu(dfr_min - h_net_vals)) ** 2
                reg_2 = l2 * h_net_excess @ obs_freqs
                loss = rho.T @ q_inv @ rho + reg_0 + reg_1 + reg_2
                loss.backward()
                return loss
            optimizer.step(closure)

        h_net_vals = h_net(wxa_vals_torch).flatten()
        rho = b @ h_net_vals - c
        loss = float(rho.T @ q_inv @ rho)

        h_vals = h_net(wxa_vals_torch).detach().flatten()
        mean_h = float(reg_freqs @ h_vals)
        std_h = float(reg_freqs @ ((h_vals - mean_h) ** 2))
        # print("mean h:", mean_h, "std h:", std_h, "loss:", loss)
        if np.isclose(target_mean, 0.0):
            norm = 1.0
        else:
            norm = mean_h / target_mean
        return h_net, norm

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
