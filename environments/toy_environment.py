import numpy as np
from scipy.spatial.distance import cdist

from environments.abstract_environment import AbstractEvaluationPolicy, \
    AbstractEnvironment
from utils.np_utils import one_hot_embed


class ToyEnvironment(AbstractEnvironment):
    def __init__(self, pci_reducer, eps_noise):
        AbstractEnvironment.__init__(self, pci_reducer)
        self.s = None

        self.num_s = 3
        self.s_values = list(range(self.num_s))
        self.num_a = 2
        self.a_values = list(range(self.num_a))
        self.init_s_probs = [0.5, 0.3, 0.2]
        # self.init_s_probs = [1.0, 0.0, 0.0]
        self.o_probs = {
            0: [1 - eps_noise, eps_noise / 2, eps_noise / 2],
            1: [eps_noise / 2, 1 - eps_noise, eps_noise / 2],
            2: [eps_noise / 2, eps_noise / 2, 1 - eps_noise],
        }
        # self.a_probs = {
        #     0: [0.75, 0.25],
        #     1: [0.3, 0.7],
        #     2: [0.2, 0.8],
        # }
        self.a_probs = {
            0: [0.8, 0.2],
            1: [0.8, 0.2],
            2: [0.2, 0.8],
        }
        self.mean_rewards = {
            (0, 0): 3.0,
            (0, 1): 0.0,
            (1, 0): 1.0,
            (1, 1): -2.0,
            (2, 0): 8.0,
            (2, 1): -2.0,
        }
        self.std_rewards = {
            (0, 0): 0.0,
            (0, 1): 0.0,
            (1, 0): 0.0,
            (1, 1): 0.0,
            (2, 0): 0.0,
            (2, 1): 0.0,
        }
        # self.std_rewards = {
        #     (0, 0): 1.0,
        #     (0, 1): 1.0,
        #     (1, 0): 1.0,
        #     (1, 1): 1.0,
        #     (2, 0): 10.0,
        #     (2, 1): 1.0,
        # }
        self.successor_states = {
            (0, 0): 1,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 2,
            (2, 0): 0,
            (2, 1): 1,
        }

    def reset(self):
        self.s = np.random.choice(self.s_values, p=self.init_s_probs)
        o_prior = self.sample_observation()
        a = self.sample_action()
        self.transition_state(a)
        return o_prior

    def sample_observation(self):
        return np.random.choice(self.s_values, p=self.o_probs[self.s])

    def sample_action(self):
        return np.random.choice(self.a_values, p=self.a_probs[self.s])

    def transition_state(self, a):
        r_mean = self.mean_rewards[(self.s, a)]
        r_std = self.std_rewards[(self.s, a)]
        r = r_mean + float(np.random.randn()) * r_std
        self.s = self.successor_states[(self.s, a)]
        return r

    def embed_a(self, a):
        return one_hot_embed(a, self.num_a)

    def embed_o(self, o):
        return one_hot_embed(o, self.num_s)

    def get_num_a(self):
        return self.num_a

    def zxa_sq_dist(self, zxa_1, zxa_2):
        zxa_1_embed = self.pci_reducer.embed_zxa(zxa_1, self, use_x=False)
        zxa_2_embed = self.pci_reducer.embed_zxa(zxa_2, self, use_x=False)
        return cdist(zxa_1_embed, zxa_2_embed, metric="sqeuclidean")

    def wxa_sq_dist(self, wxa_1, wxa_2):
        wxa_1_embed = self.pci_reducer.embed_wxa(wxa_1, self, use_x=False)
        wxa_2_embed = self.pci_reducer.embed_wxa(wxa_2, self, use_x=False)
        return cdist(wxa_1_embed, wxa_2_embed, metric="sqeuclidean")


class GeneralToyEvaluationPolicy(AbstractEvaluationPolicy):
    def __init__(self, o_to_a_map):
        AbstractEvaluationPolicy.__init__(self, num_a=2)
        self.o_to_a_map = o_to_a_map

    def get_e_t(self, t, o_t, prev_o_t, prev_a_t, prev_r_t):
        return np.array([self.o_to_a_map[o_] for o_ in o_t])


class ToyEvaluationPolicyEasy(GeneralToyEvaluationPolicy):
    def __init__(self):
        o_to_a_map_easy = {0: 0,
                           1: 0,
                           2: 1}
        GeneralToyEvaluationPolicy.__init__(self, o_to_a_map=o_to_a_map_easy)


class ToyEvaluationPolicyHard(GeneralToyEvaluationPolicy):
    def __init__(self):
        o_to_a_map_hard = {0: 1,
                           1: 1,
                           2: 0}
        GeneralToyEvaluationPolicy.__init__(self, o_to_a_map=o_to_a_map_hard)


class ToyEvaluationPolicyOptim(GeneralToyEvaluationPolicy):
    def __init__(self):
        o_to_a_map_optim = {0: 0,
                            1: 1,
                            2: 0}
        GeneralToyEvaluationPolicy.__init__(self, o_to_a_map=o_to_a_map_optim)
