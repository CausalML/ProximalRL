import numpy as np


class AbstractPCIReducer(object):
    def __init__(self):
        pass

    def compute_z(self, o, a, r):
        raise NotImplementedError()

    def compute_w(self, o, a, r):
        raise NotImplementedError()

    def compute_x(self, o, a, r):
        raise NotImplementedError()

    def embed_z(self, z, env):
        raise NotImplementedError()

    def embed_w(self, w, env):
        raise NotImplementedError()

    def embed_x(self, x, env):
        raise NotImplementedError()

    def get_z_len(self, env):
        raise NotImplementedError()

    def get_w_len(self, env):
        raise NotImplementedError()

    def get_x_len(self, env):
        raise NotImplementedError()

    def using_x(self):
        raise NotImplementedError()

    def embed_zxa(self, zxa_list, env, use_x):
        if use_x:
            z, x, a = zip(*zxa_list)
            z_embed = self.embed_z(np.array(z), env)
            x_embed = self.embed_x(np.array(x), env)
            a_embed = env.embed_a(np.array(a))
            return np.concatenate([z_embed, x_embed, a_embed], axis=1)
        else:
            z, a = zip(*zxa_list)
            z_embed = self.embed_z(np.array(z), env)
            a_embed = env.embed_a(np.array(a))
            return np.concatenate([z_embed, a_embed], axis=1)

    def embed_wxa(self, wxa_list, env, use_x):
        if use_x:
            w, x, a = zip(*wxa_list)
            w_embed = self.embed_w(np.array(w), env)
            x_embed = self.embed_x(np.array(x), env)
            a_embed = env.embed_a(np.array(a))
            return np.concatenate([w_embed, x_embed, a_embed], axis=1)
        else:
            w, a = zip(*wxa_list)
            w_embed = self.embed_w(np.array(w), env)
            a_embed = env.embed_a(np.array(a))
            return np.concatenate([w_embed, a_embed], axis=1)


class CurrPrevObsPCIReducer(AbstractPCIReducer):
    def __init__(self):
        AbstractPCIReducer.__init__(self)

    def compute_z(self, o, a, r):
        return o[:-1]

    def compute_w(self, o, a, r):
        return o[1:]

    def compute_x(self, o, a, r):
        return None

    def embed_z(self, z, env):
        return env.embed_o(z)

    def embed_w(self, w, env):
        return env.embed_o(w)

    def embed_x(self, x, env):
        return None

    def get_z_len(self, env):
        return env.get_o_len()

    def get_w_len(self, env):
        return env.get_o_len()

    def get_x_len(self, env):
        return None

    def using_x(self):
        return False
