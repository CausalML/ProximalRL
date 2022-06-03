import torch
import torch.nn as nn



class AbstractNet(nn.Module):
    def __init__(self, x_len, a_len, num_a):
        nn.Module.__init__(self)
        self.x_len = x_len
        self.a_len = a_len
        self.num_a = num_a

    def is_is_finite(self):
        raise NotImplementedError()

    def forward(self, data):
        raise NotImplementedError()


class TabularNet(AbstractNet):
    def __init__(self, x_len, a_len, num_a, bias=False):
        AbstractNet.__init__(self, x_len, a_len, num_a)
        self.x_len, self.a_len = x_len, a_len
        if x_len == 1:
            in1_dim = 2
            self.binary_x = True
        else:
            in1_dim = x_len
            self.binary_x = False

        if a_len == 1:
            in2_dim = 2
            self.binary_a = True
            assert num_a == 2
        else:
            in2_dim = a_len
            self.binary_a = False
            assert num_a == a_len

        self.net = nn.Bilinear(in1_features=in1_dim, in2_features=in2_dim,
                               out_features=1, bias=bias)

    def is_finite(self):
        for p in self.parameters():
            if not p.data.isfinite().all():
                return False
        return True

    def forward(self, data):
        if self.binary_x:
            x_0 = data[:, 0]
            x = torch.stack([x_0, 1 - x_0], dim=1)
        else:
            x = data[:, :self.x_len]
        if self.binary_a:
            a_0 = data[:, self.x_len]
            a = torch.stack([a_0, 1 - a_0], dim=1)
        else:
            a = data[:, self.x_len:]
        return self.net(x, a)


class FlexibleAbsorbingNet(AbstractNet):
    def __init__(self, x_len, a_len, num_a, absorbing_val):
        AbstractNet.__init__(self, x_len, a_len, num_a)
        self.x_len = x_len
        self.net = nn.Sequential(
            nn.Linear(x_len + a_len - 1, 50),
            nn.GELU(),
            nn.Linear(50, 50),
            nn.GELU(),
            nn.Linear(50, 1),
        )
        self.absorbing_val = absorbing_val

    def is_finite(self):
        for p in self.parameters():
            if not p.data.isfinite().all():
                return False
        return True

    def forward(self, data):
        absorbing_flag = data[:, self.x_len-1].reshape(-1, 1)
        x = data[:, :self.x_len-1]
        a = data[:, self.x_len:]
        xa_batch = torch.cat([x, a], dim=1)
        out = self.net(xa_batch)
        true_out = (out * (1.0 - absorbing_flag)
                    + self.absorbing_val * absorbing_flag)
        # print(absorbing_flag)
        # print(true_out)
        return true_out


class FlexibleCriticNet(AbstractNet):
    def __init__(self, x_len, a_len, num_a):
        AbstractNet.__init__(self, x_len, a_len, num_a)
        self.x_len = x_len
        self.net = nn.Sequential(
            nn.Linear(x_len + a_len, 50),
            nn.GELU(),
            nn.Linear(50, 50),
            nn.GELU(),
            nn.Linear(50, 1),
        )

    def is_finite(self):
        for p in self.parameters():
            if not p.data.isfinite().all():
                return False
        return True

    def forward(self, data):
        return self.net(data)
