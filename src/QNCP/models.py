import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import orthogonal
class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(data=torch.empty(1))
        nn.init.normal_(self.scale, mean=1, std=1e-2)

    def forward(self, X):
        return X * self.scale

class Sin(nn.Module):
    def __init__(self, size):
        super(Sin, self).__init__()
        self.amp = nn.Parameter(data=torch.empty(size))
        nn.init.normal_(self.amp, mean=1, std=1e-2)
        self.freq = nn.Parameter(data=torch.empty(size))
        nn.init.normal_(self.freq, mean=1, std=1e-2)
        self.phase = nn.Parameter(data=torch.empty(size))
        nn.init.normal_(self.phase, mean=0, std=1e-2)

    def forward(self, x):
        return self.amp*torch.sin(self.freq*x + self.phase)

class Sphere(nn.Module):
    def __init__(self, radius=1):
        super(Sphere, self).__init__()
        self.radius = radius

    def forward(self, X):
        assert X.shape[-1] == 2
        x = self.radius * torch.cos(X[..., 0]) * torch.sin(X[..., 1])
        y = self.radius * torch.sin(X[..., 0]) * torch.sin(X[..., 1])
        z = self.radius * torch.cos(X[..., 0])

        return torch.stack([x, y, z], axis=-1)


class Crystal_Model(nn.Module):
    def __init__(self, size):
        super(Crystal_Model, self).__init__()
        self.s_in_p = nn.Parameter(data=torch.empty(2))
        nn.init.normal_(self.s_in_p, mean=0.5, std=1e-2)
        self.map = Sphere()

        self.m_1 = nn.Sequential(
            nn.Linear(5, size),
            Sin(size),
            nn.Linear(size, size),
            Sin(size),
            nn.Linear(size, size),
            Sin(size),
        )

        self.m_2 = nn.Sequential(
            nn.Linear(5, size),
            nn.GELU(),
            nn.Linear(size, size),
            nn.GELU(),
            nn.Linear(size, size),
            nn.GELU(),
        )

        self.m_3 = nn.Sequential(
            nn.Linear(5, size),
            nn.Tanh(),
            nn.Linear(size, size),
            nn.Tanh(),
            nn.Linear(size, size),
            nn.Tanh(),
            Scale()
        )

        self.m_4 = nn.Sequential(
                nn.Linear(5, size),
                nn.GELU(),
                nn.Linear(size, size),
                nn.GELU(),
                nn.Linear(size, size),
                nn.GELU(),
                nn.Linear(size, size),
                nn.GELU(),
                nn.Linear(size, size),
                nn.GELU()
        )

        self.final = nn.Sequential(
                nn.Linear(4*size, 4*size),
                nn.GELU(),
                nn.Linear(4*size, 3),
                nn.Tanh(),
                Scale()
        )

    def forward(self, u):
        s_in = self.map(self.s_in_p)
        s_in = s_in.unsqueeze(0).repeat(u.shape[0], 1)
        u = torch.cat([u, s_in], axis=-1)
        res = torch.cat([
            self.m_1(u),
            self.m_2(u),
            self.m_3(u),
            self.m_4(u)
        ], dim=-1)

        return self.final(res)

