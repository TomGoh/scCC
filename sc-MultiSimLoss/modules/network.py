import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, mlp, feature_dim):
        super(Network, self).__init__()
        self.mlp = mlp
        self.feature_dim = feature_dim
        # self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.mlp.rep_dim, self.mlp.rep_dim),
            nn.ReLU(),
            nn.Linear(self.mlp.rep_dim, self.feature_dim),
        )

    def forward(self, x):
        h_i = self.mlp(x)
        # h_j = self.mlp(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        # z_j = normalize(self.instance_projector(h_j), dim=1)
        # z_i = self.instance_projector(h_i)
        # z_j = self.instance_projector(h_j)
        return z_i

    def forward_embedding(self,x):
        return self.mlp(x)
