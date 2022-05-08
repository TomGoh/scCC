import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, mlp, feature_dim):
        super(Network, self).__init__()
        self.mlp = mlp
        # self.cluster_num=class_num
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.mlp.rep_dim, self.mlp.rep_dim),
            nn.ReLU(),
            nn.Linear(self.mlp.rep_dim, self.feature_dim),
        )
        # self.cluster_projector = nn.Sequential(
        #     nn.Linear(self.mlp.rep_dim, self.mlp.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.mlp.rep_dim, self.cluster_num),
        #     # nn.Softmax(dim=1)
        #     nn.Softmax()
        # )

    def forward(self, x):
        h_i = self.mlp(x)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        # c_i = self.cluster_projector(h_i)

        return z_i

    def forward_embedding(self,x):
        return self.mlp(x)

    # def forward_cluster(self, x):
    #     h = self.mlp(x)
    #     c = self.cluster_projector(h)
    #     c = torch.argmax(c, dim=1)
    #     return c
