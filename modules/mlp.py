from torch import nn


def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        # nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class MLP(nn.Module):

    def __init__(self, num_genes=5000, num_hidden=128, p_drop=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.3),
            full_block(in_features=num_genes, out_features=1024, p_drop=p_drop),
            full_block(in_features=1024, out_features=512, p_drop=p_drop),
            # full_block(in_features=2048, out_features=1024, p_drop=p_drop),
            full_block(in_features=512, out_features=256, p_drop=p_drop),
            full_block(in_features=256, out_features=128, p_drop=p_drop),
        )
        self.rep_dim = num_hidden

    def forward(self, x):
        x = self.encoder(x)

        return x


class ValidMLP(nn.Module):

    def __init__(self, num_genes, p_drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            full_block(in_features=num_genes, out_features=1024, p_drop=p_drop),
            full_block(in_features=1024, out_features=512, p_drop=p_drop),
            full_block(in_features=512, out_features=256, p_drop=p_drop),
            full_block(in_features=256, out_features=128, p_drop=p_drop),
            full_block(in_features=128, out_features=64, p_drop=p_drop),
            nn.Linear(in_features=64, out_features=14),
            # nn.LayerNorm(out_features),
        )

    def forward(self, x):
        output = self.encoder(x)
        return output
