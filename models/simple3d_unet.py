import torch
import torch.nn as nn

class SimpleOccNet(nn.Module):
    """
    Baseline 3D occupancy forecasting model.
    Input:  [B, obs_len, L, W, H]
    Output: [B, fut_len, L, W, H]
    """
    def __init__(self, obs_len=8, fut_len=12, hidden_dim=32):
        super().__init__()
        self.obs_len = obs_len
        self.fut_len = fut_len

        self.encoder = nn.Sequential(
            nn.Conv3d(obs_len, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, fut_len, 1)
        )

    def forward(self, occ_obs):
        """
        occ_obs : [B, obs_len, L, W, H]
        """
        x = self.encoder(occ_obs)
        pred = self.decoder(x)
        return pred
