import torch.nn as nn

class OccLoss(nn.Module):
    """
    BCE loss for binary occupancy forecasting
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        return self.bce(pred, gt.float())
