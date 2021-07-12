from torch import nn
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L


class CompoundLoss(smp.utils.losses.base.Loss):
    def __init__(self, coef1=1.0, coef2=1.0, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
        self.lovasz = L.LovaszLoss()
        self.coef1 = coef1
        self.coef2 = coef2

    def forward(self, y_pr, y_gt):
        return self.coef1 * self.cross_entropy.forward(y_pr, y_gt) + \
            self.coef2 * self.lovasz.forward(y_pr, y_gt)
