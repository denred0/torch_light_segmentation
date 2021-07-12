import pytorch_lightning as pl
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.classes import LASER_CLASSES
from src.utils.weights import LASER_WEIGHTS
from src.seg_models.losses import CompoundLoss
from src.cloned_repo.DPT.dpt.models import DPTSegmentationModel
from src.seg_models.poly_lr_decay import PolynomialLRDecay
from src.seg_models.madgrad import MADGRAD
from src.seg_models.losses import CompoundLoss


class MainModelDPT(pl.LightningModule):
    def __init__(self, model_type, pretrained_weights, lr=0.001, encoder_pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        # load network
        if model_type == "dpt_large":
            self.model = DPTSegmentationModel(
                len(LASER_CLASSES),
                path=pretrained_weights,
                backbone="vitl16_384",
                encoder_pretrained=encoder_pretrained,
            )
        elif model_type == "dpt_hybrid":
            self.model = DPTSegmentationModel(
                len(LASER_CLASSES),
                path=pretrained_weights,
                backbone="vitb_rn50_384",
                encoder_pretrained=encoder_pretrained
            )
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

        # weights = torch.Tensor(LASER_WEIGHTS)
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.CrossEntropyLoss(weight=weights)
        self.loss = CompoundLoss(coef1=0.75, coef2=0.25)

        self.f1 = torchmetrics.F1(num_classes=len(LASER_CLASSES))
        self.iou = torchmetrics.IoU(num_classes=len(LASER_CLASSES))

        # self.train_f1 = f1.clone()
        # self.val_f1 = f1.clone()
        # self.test_f1 = f1.clone()

        # self.train_iou = iou.clone()
        # self.val_iou = iou.clone()
        # self.test_iou = iou.clone()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            # self.train_f1.reset()
            # self.train_iou.reset()
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)

        output = output.argmax(dim=1)
        # f1 = self.train_f1(probs, mask.long())
        # iou = self.train_iou(probs, mask.long())
        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        self.log('train_f1_step', f1, prog_bar=True)
        self.log('train_iou_step', iou, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # self.val_f1.reset()
            # self.val_iou.reset()
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)

        output = output.argmax(dim=1)
        # f1 = self.val_f1(probs, mask.long())
        # iou = self.val_iou(probs, mask.long())
        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        self.log('val_f1_step', f1)
        self.log('val_iou_step', iou)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            # self.test_f1.reset()
            # self.test_iou.reset()
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)

        output = output.argmax(dim=1)
        # f1 = self.test_f1(probs, mask.long())
        # iou = self.test_iou(probs, mask.long())
        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        self.log('test_f1_step', f1, prog_bar=True)
        self.log('test_iou_step', iou, prog_bar=True)
        self.log('test_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        # self.log('train_f1_epoch', self.train_f1.compute())
        # self.log('train_iou_epoch', self.train_iou.compute())
        # self.train_f1.reset()
        # self.train_iou.reset()
        self.log('train_f1_epoch', self.f1.compute())
        self.log('train_iou_epoch', self.iou.compute())
        self.f1.reset()
        self.iou.reset()

    def validation_epoch_end(self, outputs):
        # self.log('val_f1_epoch', self.val_f1.compute(), prog_bar=True)
        # self.log('val_iou_epoch', self.val_iou.compute(), prog_bar=True)
        # self.val_f1.reset()
        # self.val_iou.reset()
        self.log('val_f1_epoch', self.f1.compute(), prog_bar=True)
        self.log('val_iou_epoch', self.iou.compute(), prog_bar=True)
        self.f1.reset()
        self.iou.reset()

    def test_epoch_end(self, outputs):
        # self.log('test_f1_epoch', self.test_f1.compute(), prog_bar=True)
        # self.log('test_iou_epoch', self.test_iou.compute(), prog_bar=True)
        # self.test_f1.reset()
        # self.test_iou.reset()
        self.log('test_f1_epoch', self.f1.compute())
        self.log('test_iou_epoch', self.iou.compute())
        self.f1.reset()
        self.iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-5)
        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=50,
                                         end_learning_rate=1e-6,
                                         power=0.9, verbose=True)

        # optimizer = MADGRAD(self.parameters(), lr=self.lr, weight_decay=2.5e-5)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True)


        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     factor=0.5,
        #     threshold=0.01,
        #     threshold_mode='rel',
        #     cooldown=3,
        #     mode='max',
        #     min_lr=1e-6,
        #     verbose=True,
        #     )

        lr_dict = {
                    'scheduler': lr_scheduler,
                    'reduce_on_plateau': False,
                    'monitor': 'val_f1_epoch',
                    'interval': 'epoch',
                    'frequency': 1,
        }

        return [optimizer], [lr_dict]
