import pytorch_lightning as pl
import torch
import torch.optim
import torch.nn as nn
import torchmetrics
import numpy as np

from project.src.classes import LASER_CLASSES
from project.seg_models.poly_lr_decay import PolynomialLRDecay

from project.cloned_repo.DPT.dpt.models import DPTSegmentationModel

from pytorch_lightning.metrics import F1, IoU

import segmentation_models_pytorch as smp
from project.seg_models.losses import CompoundLoss
from pytorch_toolbelt import losses as L
from project.src.losses import Combo_Loss


class ISPModel(pl.LightningModule):
    def __init__(self, architecture, encoder, pretrained_weights, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        # load network
        self.pretrained_weights = pretrained_weights
        self.architecture = architecture
        self.encoder = encoder

        if self.architecture == 'FPN':
            self.model = smp.FPN(
                encoder_name=self.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.pretrained_weights,
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=len(LASER_CLASSES)  # model output channels (number of classes in your dataset)
            )
        elif self.architecture == 'PSPNet':
            self.model = smp.PSPNet(
                encoder_name=self.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.pretrained_weights,
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=len(LASER_CLASSES)  # model output channels (number of classes in your dataset)
            )
        elif self.architecture == 'DPTSegmentationModel':
            self.model = DPTSegmentationModel(
                len(LASER_CLASSES),
                path=self.pretrained_weights,
                backbone=self.encoder,
                encoder_pretrained=True
            )
        else:
            assert (
                False
            ), f"model_type '{self.architecture} {self.encoder}' not implemented"

        # FPN_res34 = smp.FPN(encoder_name='resnet34', classes=len(LASER_CLASSES), encoder_weights=encoder_weights)
        # PSP_seresx50 = smp.PSPNet(encoder_name='se_resnext50_32x4d', classes=len(LASER_CLASSES),
        #                           encoder_weights=encoder_weights)
        # FPN_seresx50 = smp.FPN(encoder_name='se_resnext50_32x4d', classes=len(LASER_CLASSES),
        #                        encoder_weights=encoder_weights)
        # FPN_b0 = smp.FPN(encoder_name='efficientnet-b0', classes=len(LASER_CLASSES),
        #                  encoder_weights=encoder_weights)
        # FPN_b1 = smp.FPN(encoder_name='efficientnet-b1', classes=len(LASER_CLASSES),
        #                  encoder_weights=encoder_weights)
        # FPN_b2 = smp.FPN(encoder_name='efficientnet-b2', classes=len(LASER_CLASSES),
        #                  encoder_weights=encoder_weights)
        # FPN_b3 = smp.FPN(encoder_name='efficientnet-b3', classes=len(LASER_CLASSES),
        #                  encoder_weights=encoder_weights)
        # FPN_b4 = smp.FPN(encoder_name='efficientnet-b4', classes=len(LASER_CLASSES),
        #                  encoder_weights=encoder_weights)
        #
        # self.model = Model([FPN_b3, FPN_seresx50, FPN_b0, FPN_b1, FPN_b2, FPN_b4])

        self.loss = nn.CrossEntropyLoss()

       # self.loss = CompoundLoss(coef1=0.75, coef2=0.25)
        # self.loss = Combo_Loss()

        # self.f1 = F1(num_classes=len(LASER_CLASSES))  # average f1_score of all classes
        # self.iou = IoU(num_classes=len(LASER_CLASSES), ignore_index=0)

        self.f1 = torchmetrics.F1(num_classes=len(LASER_CLASSES))
        self.iou = torchmetrics.IoU(num_classes=len(LASER_CLASSES))

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)
        # loss = Combo_Loss(mask.cpu(), output.cpu())

        output = output.argmax(dim=1)

        # print()
        # iou_pytorch_lightning_metrics = IoU(num_classes=5)
        # iou_pytorch_lightning_metrics_ = iou_pytorch_lightning_metrics(output.cpu(), mask.cpu())
        # print('iou_pytorch_lightning_metrics_', iou_pytorch_lightning_metrics_.cpu())
        #
        #
        # print('output train', np.unique(output.cpu()))
        # print('mask train', np.unique(mask.cpu()))



        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        self.log('train_f1_step', f1, prog_bar=True)
        self.log('train_iou_step', iou, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)
        # loss = Combo_Loss(mask.cpu(), output.cpu())

        output = output.argmax(dim=1)

        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        # self.log('val_f1_step', f1)
        # self.log('val_iou_step', iou)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.f1.reset()
            self.iou.reset()

        x, mask, _, _ = batch
        output = self.forward(x)
        loss = self.loss(output, mask)
        # loss = Combo_Loss(mask.cpu(), output.cpu())

        output = output.argmax(dim=1)

        f1 = self.f1(output, mask)
        iou = self.iou(output, mask)

        # self.log('test_f1_step', f1, prog_bar=True)
        # self.log('test_iou_step', iou, prog_bar=True)
        self.log('test_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_f1_epoch', self.f1.compute())
        self.log('train_iou_epoch', self.iou.compute())
        self.f1.reset()
        self.iou.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_f1_epoch', self.f1.compute(), prog_bar=True)
        self.log('val_iou_epoch', self.iou.compute(), prog_bar=True)
        self.f1.reset()
        self.iou.reset()

    def test_epoch_end(self, outputs):
        self.log('test_f1_epoch', self.f1.compute())
        self.log('test_iou_epoch', self.iou.compute())
        self.f1.reset()
        self.iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=2e-5)
        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=50,
                                         end_learning_rate=1e-6,
                                         power=0.9, verbose=True)

        lr_dict = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': False,
            'monitor': 'val_f1_epoch',
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [lr_dict]

        # gen_opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #
        # gen_sched = {'scheduler':
        #                  torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=0.999, verbose=False),
        #              'interval': 'step'}  # called after each training step
        #
        # # dis_sched = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt,
        # #                                                        T_max=10)  # called every epoch,
        # # lower the learning rate to its minimum in each epoch and then restart from the base learning rate
        # return [gen_opt], [gen_sched]
