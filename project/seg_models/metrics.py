import torch
import segmentation_models_pytorch as smp


class Fscore_batch(smp.utils.metrics.Fscore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        batch_scores = []

        for b in range(y_pr.shape[0]):
            batch_scores.append(smp.utils.functional.f_score(
                torch.unsqueeze(y_pr[b], 0), torch.unsqueeze(y_gt[b], 0),
                eps=self.eps,
                beta=self.beta,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
                )
            )
        return sum(batch_scores) / (len(batch_scores) + 1e-7)


class IoU_batch(smp.utils.metrics.IoU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        batch_scores = []

        for b in range(y_pr.shape[0]):
            batch_scores.append(smp.utils.functional.iou(
                torch.unsqueeze(y_pr[b], 0), torch.unsqueeze(y_gt[b], 0),
                eps=self.eps,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
                )
            )

        return sum(batch_scores) / (len(batch_scores) + 1e-7)
