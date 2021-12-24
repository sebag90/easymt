import torch
import torch.nn as nn


class PMaskedLoss:
    """
    requires pytorch 1.10
    """
    def __init__(self, padding_idx, smoothing):
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            label_smoothing=smoothing,
            reduction="mean"
        )

    def __call__(self, inp, target):
        loss = self.criterion(inp, target)
        return loss


class MaskedLoss(nn.Module):
    """
    Cross Entropy with label smoothing
    """
    def __init__(self, padding_idx, smoothing):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.padding_idx = padding_idx

    def forward(self, x, target):
        # apply mask
        mask = (target != self.padding_idx)
        x = x[mask]
        target = target[mask]

        # calculate log probability
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # apply smoothing
        smooth_loss = -logprobs.mean(dim=-1)
        conf_loss = self.confidence * nll_loss
        smoothing = self.smoothing * smooth_loss
        loss = conf_loss + smoothing

        return loss.mean()
