import torch


class MaskedLoss:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self, inp, target):
        loss = self.criterion(inp, target)
        return loss
