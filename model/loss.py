import torch.nn.functional as F


class MaskedLoss:
    def __call__(self, inp, target, mask):
        total = mask.sum()
        loss = F.cross_entropy(inp[mask], target[mask])
        return loss, total.item()
