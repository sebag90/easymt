import torch.nn.functional as F

"""
alternative:
    criterions = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(inp, target)
"""


class MaskedLoss:
    def __call__(self, inp, target, mask):
        loss = F.cross_entropy(inp[mask], target[mask])
        return loss
