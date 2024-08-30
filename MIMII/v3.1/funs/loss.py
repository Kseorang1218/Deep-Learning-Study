# loss.py
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss