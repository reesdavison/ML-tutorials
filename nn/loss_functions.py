import numpy as np


class MSELoss():
    def __init__(self):
        self.x = 0
        self.gt = 0

    def __call__(self, x, gt):
        return self.forward(x, gt)

    def forward(self, x, gt):
        self.x = x.copy()
        self.gt = gt.copy()
        loss = 0.5 * np.sum(np.square(self.x - self.gt))
        return loss

    def backward(self):
        out = self.x - self.gt
        return out
