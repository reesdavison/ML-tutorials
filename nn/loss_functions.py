import numpy as np


class MSELoss:
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


class Softmax:
    def __init__(self):
        self.x = 0
        self.stability_factor = 0.2

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        input x will be an Nx1 row array
        softmax function has the form e^fi/SUM_over_f e^fi
        with loss, has the form:
            Loss = -log(softmax)

            for stability:
                e^fi * e^N/SUM(e^fi) * e^N
                then same as
                e^(fi + s)/SUM(e^(fi+s))
        """
        self.x = x.copy()
        numerator = np.exp(x + self.stability_factor)
        denominator = np.sum(np.exp(x + self.stability_factor))
        y = numerator / denominator
        return y


def test_softmax():
    N = 5
    x = np.ones((N, 1))
    softmax = Softmax()
    y = softmax(x)
    print("x ", x)
    print("y ", y)
    print(np.sum(y))
    x = np.array([[0.2, 1.0, 0.2, 0.2, 0.2, 0.2]])
    print(x)
    y = softmax(x)
    print(y)
    print(np.sum(y))


if __name__ == "__main__":
    test_softmax()
