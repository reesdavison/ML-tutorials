import numpy as np


class Relu():
    def __init__(self):
        self.x = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x.copy()
        return np.maximum(x, 0)

    def backward(self, J_y):
        J_x = self.x.copy()
        J_x[self.x <= 0] = 0
        J_x[self.x > 0] = 1
        return J_x


class LeakyRelu():
    def __init__(self, leak=0.1):
        self.x = 0
        self.leak = leak

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x.copy()
        out = x.copy()
        out[x <= 0] = out[x <= 0] * self.leak
        return out

    def backward(self, J_y):
        J_x = self.x.copy()
        J_x[self.x <= 0] = self.leak
        J_x[self.x > 0] = 1
        return J_x


def test_relu():
    N = 5
    x = np.linspace(-50, 50, N).reshape(1, N)
    relu = Relu()
    y = relu(x)
    J_x = relu.backward(y)
    print x, 'x'
    print y, 'y'
    print J_x, 'J_x'


if __name__ == '__main__':
    test_relu()
