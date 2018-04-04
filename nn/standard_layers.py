import numpy as np


class LinearLayer():
    def __init__(self, input_size, output_size, initialisation):
        if initialisation == 'random':
            self.w = np.random.normal(0.0, 1.0, (output_size, input_size))
            self.b = np.random.normal(0.0, 1.0, (output_size, 1))
        elif initialisation == 'ones':
            self.w = np.ones((output_size, input_size))
            self.b = np.ones((output_size, 1))
        else:
            raise Exception('initialisation unknown')
        self.x = 0
        self.J_w = 0
        self.J_b = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x.copy()
        out = self.w.dot(self.x) + np.tile(self.b, (1, self.x.shape[-1]))
        return out

    def backward(self, J_y):
        self.J_w = J_y.dot(self.x.T)
        self.J_b = np.ones(self.b.shape) * self.b.shape[-1]
        J_x = self.w.T.dot(J_y)
        return J_x

    def step_update(self, learning_rate):
        self.w -= learning_rate * self.J_w
        self.b -= learning_rate * self.J_b


def test_linear():
    N = 5
    # x = np.linspace(-50, 50, N).reshape(1, N)
    x = np.ones((1)) * 50
    linear = LinearLayer(1, 1, initialisation='ones')
    y = linear(x)
    J_y = np.ones((1)) * 2
    J_x = linear.backward(J_y)
    print x, 'x'
    print y, 'y'
    print J_y, 'J_y'
    print J_x, 'J_x'


if __name__ == '__main__':
    test_linear()
