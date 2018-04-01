import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self):
        self.x = 0
        self.leak = 0.1

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


class LinearLayer():
    def __init__(self, input_size, output_size, initialisation):
        if initialisation == 'random':
            self.w = np.random.randn(output_size, input_size)
            self.b = np.random.randn(output_size, 1)
            # self.w = np.random.normal(0.0, 1.0, (output_size, input_size))
            # self.b = np.random.normal(0.0, 1.0, (output_size, 1))
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


class Perceptron():
    def __init__(self, input_size, output_size, final_layer=False, initialisation='random'):
        self.linear_layer = LinearLayer(
            input_size, output_size, initialisation)
        self.final_layer = final_layer
        if self.final_layer == False:
            self.leaky_relu = LeakyRelu()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.linear_layer(x)
        if self.final_layer == False:
            out = self.leaky_relu(out)
        return out

    def backward(self, y):
        if self.final_layer == False:
            dy_dx = self.leaky_relu.backward(y)
            dy_dx = self.linear_layer.backward(dy_dx)
        else:
            dy_dx = self.linear_layer.backward(y)
        return dy_dx

    def step_update(self, learning_rate):
        self.linear_layer.step_update(learning_rate)


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


class MultiLayerPerceptron():
    def __init__(self, input_size, hidden_size, output_size):
        self.perceptron1 = Perceptron(input_size, hidden_size)
        self.perceptron2 = Perceptron(
            hidden_size, output_size, final_layer=True)
        # self.perceptron1 = Perceptron(
        #    input_size, output_size, final_layer=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.perceptron1(x)
        # print 'out1', out
        out = self.perceptron2(out)
        # print 'out2', out
        return out

    def backward(self, loss):
        x = self.perceptron2.backward(loss)
        x = self.perceptron1.backward(x)
        return x

    def step_update(self, learning_rate):
        self.perceptron2.step_update(learning_rate)
        self.perceptron1.step_update(learning_rate)


class Optimiser():
    def __init__(self, policy={'type': 'step', 'base_lr': 0.01, 'step_size': 50, 'lr_multipier': 0.5}):
        self.learning_rate = policy['base_lr']
        self.policy = policy
        if self.policy['type'] == 'step':
            self.optimiser = StepOptimiser(
                self.learning_rate, policy['step_size'], policy['lr_multiplier'])
        elif self.policy['type'] == 'poly':
            self.optimiser = PolynomialOptimiser(
                self.learning_rate, policy['max_iter'])
        else:
            raise Exception('learning policy undefined')

    def get_learning_rate(self, iteration):
        lr = self.optimiser.get_learning_rate(iteration)
        return lr


class StepOptimiser():
    def __init__(self, base_lr, step_size, lr_multiplier):
        self.base_lr = base_lr
        self.step_size = step_size
        self.lr_multiplier = lr_multiplier

    def get_learning_rate(self, iteration):
        lr = self.base_lr * (self.lr_multiplier**(iteration//self.step_size))
        return lr


class PolynomialOptimiser():
    def __init__(self, base_lr, max_iter):
        self.power = 4
        self.base_lr = base_lr
        self.max_iter = max_iter

    def get_learning_rate(self, iteration):
        lr = self.base_lr * \
            (1.0 - float(iteration)/float(self.max_iter))**(self.power)
        return lr


def test():
    # # N batch size
    # # D_in is input dimension
    # # H is the hidden dimension
    # # D_out is the output dimension
    np.random.seed(3)
    N, D_in, H, D_out = 64, 1, 100, 1
    model = MultiLayerPerceptron(D_in, H, D_out)

    # Create random input and output data
    x = np.random.randn(D_in, N)
    # y = np.random.randn(D_out, N)

    # x = np.linspace(0, 100, N).reshape(1, N)
    # x = np.ones((1, 1))
    y = x*3 + np.random.randn(D_out, N) * 0.01 + 0.5

    loss_function = MSELoss()
    step_policy = {'type': 'step',
                   'base_lr': 0.01,
                   'max_iter': 200,
                   'step_size': 100,
                   'lr_multiplier': 0.5}

    poly_policy = {'type': 'poly',
                   'base_lr': 0.0001,
                   'max_iter': 600}

    policy = poly_policy
    optimiser = Optimiser(policy=policy)

    axes = plt.gca()
    lines = axes.plot(x, y, 'ro', x, y, 'go')
    # line_model = axes.plot(x, y, 'go')
    plt.draw()
    plt.pause(1e-3)
    # ax.plot(x, y, 'ro')

    for t in range(policy['max_iter']):
        out = model.forward(x)
        # print 'y', out
        # raw_input('enter to continue')
        loss = loss_function(out, y)
        model.backward(loss_function.backward())
        lr = optimiser.get_learning_rate(t)
        model.step_update(lr)
        print t, lr, loss

        lines[1].set_xdata(x)
        lines[1].set_ydata(out)
        plt.draw()
        plt.pause(1e-3)
        # ax.plot(x, out, 'bo')
        # plt.show()

        # plt.pause(0.00001)

    x = np.random.randn(D_in, N)
    y_model = model.forward(x)
    line_final = axes.plot(x, y_model, 'bo')
    plt.draw()
    print y_model
    # ax.plot(x, y_model, 'go')

    plt.show()


def test_relu():
    N = 5
    x = np.linspace(-50, 50, N).reshape(1, N)
    relu = Relu()
    y = relu(x)
    J_x = relu.backward(y)
    print x, 'x'
    print y, 'y'
    print J_x, 'J_x'


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
    test()
