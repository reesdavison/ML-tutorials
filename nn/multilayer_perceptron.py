import numpy as np

import standard_layers as std
import activation_functions as act


class Perceptron():
    def __init__(self, input_size, output_size, final_layer=False, initialisation='random'):
        self.linear_layer = std.LinearLayer(
            input_size, output_size, initialisation)
        self.final_layer = final_layer
        if self.final_layer == False:
            self.leaky_relu = act.LeakyRelu()

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


class TwoLayerPerceptron():
    def __init__(self, input_size, hidden_size, output_size):
        self.perceptron1 = Perceptron(input_size, hidden_size)
        self.perceptron2 = Perceptron(
            hidden_size, output_size, final_layer=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.perceptron1(x)
        out = self.perceptron2(out)
        return out

    def backward(self, loss):
        x = self.perceptron2.backward(loss)
        x = self.perceptron1.backward(x)
        return x

    def step_update(self, learning_rate):
        self.perceptron2.step_update(learning_rate)
        self.perceptron1.step_update(learning_rate)


class MultiLayerPerceptron():
    def __init__(self, sizes=[1, 100, 1]):
        self.sizes = sizes
        self.perceptrons = []
        for i in range(len(self.sizes) - 2):
            self.perceptrons.append(Perceptron(sizes[i], sizes[i+1]))
        self.perceptrons.append(Perceptron(
            sizes[-2], sizes[-1], final_layer=True))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for perceptron in self.perceptrons:
            x = perceptron(x)
        return x

    def backward(self, y):
        for perceptron in reversed(self.perceptrons):
            y = perceptron.backward(y)
        return y

    def step_update(self, learning_rate):
        for perceptron in self.perceptrons:
            perceptron.step_update(learning_rate)
