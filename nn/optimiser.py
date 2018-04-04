import numpy as np


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
