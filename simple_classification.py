import numpy as np
import matplotlib.pyplot as plt

import nn


def simple_classification():
    # # N batch size
    # # D_in is input dimension
    # # H is the hidden dimension
    # # D_out is the output dimension
    np.random.seed(3)
    N, D_in, H, D_out = 64, 1, 100, 1
    model = nn.mlp.MultiLayerPerceptron([D_in, H, D_out])

    # Create random input and output data
    x = np.random.randn(D_in, N)
    # y = np.random.randn(D_out, N)

    # x = np.linspace(0, 100, N).reshape(1, N)
    # x = np.ones((1, 1))
    y = x ** 2 + np.random.randn(D_out, N) * 0.01 + 0.5

    loss_function = nn.lsf.MSELoss()
    step_policy = {
        "type": "step",
        "base_lr": 0.01,
        "max_iter": 200,
        "step_size": 100,
        "lr_multiplier": 0.5,
    }

    poly_policy = {"type": "poly", "base_lr": 0.0001, "max_iter": 600}

    policy = poly_policy
    optimiser = nn.opt.Optimiser(policy=policy)

    axes = plt.gca()
    lines = axes.plot(x, y, "ro", x, y, "go")
    # line_model = axes.plot(x, y, 'go')
    plt.draw()
    plt.pause(1e-3)
    # ax.plot(x, y, 'ro')

    for t in range(policy["max_iter"]):
        out = model.forward(x)
        # print('y', out)
        # raw_input('enter to continue')
        loss = loss_function(out, y)
        model.backward(loss_function.backward())
        lr = optimiser.get_learning_rate(t)
        model.step_update(lr)
        print(t, lr, loss)

        lines[1].set_xdata(x)
        lines[1].set_ydata(out)
        plt.draw()
        plt.pause(1e-3)
        # ax.plot(x, out, 'bo')
        # plt.show()

        # plt.pause(0.00001)

    x = np.random.randn(D_in, N)
    y_model = model.forward(x)
    line_final = axes.plot(x, y_model, "bo")
    plt.draw()
    print(y_model)
    # ax.plot(x, y_model, 'go')

    plt.show()


if __name__ == "__main__":
    simple_classification()
