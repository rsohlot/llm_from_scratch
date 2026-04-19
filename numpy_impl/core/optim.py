"""
Optimizers using NumPy.
"""

import numpy as np


class Optimizer:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = {}

        for i, param in enumerate(parameters):
            self.velocity[i] = np.zeros_like(param.data)

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                param.data -= self.lr * self.velocity[i]
            else:
                param.data -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

        for i, param in enumerate(parameters):
            self.m[i] = np.zeros_like(param.data)
            self.v[i] = np.zeros_like(param.data)

    def step(self):
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            g = param.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(
        self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            g = param.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param.data -= self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data
            )
