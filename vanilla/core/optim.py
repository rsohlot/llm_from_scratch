from core.tensor import Tensor


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
            self.velocity[i] = [
                [0.0 for _ in range(param.shape[1])] for _ in range(param.shape[0])
            ]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            for r in range(param.shape[0]):
                for c in range(param.shape[1]):
                    grad_val = param.grad[r][c] if param.grad else 0.0

                    if self.momentum > 0:
                        self.velocity[i][r][c] = (
                            self.momentum * self.velocity[i][r][c] + grad_val
                        )
                        param.data[r][c] -= self.lr * self.velocity[i][r][c]
                    else:
                        param.data[r][c] -= self.lr * grad_val


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
            self.m[i] = [
                [0.0 for _ in range(param.shape[1])] for _ in range(param.shape[0])
            ]
            self.v[i] = [
                [0.0 for _ in range(param.shape[1])] for _ in range(param.shape[0])
            ]

    def step(self):
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            for r in range(param.shape[0]):
                for c in range(param.shape[1]):
                    g = grad[r][c] if isinstance(grad, list) and len(grad) > r else 0.0
                    if (
                        isinstance(grad, list)
                        and len(grad) > 0
                        and isinstance(grad[0], list)
                    ):
                        g = grad[r][c]
                    elif isinstance(grad, list):
                        g = grad[r][0] if len(grad) > r else 0.0
                    else:
                        g = 0.0

                    self.m[i][r][c] = (
                        self.beta1 * self.m[i][r][c] + (1 - self.beta1) * g
                    )
                    self.v[i][r][c] = (
                        self.beta2 * self.v[i][r][c] + (1 - self.beta2) * g * g
                    )

                    m_hat = self.m[i][r][c] / (1 - self.beta1**self.t)
                    v_hat = self.v[i][r][c] / (1 - self.beta2**self.t)

                    param.data[r][c] -= self.lr * m_hat / (v_hat**0.5 + self.eps)


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

            for r in range(param.shape[0]):
                for c in range(param.shape[0]):
                    if c < param.shape[1]:
                        g = (
                            param.grad[r][c]
                            if isinstance(param.grad, list)
                            and len(param.grad) > r
                            and len(param.grad[r]) > c
                            else 0.0
                        )

                        self.m[i][r][c] = (
                            self.beta1 * self.m[i][r][c] + (1 - self.beta1) * g
                        )
                        self.v[i][r][c] = (
                            self.beta2 * self.v[i][r][c] + (1 - self.beta2) * g * g
                        )

                        m_hat = self.m[i][r][c] / (1 - self.beta1**self.t)
                        v_hat = self.v[i][r][c] / (1 - self.beta2**self.t)

                        param.data[r][c] -= self.lr * (
                            m_hat / (v_hat**0.5 + self.eps)
                            + self.weight_decay * param.data[r][c]
                        )
