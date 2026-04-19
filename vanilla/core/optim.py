"""Optimizers operating on Tensor parameters (pure Python)."""


def _zeros_like(data):
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [[0.0 for _ in row] for row in data]
    return [0.0 for _ in data]


def _apply(data, fn):
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [[fn(x) for x in row] for row in data]
    return [fn(x) for x in data]


def _pairwise(a, b, fn):
    if isinstance(a, list) and a and isinstance(a[0], list):
        return [[fn(a[i][j], b[i][j]) for j in range(len(a[0]))] for i in range(len(a))]
    return [fn(a[i], b[i]) for i in range(len(a))]


class Optimizer:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [_zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum > 0:
                self.velocity[i] = _pairwise(
                    self.velocity[i], p.grad, lambda v, g: self.momentum * v + g
                )
                upd = self.velocity[i]
            else:
                upd = p.grad
            p.data = _pairwise(p.data, upd, lambda x, g: x - self.lr * g)


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [_zeros_like(p.data) for p in self.parameters]
        self.v = [_zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2, eps, lr = self.beta1, self.beta2, self.eps, self.lr
        bc1 = 1 - b1**self.t
        bc2 = 1 - b2**self.t
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.m[i] = _pairwise(self.m[i], p.grad, lambda m, g: b1 * m + (1 - b1) * g)
            self.v[i] = _pairwise(
                self.v[i], p.grad, lambda v, g: b2 * v + (1 - b2) * g * g
            )
            m_hat = _apply(self.m[i], lambda x: x / bc1)
            v_hat = _apply(self.v[i], lambda x: x / bc2)
            upd = _pairwise(m_hat, v_hat, lambda m, v: m / (v**0.5 + eps))
            p.data = _pairwise(p.data, upd, lambda x, u: x - lr * u)


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        b1, b2, eps, lr, wd = self.beta1, self.beta2, self.eps, self.lr, self.weight_decay
        bc1 = 1 - b1**self.t
        bc2 = 1 - b2**self.t
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.m[i] = _pairwise(self.m[i], p.grad, lambda m, g: b1 * m + (1 - b1) * g)
            self.v[i] = _pairwise(
                self.v[i], p.grad, lambda v, g: b2 * v + (1 - b2) * g * g
            )
            m_hat = _apply(self.m[i], lambda x: x / bc1)
            v_hat = _apply(self.v[i], lambda x: x / bc2)
            upd = _pairwise(m_hat, v_hat, lambda m, v: m / (v**0.5 + eps))
            p.data = _pairwise(
                p.data, upd, lambda x, u: x - lr * (u + wd * x)
            )
