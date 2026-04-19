"""
Core tensor operations using NumPy for high performance.
"""

import numpy as np
import math


class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None
        self._grad_fn = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad
        if self._grad_fn:
            self._grad_fn()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data + other, requires_grad=self.requires_grad)
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        result = Tensor(
            self.data + other_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data * other, requires_grad=self.requires_grad)
            return result
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (1.0 / other)

    def sum(self, dim=None):
        result = Tensor(self.data.sum(axis=dim), requires_grad=self.requires_grad)
        return result

    def mean(self, dim=None):
        return self.sum(dim=dim) / (
            self.data.size if dim is None else self.data.shape[dim]
        )

    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad)

    def log(self):
        return Tensor(
            np.log(np.clip(self.data, 1e-9)), requires_grad=self.requires_grad
        )

    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def matmul(self, other):
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        return Tensor(
            self.data @ other_data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def view(self, *shape):
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    def reshape(self, shape):
        return self.view(*shape)

    def softmax(self, dim=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        return Tensor(
            exp_data / exp_data.sum(axis=dim, keepdims=True),
            requires_grad=self.requires_grad,
        )

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        return Tensor(self.data[key], requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape):
    return Tensor(np.zeros(shape))


def ones(*shape):
    return Tensor(np.ones(shape))


def randn(*shape):
    return Tensor(np.random.randn(*shape))


def arange(start, end, step=1):
    return Tensor(np.arange(start, end, step, dtype=np.float64))
