"""Autograd-enabled Tensor with a NumPy backend."""

import numpy as np


def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            arr = data.data
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)
        if arr.dtype not in (np.int32, np.int64):
            arr = arr.astype(np.float64)
        self.data = arr
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = tuple(_children)
        self._backward_fn = lambda: None
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self):
        self.grad = None

    def _accum(self, g):
        g = np.asarray(g, dtype=np.float64)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        self.grad = np.asarray(grad, dtype=np.float64)
        topo, visited = [], set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        for v in reversed(topo):
            v._backward_fn()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _bw():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.shape))

        out._backward_fn = _bw
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="sub",
        )

        def _bw():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(-out.grad, other.shape))

        out._backward_fn = _bw
        return out

    def __rsub__(self, other):
        return (Tensor(other) if not isinstance(other, Tensor) else other) - self

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            scalar = float(other)
            out = Tensor(
                self.data * scalar,
                requires_grad=self.requires_grad,
                _children=(self,),
                _op="mul_scalar",
            )

            def _bw():
                if self.requires_grad:
                    self._accum(out.grad * scalar)

            out._backward_fn = _bw
            return out
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _bw():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad * self.data, other.shape))

        out._backward_fn = _bw
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / float(other))
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _bw():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                other._accum(
                    _unbroadcast(
                        -out.grad * self.data / (other.data * other.data), other.shape
                    )
                )

        out._backward_fn = _bw
        return out

    def __pow__(self, p):
        p = float(p)
        out_data = self.data**p
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="pow",
        )

        def _bw():
            if self.requires_grad:
                self._accum(out.grad * p * self.data ** (p - 1))

        out._backward_fn = _bw
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _bw():
            if self.requires_grad:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accum(_unbroadcast(g, self.shape))
            if other.requires_grad:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accum(_unbroadcast(g, other.shape))

        out._backward_fn = _bw
        return out

    def sum(self, dim=None, keepdim=False):
        out = Tensor(
            self.data.sum(axis=dim, keepdims=keepdim),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _bw():
            if self.requires_grad:
                g = out.grad
                if dim is None:
                    g = np.ones_like(self.data) * g
                else:
                    if not keepdim:
                        g = np.expand_dims(g, axis=dim)
                    g = np.broadcast_to(g, self.shape).copy()
                self._accum(g)

        out._backward_fn = _bw
        return out

    def mean(self, dim=None, keepdim=False):
        n = self.data.size if dim is None else self.data.shape[dim]
        return self.sum(dim=dim, keepdim=keepdim) / n

    def exp(self):
        data = np.exp(self.data)
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="exp"
        )

        def _bw():
            if self.requires_grad:
                self._accum(out.grad * data)

        out._backward_fn = _bw
        return out

    def log(self):
        out = Tensor(
            np.log(np.clip(self.data, 1e-12, None)),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _bw():
            if self.requires_grad:
                self._accum(out.grad / np.clip(self.data, 1e-12, None))

        out._backward_fn = _bw
        return out

    def transpose(self, *axes):
        if not axes:
            data = self.data.T
            inv = None
        else:
            data = self.data.transpose(axes)
            inv = tuple(int(i) for i in np.argsort(axes))
        out = Tensor(
            data, requires_grad=self.requires_grad, _children=(self,), _op="transpose"
        )

        def _bw():
            if self.requires_grad:
                if inv is None:
                    self._accum(out.grad.T)
                else:
                    self._accum(out.grad.transpose(inv))

        out._backward_fn = _bw
        return out

    def T(self):
        return self.transpose()

    def swapaxes(self, a, b):
        out = Tensor(
            np.swapaxes(self.data, a, b),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="swapaxes",
        )

        def _bw():
            if self.requires_grad:
                self._accum(np.swapaxes(out.grad, a, b))

        out._backward_fn = _bw
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.shape
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _bw():
            if self.requires_grad:
                self._accum(out.grad.reshape(original_shape))

        out._backward_fn = _bw
        return out

    def view(self, *shape):
        return self.reshape(*shape)

    def masked_fill(self, mask, value):
        mask_arr = np.asarray(mask, dtype=bool)
        value = float(value)
        out = Tensor(
            np.where(mask_arr, value, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="masked_fill",
        )

        def _bw():
            if self.requires_grad:
                self._accum(np.where(mask_arr, 0.0, out.grad))

        out._backward_fn = _bw
        return out

    def softmax(self, dim=-1):
        x = self.data
        shifted = x - x.max(axis=dim, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / exp_x.sum(axis=dim, keepdims=True)
        out = Tensor(
            probs,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax",
        )

        def _bw():
            if self.requires_grad:
                dot = (out.grad * probs).sum(axis=dim, keepdims=True)
                self._accum(probs * (out.grad - dot))

        out._backward_fn = _bw
        return out

    def gelu(self):
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x**3)
        t = np.tanh(inner)
        out_data = 0.5 * x * (1.0 + t)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="gelu",
        )

        def _bw():
            if self.requires_grad:
                sech2 = 1.0 - t * t
                dinner_dx = c * (1.0 + 3.0 * 0.044715 * x * x)
                local = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner_dx
                self._accum(out.grad * local)

        out._backward_fn = _bw
        return out

    def cross_entropy(self, targets):
        x = self.data
        flat = x.reshape(-1, x.shape[-1])
        t = np.asarray(targets).reshape(-1).astype(np.int64)
        shifted = flat - flat.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        n = flat.shape[0]
        nll = -log_probs[np.arange(n), t].mean()
        out = Tensor(
            np.array(nll),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="cross_entropy",
        )

        def _bw():
            if self.requires_grad:
                probs = np.exp(log_probs)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n), t] = 1.0
                grad_flat = (probs - one_hot) / n * out.grad
                self._accum(grad_flat.reshape(x.shape))

        out._backward_fn = _bw
        return out

    def embedding_select(self, weight):
        idx = self.data.astype(np.int64)
        out = Tensor(
            weight.data[idx],
            requires_grad=weight.requires_grad,
            _children=(weight,),
            _op="embedding_select",
        )

        def _bw():
            if weight.requires_grad:
                g = np.zeros_like(weight.data, dtype=np.float64)
                np.add.at(g, idx, out.grad)
                weight._accum(g)

        out._backward_fn = _bw
        return out

    def __getitem__(self, key):
        out = Tensor(
            self.data[key],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _bw():
            if self.requires_grad:
                g = np.zeros_like(self.data, dtype=np.float64)
                np.add.at(g, key, out.grad)
                self._accum(g)

        out._backward_fn = _bw
        return out

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape):
    return Tensor(np.zeros(shape), requires_grad=False)


def ones(*shape):
    return Tensor(np.ones(shape), requires_grad=False)


def randn(*shape):
    return Tensor(np.random.randn(*shape), requires_grad=False)


def arange(start, end, step=1):
    return Tensor(np.arange(start, end, step, dtype=np.float64))
