import math


class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, list):
            if len(data) == 0:
                self.data = []
            elif isinstance(data[0], list):
                self.data = [[float(x) for x in row] for row in data]
            elif isinstance(data[0], (int, float)):
                self.data = [float(x) for x in data]
            else:
                self.data = data
        else:
            self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.shape = self._get_shape()

    def _get_shape(self):
        if not self.data or not isinstance(self.data[0], list):
            return (len(self.data),) if self.data else (0,)
        return (len(self.data), len(self.data[0]) if self.data[0] else 0)

    def zero_grad(self):
        self.grad = None
        self._grad_fn = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.shape == (len(self.data),):
                grad = [[1.0] for _ in range(len(self.data))]
            else:
                grad = [
                    [1.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])
                ]
        self.grad = grad
        if self._grad_fn:
            self._grad_fn()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self._add_scalar(other)
        return self._add_tensor(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._mul_scalar(other)
        return self._mul_tensor(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self._mul_scalar(-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def _add_scalar(self, scalar):
        result = [[x + scalar for x in row] for row in self.data]
        tensor = Tensor(result, requires_grad=self.requires_grad)
        if self.requires_grad:
            original = self

            def grad_fn():
                if original.grad is None:
                    original.grad = [
                        [1.0 for _ in range(original.shape[1])]
                        for _ in range(original.shape[0])
                    ]

            tensor._grad_fn = grad_fn
        return tensor

    def _add_tensor(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                row.append(self.data[i][j] + other_data[i][j])
            result.append(row)
        tensor = Tensor(
            result,
            requires_grad=self.requires_grad
            or (isinstance(other, Tensor) and other.requires_grad),
        )
        if tensor.requires_grad:

            def grad_fn():
                if self.grad is None:
                    self.grad = [
                        [1.0 for _ in range(self.shape[1])]
                        for _ in range(self.shape[0])
                    ]

            tensor._grad_fn = grad_fn
        return tensor

    def _mul_scalar(self, scalar):
        result = [[x * scalar for x in row] for row in self.data]
        tensor = Tensor(result, requires_grad=self.requires_grad)
        if self.requires_grad:
            original = self

            def grad_fn():
                if original.grad is None:
                    original.grad = [
                        [scalar for _ in range(original.shape[1])]
                        for _ in range(original.shape[0])
                    ]

            tensor._grad_fn = grad_fn
        return tensor

    def _mul_tensor(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                row.append(self.data[i][j] * other_data[i][j])
            result.append(row)
        return Tensor(result, requires_grad=self.requires_grad)

    def sum(self, dim=None):
        if dim is None:
            total = sum(sum(row) for row in self.data)
            result = Tensor([[total]], requires_grad=self.requires_grad)
            return result
        return self

    def mean(self, dim=None):
        total = self.sum()
        count = self.shape[0] * self.shape[1]
        return total / count

    def softmax(self, dim=-1):
        max_vals = self.max(dim, keepdim=True)
        exp_data = self.sub(max_vals).exp()
        sum_exp = exp_data.sum(dim, keepdim=True)
        return exp_data.div(sum_exp)

    def max(self, dim=None, keepdim=False):
        if dim is None or dim == -1:
            max_val = max(max(row) for row in self.data)
            if keepdim:
                return Tensor([[max_val]], requires_grad=self.requires_grad)
            return Tensor([max_val], requires_grad=self.requires_grad)
        return self

    def exp(self):
        result = [[math.exp(x) for x in row] for row in self.data]
        return Tensor(result, requires_grad=self.requires_grad)

    def log(self):
        result = [[math.log(max(x, 1e-9)) for x in row] for row in self.data]
        return Tensor(result, requires_grad=self.requires_grad)

    def div(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            result = [
                [self.data[i][j] / other_data[i][j] for j in range(self.shape[1])]
                for i in range(self.shape[0])
            ]
        else:
            result = [[x / other for x in row] for row in self.data]
        return Tensor(result, requires_grad=self.requires_grad)

    def sub(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            result = [
                [self.data[i][j] - other_data[i][j] for j in range(self.shape[1])]
                for i in range(self.shape[0])
            ]
        else:
            result = [[x - other for x in row] for row in self.data]
        return Tensor(result, requires_grad=self.requires_grad)

    def T(self):
        if len(self.shape) == 1:
            return Tensor([[x] for x in self.data], requires_grad=self.requires_grad)
        result = [
            [self.data[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ]
        return Tensor(result, requires_grad=self.requires_grad)

    def view(self, *shape):
        flat = []
        for row in self.data:
            if isinstance(row, list):
                flat.extend(row)
            else:
                flat.append(row)
        return Tensor(flat, requires_grad=self.requires_grad).reshape(shape)

    def reshape(self, shape):
        flat = []
        for row in self.data:
            if isinstance(row, list):
                flat.extend(row)
            else:
                flat.append(row)
        if len(shape) == 2:
            result = []
            for i in range(shape[0]):
                row = flat[i * shape[1] : (i + 1) * shape[1]]
                result.append(row)
            return Tensor(result, requires_grad=self.requires_grad)
        return Tensor(flat, requires_grad=self.requires_grad)

    def transpose(self):
        return self.T()

    def matmul(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(other_data[0])):
                val = sum(
                    self.data[i][k] * other_data[k][j] for k in range(len(self.data[0]))
                )
                row.append(val)
            result.append(row)
        return Tensor(
            result,
            requires_grad=self.requires_grad
            or (isinstance(other, Tensor) and other.requires_grad),
        )

    def unsqueeze(self, dim=0):
        if dim == 0:
            return Tensor([self.data], requires_grad=self.requires_grad)
        return self

    def squeeze(self, dim=None):
        if self.shape[0] == 1:
            return Tensor(self.data[0], requires_grad=self.requires_grad)
        return self

    def cross_entropy(self, targets):
        log_probs = self.log()
        nll = 0.0
        for i in range(len(targets)):
            target_idx = (
                int(targets[i]) if isinstance(targets[i], float) else targets[i]
            )
            nll -= log_probs.data[i][target_idx]
        nll /= len(targets)
        result = Tensor([[nll]], requires_grad=True)
        return result

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data

    def __getitem__(self, key):
        if isinstance(key, int):
            return Tensor(self.data[key], requires_grad=self.requires_grad)
        if isinstance(key, tuple):
            row, col = key
            return Tensor([[self.data[row][col]]], requires_grad=self.requires_grad)
        return self

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.data[key] = value
        elif isinstance(key, tuple):
            row, col = key
            self.data[row][col] = value

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape):
    if len(shape) == 1:
        return Tensor([[0.0] * shape[0]], requires_grad=False)
    return Tensor(
        [[0.0 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False
    )


def ones(*shape):
    if len(shape) == 1:
        return Tensor([[1.0] * shape[0]], requires_grad=False)
    return Tensor(
        [[1.0 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False
    )


def randn(*shape):
    import random

    if len(shape) == 1:
        return Tensor([[random.gauss(0, 1)] * shape[0]], requires_grad=False)
    result = []
    for i in range(shape[0]):
        row = [random.gauss(0, 1) for _ in range(shape[1])]
        result.append(row)
    return Tensor(result, requires_grad=False)


def arange(start, end, step=1):
    result = [[float(i)] for i in range(start, end, step)]
    return Tensor(result, requires_grad=False)
