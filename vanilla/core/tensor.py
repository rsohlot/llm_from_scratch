"""Pure-Python autograd-enabled Tensor.

Supports 1-D and 2-D tensors stored as plain (nested) Python lists. Broadcasting
is limited to the cases the transformer actually needs:
  - scalar + tensor
  - 1-D row vector added to a 2-D tensor (bias-style broadcast)
  - 2-D tensor with leading dim 1 broadcast along rows

Every differentiable op records its parents and sets a `_backward_fn`. Calling
`backward()` performs a topological sort and accumulates gradients through the
graph.
"""

import math
import random


def _shape(data):
    if isinstance(data, list):
        if not data:
            return (0,)
        if isinstance(data[0], list):
            return (len(data), len(data[0]) if data[0] else 0)
        return (len(data),)
    return ()


def _zeros(shape):
    if len(shape) == 0:
        return 0.0
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def _copy(data):
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            return [row[:] for row in data]
        return data[:]
    return data


def _add_data(a, b):
    """a + b elementwise, same shape (no broadcast). Works for 1-D/2-D/scalar."""
    if isinstance(a, list) and a and isinstance(a[0], list):
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    if isinstance(a, list):
        return [a[i] + b[i] for i in range(len(a))]
    return a + b


def _scale(a, s):
    if isinstance(a, list) and a and isinstance(a[0], list):
        return [[x * s for x in row] for row in a]
    if isinstance(a, list):
        return [x * s for x in a]
    return a * s


def _is_2d(x):
    return isinstance(x, list) and x and isinstance(x[0], list)


def _broadcast_op(a, b, fn):
    """Apply fn(x, y) elementwise, with broadcasting between scalars, 1-D and 2-D."""
    if not _is_2d(a) and _is_2d(b):
        return _broadcast_op(b, a, lambda x, y: fn(y, x))
    if not _is_2d(a):
        if isinstance(a, list) and isinstance(b, list):
            return [fn(a[i], b[i]) for i in range(len(a))]
        if isinstance(a, list):
            return [fn(ai, b) for ai in a]
        if isinstance(b, list):
            return [fn(a, bi) for bi in b]
        return fn(a, b)
    rows, cols = len(a), len(a[0])
    if isinstance(b, (int, float)):
        return [[fn(a[i][j], b) for j in range(cols)] for i in range(rows)]
    if _is_2d(b):
        br, bc = len(b), len(b[0])
        if br == rows and bc == cols:
            return [[fn(a[i][j], b[i][j]) for j in range(cols)] for i in range(rows)]
        if br == 1 and bc == cols:
            return [[fn(a[i][j], b[0][j]) for j in range(cols)] for i in range(rows)]
        if br == rows and bc == 1:
            return [[fn(a[i][j], b[i][0]) for j in range(cols)] for i in range(rows)]
        if br == 1 and bc == 1:
            return [[fn(a[i][j], b[0][0]) for j in range(cols)] for i in range(rows)]
        raise ValueError(f"cannot broadcast {(br, bc)} to {(rows, cols)}")
    return [[fn(a[i][j], b[j]) for j in range(cols)] for i in range(rows)]


def _broadcast_add(a, b):
    return _broadcast_op(a, b, lambda x, y: x + y)


def _broadcast_mul(a, b):
    return _broadcast_op(a, b, lambda x, y: x * y)


def _broadcast_div(a, b):
    return _broadcast_op(a, b, lambda x, y: x / y)


def _unbroadcast(grad, target_shape):
    """Reduce grad so it matches target_shape after summing broadcast axes."""
    if len(target_shape) == 0:
        total = 0.0
        if isinstance(grad, list):
            for row in grad:
                total += sum(row) if isinstance(row, list) else row
        else:
            total = grad
        return total
    if len(target_shape) == 1:
        if isinstance(grad, list) and grad and isinstance(grad[0], list):
            C = target_shape[0]
            return [sum(grad[i][j] for i in range(len(grad))) for j in range(C)]
        return grad
    tr, tc = target_shape
    gr, gc = len(grad), len(grad[0])
    if gr == tr and gc == tc:
        return grad
    if tr == 1 and tc == gc:
        return [[sum(grad[i][j] for i in range(gr)) for j in range(tc)]]
    if tc == 1 and tr == gr:
        return [[sum(grad[i][j] for j in range(gc))] for i in range(tr)]
    if tr == 1 and tc == 1:
        return [[sum(sum(row) for row in grad)]]
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = _copy(data.data)
        elif isinstance(data, list):
            data = _copy(data)
            if data and isinstance(data[0], list):
                data = [[float(x) for x in row] for row in data]
            else:
                data = [float(x) for x in data]
        self.data = data
        self.shape = _shape(self.data)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = tuple(_children)
        self._backward_fn = lambda: None
        self._op = _op

    def zero_grad(self):
        self.grad = None

    def _accum(self, g):
        if self.grad is None:
            self.grad = _copy(g)
        else:
            self.grad = _add_data(self.grad, g)

    def backward(self, grad=None):
        if grad is None:
            grad = (
                [[1.0] * self.shape[1] for _ in range(self.shape[0])]
                if len(self.shape) == 2
                else [1.0] * self.shape[0]
                if len(self.shape) == 1
                else 1.0
            )
        self.grad = grad
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

    # ---- arithmetic ----
    def __add__(self, other):
        if isinstance(other, (int, float)):
            data = _broadcast_add(self.data, other)
            out = Tensor(
                data,
                requires_grad=self.requires_grad,
                _children=(self,),
                _op="add_scalar",
            )

            def _bw():
                if self.requires_grad:
                    self._accum(out.grad)

            out._backward_fn = _bw
            return out
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = _broadcast_add(self.data, other.data)
        out = Tensor(
            data,
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

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(
                _scale(self.data, float(other)),
                requires_grad=self.requires_grad,
                _children=(self,),
                _op="mul_scalar",
            )

            def _bw():
                if self.requires_grad:
                    self._accum(_scale(out.grad, float(other)))

            out._backward_fn = _bw
            return out
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = _broadcast_mul(self.data, other.data)
        out = Tensor(
            data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _bw():
            if self.requires_grad:
                self._accum(_unbroadcast(_broadcast_mul(out.grad, other.data), self.shape))
            if other.requires_grad:
                other._accum(
                    _unbroadcast(_broadcast_mul(out.grad, self.data), other.shape)
                )

        out._backward_fn = _bw
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / float(other))
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = _broadcast_div(self.data, other.data)
        out = Tensor(
            data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _bw():
            if self.requires_grad:
                self._accum(
                    _unbroadcast(_broadcast_div(out.grad, other.data), self.shape)
                )
            if other.requires_grad:
                b_sq = _broadcast_op(other.data, other.data, lambda x, y: x * y)
                numerator = _broadcast_op(
                    out.grad, self.data, lambda go, a: -go * a
                )
                grad_b = _broadcast_div(numerator, b_sq)
                other._accum(_unbroadcast(grad_b, other.shape))

        out._backward_fn = _bw
        return out

    def __pow__(self, p):
        p = float(p)
        rows, cols = self.shape
        data = [[self.data[i][j] ** p for j in range(cols)] for i in range(rows)]
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="pow")

        def _bw():
            if self.requires_grad:
                g = [
                    [
                        out.grad[i][j] * p * self.data[i][j] ** (p - 1)
                        for j in range(cols)
                    ]
                    for i in range(rows)
                ]
                self._accum(g)

        out._backward_fn = _bw
        return out

    # ---- linear algebra ----
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        m, k = self.shape
        k2, n = other.shape
        assert k == k2, f"matmul shape mismatch: {self.shape} x {other.shape}"
        a, b = self.data, other.data
        data = [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
        out = Tensor(
            data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _bw():
            go = out.grad
            if self.requires_grad:
                # dA = dC @ B.T: shape (m, k)
                dA = [
                    [sum(go[i][j] * b[ii][j] for j in range(n)) for ii in range(k)]
                    for i in range(m)
                ]
                self._accum(dA)
            if other.requires_grad:
                # dB = A.T @ dC: shape (k, n)
                dB = [
                    [sum(a[i][ii] * go[i][j] for i in range(m)) for j in range(n)]
                    for ii in range(k)
                ]
                other._accum(dB)

        out._backward_fn = _bw
        return out

    def T(self):
        rows, cols = self.shape
        data = [[self.data[i][j] for i in range(rows)] for j in range(cols)]
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="T")

        def _bw():
            if self.requires_grad:
                g = [[out.grad[j][i] for j in range(cols)] for i in range(rows)]
                self._accum(g)

        out._backward_fn = _bw
        return out

    def transpose(self):
        return self.T()

    # ---- reductions ----
    def sum(self, dim=None, keepdim=False):
        if dim is None:
            total = 0.0
            if len(self.shape) == 2:
                for row in self.data:
                    total += sum(row)
            else:
                total = sum(self.data)
            out_data = [[total]] if keepdim else [total]
            out = Tensor(
                out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum"
            )

            def _bw():
                if self.requires_grad:
                    g_scalar = out.grad[0][0] if keepdim else out.grad[0]
                    if len(self.shape) == 2:
                        rows, cols = self.shape
                        g = [[g_scalar] * cols for _ in range(rows)]
                    else:
                        g = [g_scalar] * self.shape[0]
                    self._accum(g)

            out._backward_fn = _bw
            return out
        rows, cols = self.shape
        if dim == -1 or dim == 1:
            reduced = [sum(row) for row in self.data]
            out_data = [[x] for x in reduced] if keepdim else reduced
            out = Tensor(
                out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum_row"
            )

            def _bw():
                if self.requires_grad:
                    if keepdim:
                        g = [[out.grad[i][0]] * cols for i in range(rows)]
                    else:
                        g = [[out.grad[i]] * cols for i in range(rows)]
                    self._accum(g)

            out._backward_fn = _bw
            return out
        raise NotImplementedError(f"sum dim={dim} not supported")

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            n = self.shape[0] * (self.shape[1] if len(self.shape) == 2 else 1)
        else:
            n = self.shape[-1]
        return self.sum(dim=dim, keepdim=keepdim) / n

    # ---- elementwise ----
    def exp(self):
        rows, cols = self.shape
        data = [[math.exp(self.data[i][j]) for j in range(cols)] for i in range(rows)]
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _bw():
            if self.requires_grad:
                g = [
                    [out.grad[i][j] * data[i][j] for j in range(cols)]
                    for i in range(rows)
                ]
                self._accum(g)

        out._backward_fn = _bw
        return out

    def log(self):
        rows, cols = self.shape
        data = [
            [math.log(max(self.data[i][j], 1e-12)) for j in range(cols)]
            for i in range(rows)
        ]
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _bw():
            if self.requires_grad:
                g = [
                    [out.grad[i][j] / max(self.data[i][j], 1e-12) for j in range(cols)]
                    for i in range(rows)
                ]
                self._accum(g)

        out._backward_fn = _bw
        return out

    def gelu(self):
        rows, cols = self.shape
        c = math.sqrt(2.0 / math.pi)
        t_vals = [
            [math.tanh(c * (self.data[i][j] + 0.044715 * self.data[i][j] ** 3)) for j in range(cols)]
            for i in range(rows)
        ]
        data = [
            [0.5 * self.data[i][j] * (1.0 + t_vals[i][j]) for j in range(cols)]
            for i in range(rows)
        ]
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op="gelu")

        def _bw():
            if self.requires_grad:
                g = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        x = self.data[i][j]
                        t = t_vals[i][j]
                        sech2 = 1.0 - t * t
                        dinner = c * (1.0 + 3.0 * 0.044715 * x * x)
                        local = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner
                        row.append(out.grad[i][j] * local)
                    g.append(row)
                self._accum(g)

        out._backward_fn = _bw
        return out

    # ---- softmax / loss ----
    def softmax(self, dim=-1):
        rows, cols = self.shape
        probs = []
        for row in self.data:
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps)
            probs.append([e / s for e in exps])
        out = Tensor(
            probs, requires_grad=self.requires_grad, _children=(self,), _op="softmax"
        )

        def _bw():
            if self.requires_grad:
                g = []
                for i in range(rows):
                    p = probs[i]
                    go_row = out.grad[i]
                    dot = sum(go_row[j] * p[j] for j in range(cols))
                    g.append([p[j] * (go_row[j] - dot) for j in range(cols)])
                self._accum(g)

        out._backward_fn = _bw
        return out

    def masked_fill(self, mask, value):
        """mask is a 2-D boolean list matching shape; True = replace with value."""
        rows, cols = self.shape
        data = [
            [value if mask[i][j] else self.data[i][j] for j in range(cols)]
            for i in range(rows)
        ]
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="masked_fill",
        )

        def _bw():
            if self.requires_grad:
                g = [
                    [0.0 if mask[i][j] else out.grad[i][j] for j in range(cols)]
                    for i in range(rows)
                ]
                self._accum(g)

        out._backward_fn = _bw
        return out

    def cross_entropy(self, targets):
        """self: [T, V] logits. targets: list of int length T."""
        rows, cols = self.shape
        log_probs = []
        for i in range(rows):
            row = self.data[i]
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps)
            lse = m + math.log(s)
            log_probs.append([x - lse for x in row])
        nll = 0.0
        for i, t in enumerate(targets):
            nll -= log_probs[i][int(t)]
        nll /= len(targets)
        out = Tensor(
            [[nll]],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="cross_entropy",
        )

        def _bw():
            if self.requires_grad:
                coef = out.grad[0][0] / rows
                g = []
                for i in range(rows):
                    p = [math.exp(lp) for lp in log_probs[i]]
                    t = int(targets[i])
                    row = [coef * p[j] for j in range(cols)]
                    row[t] -= coef
                    g.append(row)
                self._accum(g)

        out._backward_fn = _bw
        return out

    # ---- indexing / structural ----
    def embedding_select(self, weight):
        """self holds int indices (row of ints). weight is [V, D]."""
        idx = self.data if not (self.data and isinstance(self.data[0], list)) else self.data[0]
        idx = [int(i) for i in idx]
        data = [weight.data[i][:] for i in idx]
        out = Tensor(
            data,
            requires_grad=weight.requires_grad,
            _children=(weight,),
            _op="embedding",
        )

        def _bw():
            if weight.requires_grad:
                D = weight.shape[1]
                g = [[0.0] * D for _ in range(weight.shape[0])]
                for pos, i in enumerate(idx):
                    for j in range(D):
                        g[i][j] += out.grad[pos][j]
                weight._accum(g)

        out._backward_fn = _bw
        return out

    def col_slice(self, start, end):
        """Select columns [start, end) from a 2-D tensor."""
        rows = self.shape[0]
        data = [self.data[i][start:end] for i in range(rows)]
        out = Tensor(
            data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="col_slice",
        )

        def _bw():
            if self.requires_grad:
                g = [[0.0] * self.shape[1] for _ in range(rows)]
                for i in range(rows):
                    for j in range(end - start):
                        g[i][start + j] = out.grad[i][j]
                self._accum(g)

        out._backward_fn = _bw
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def hconcat(tensors):
    """Horizontal concat of a list of 2-D tensors along columns."""
    rows = tensors[0].shape[0]
    widths = [t.shape[1] for t in tensors]
    data = []
    for i in range(rows):
        row = []
        for t in tensors:
            row.extend(t.data[i])
        data.append(row)
    out = Tensor(
        data,
        requires_grad=any(t.requires_grad for t in tensors),
        _children=tuple(tensors),
        _op="hconcat",
    )

    def _bw():
        offset = 0
        for t, w in zip(tensors, widths):
            if t.requires_grad:
                g = [[out.grad[i][offset + j] for j in range(w)] for i in range(rows)]
                t._accum(g)
            offset += w

    out._backward_fn = _bw
    return out


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape):
    return Tensor(_zeros(shape), requires_grad=False)


def ones(*shape):
    if len(shape) == 1:
        return Tensor([1.0] * shape[0], requires_grad=False)
    return Tensor(
        [[1.0 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False
    )


def randn(*shape):
    if len(shape) == 1:
        return Tensor([random.gauss(0.0, 1.0) for _ in range(shape[0])])
    return Tensor(
        [[random.gauss(0.0, 1.0) for _ in range(shape[1])] for _ in range(shape[0])]
    )


def arange(start, end, step=1):
    return Tensor([float(i) for i in range(start, end, step)])
