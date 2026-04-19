"""Pure-Python neural-network layers built on the autograd Tensor."""

import math
import random

from core.tensor import Tensor, hconcat, randn, zeros, ones


class Module:
    training = True

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for child in self._children():
            child.train()

    def eval(self):
        self.training = False
        for child in self._children():
            child.eval()

    def _children(self):
        for v in vars(self).values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        yield item


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        scale = math.sqrt(2.0 / in_features)
        w = randn(in_features, out_features)
        w.data = [[x * scale for x in row] for row in w.data]
        w.requires_grad = True
        self.weight = w
        if bias:
            b = zeros(out_features)
            b.requires_grad = True
            self.bias = b
        else:
            self.bias = None

    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class LayerNorm(Module):
    def __init__(self, features, eps=1e-5):
        self.features = features
        self.eps = eps
        g = ones(features)
        g.requires_grad = True
        b = zeros(features)
        b.requires_grad = True
        self.gamma = g
        self.beta = b

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        std = (var + self.eps) ** 0.5
        x_norm = centered / std
        return self.gamma * x_norm + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        rows, cols = x.shape
        scale = 1.0 / (1.0 - self.p)
        mask = [
            [scale if random.random() >= self.p else 0.0 for _ in range(cols)]
            for _ in range(rows)
        ]
        return x * Tensor(mask, requires_grad=False)


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def _children(self):
        return iter(self.layers)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        scale = 1.0 / math.sqrt(embedding_dim)
        w = randn(num_embeddings, embedding_dim)
        w.data = [[x * scale for x in row] for row in w.data]
        w.requires_grad = True
        self.weight = w

    def forward(self, x):
        return x.embedding_select(self.weight)

    def parameters(self):
        return [self.weight]


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        pe = [[0.0 for _ in range(d_model)] for _ in range(max_len)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pe[pos][i] = math.sin(angle)
                if i + 1 < d_model:
                    pe[pos][i + 1] = math.cos(angle)
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[0]
        pe_slice = [self.pe[i][:] for i in range(seq_len)]
        return x + Tensor(pe_slice, requires_grad=False)


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)
        scale = 1.0 / math.sqrt(self.d_k)

        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.d_k
            end = start + self.d_k
            q_h = Q.col_slice(start, end)
            k_h = K.col_slice(start, end)
            v_h = V.col_slice(start, end)
            scores = q_h.matmul(k_h.T()) * scale
            if mask is not None:
                scores = scores.masked_fill(mask, -1e9)
            attn = scores.softmax(dim=-1)
            head_outputs.append(attn.matmul(v_h))

        concat = hconcat(head_outputs) if len(head_outputs) > 1 else head_outputs[0]
        return self.W_o.forward(concat)

    def parameters(self):
        return (
            self.W_q.parameters()
            + self.W_k.parameters()
            + self.W_v.parameters()
            + self.W_o.parameters()
        )


class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.0):
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, ff_dim),
            GELU(),
            Linear(ff_dim, d_model),
        )

    def forward(self, x, mask=None):
        h = self.norm1.forward(x)
        x = x + self.attention.forward(h, h, h, mask)
        x = x + self.ff.forward(self.norm2.forward(x))
        return x

    def parameters(self):
        return (
            self.attention.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.ff.parameters()
        )


def causal_mask(seq_len):
    """2-D boolean mask: True = position should be zeroed out."""
    return [[j > i for j in range(seq_len)] for i in range(seq_len)]
