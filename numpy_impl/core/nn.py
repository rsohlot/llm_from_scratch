"""Neural network layers built on the autograd Tensor."""

import math

import numpy as np

from core.tensor import Tensor


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
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale, requires_grad=True
        )
        self.bias = (
            Tensor(np.zeros((out_features,)), requires_grad=True) if bias else None
        )

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
        self.gamma = Tensor(np.ones((features,)), requires_grad=True)
        self.beta = Tensor(np.zeros((features,)), requires_grad=True)

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
        mask = (np.random.rand(*x.shape) >= self.p).astype(np.float64) / (1.0 - self.p)
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
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * scale, requires_grad=True
        )

    def forward(self, x):
        return x.embedding_select(self.weight)

    def parameters(self):
        return [self.weight]


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[-2]
        pe_slice = Tensor(self.pe[:seq_len, :], requires_grad=False)
        return x + pe_slice


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
        self.attn_dropout = Dropout(dropout)

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, t, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * dk)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.W_q.forward(query))
        k = self._split_heads(self.W_k.forward(key))
        v = self._split_heads(self.W_v.forward(value))

        scores = q.matmul(k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout.forward(attn)
        context = attn.matmul(v)
        return self.W_o.forward(self._merge_heads(context))

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
            Linear(d_model, ff_dim), GELU(), Linear(ff_dim, d_model), Dropout(dropout)
        )
        self.resid_dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm1.forward(x)
        x = x + self.resid_dropout.forward(self.attention.forward(h, h, h, mask))
        x = x + self.resid_dropout.forward(self.ff.forward(self.norm2.forward(x)))
        return x

    def parameters(self):
        return (
            self.attention.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.ff.parameters()
        )


def causal_mask(seq_len):
    """Returns a boolean mask of shape (1, 1, T, T) where True = blocked."""
    m = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return m.reshape(1, 1, seq_len, seq_len)
