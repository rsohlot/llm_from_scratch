"""
Neural Network layers using NumPy for high performance.
"""

import numpy as np
import math
from core.tensor import Tensor, tensor, zeros, ones, randn


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale, requires_grad=True
        )

        if bias:
            self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x_data = x.data if isinstance(x, Tensor) else np.array(x)

        if x_data.ndim == 3:
            batch_size, seq_len, _ = x_data.shape
            x_2d = x_data.reshape(-1, self.in_features)
            result_2d = x_2d @ self.weight.data
            if self.bias:
                result_2d = result_2d + self.bias.data
            result = result_2d.reshape(batch_size, seq_len, self.out_features)
        else:
            result = x_data @ self.weight.data
            if self.bias:
                result = result + self.bias.data

        return Tensor(result, requires_grad=x.requires_grad)

    def parameters(self):
        params = [self.weight]
        if self.bias:
            params.append(self.bias)
        return params


class LayerNorm(Module):
    def __init__(self, features, eps=1e-5):
        self.features = features
        self.eps = eps
        self.gamma = Tensor(np.ones((1, features)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, features)), requires_grad=True)

    def forward(self, x):
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x.data - mean) / std
        return self.gamma * Tensor(x_norm, requires_grad=x.requires_grad) + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p

    def forward(self, x):
        return x


class GELU(Module):
    def forward(self, x):
        return Tensor(
            0.5
            * x.data
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3))),
            requires_grad=x.requires_grad,
        )


class Softmax(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return x.softmax(dim=self.dim)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        scale = math.sqrt(1.0 / num_embeddings)
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * scale, requires_grad=True
        )

    def forward(self, x):
        indices = x.data.astype(int)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
        indices = np.clip(indices, 0, self.num_embeddings - 1)
        embeddings = self.weight.data[indices]
        if embeddings.ndim == 2:
            embeddings = embeddings.reshape(1, embeddings.shape[0], embeddings.shape[1])
        return Tensor(embeddings, requires_grad=x.requires_grad)

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

        self.pe = Tensor(pe, requires_grad=False)

    def forward(self, x):
        seq_len = x.data.shape[1]
        pe_slice = self.pe.data[:seq_len, :]
        return x + Tensor(pe_slice, requires_grad=False)


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
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

        Q_data = Q.data
        K_data = K.data
        V_data = V.data

        if Q_data.ndim == 2:
            Q_data = Q_data.reshape(1, -1, self.d_model)
            K_data = K_data.reshape(1, -1, self.d_model)
            V_data = V_data.reshape(1, -1, self.d_model)

        batch_size, seq_len, _ = Q_data.shape

        Q = Q_data.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            0, 2, 1, 3
        )
        K = K_data.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            0, 2, 1, 3
        )
        V = V_data.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            0, 2, 1, 3
        )

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        attn_weights = Tensor(scores).softmax(dim=-1)

        context = np.matmul(attn_weights.data, V)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o.forward(Tensor(context, requires_grad=query.requires_grad))
        return output

    def parameters(self):
        return (
            self.W_q.parameters()
            + self.W_k.parameters()
            + self.W_v.parameters()
            + self.W_o.parameters()
        )


class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, ff_dim), GELU(), Linear(ff_dim, d_model), Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)

        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)

        return x

    def parameters(self):
        return (
            self.attention.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.ff.parameters()
        )
