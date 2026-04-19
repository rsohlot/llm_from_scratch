"""
Transformer model using NumPy for high performance.
"""

import numpy as np
import math
import random
from core.tensor import Tensor, tensor
from core.nn import (
    Module,
    Linear,
    LayerNorm,
    Dropout,
    GELU,
    Embedding,
    PositionalEncoding,
    TransformerBlock,
    Sequential,
)


class Transformer(Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        max_len=512,
        dropout=0.1,
        pad_idx=0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, ff_dim, dropout))

        self.norm = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        x = self.token_embedding.forward(x)
        x = self.pos_encoding.forward(x)

        for layer in self.layers:
            x = layer.forward(x)

        x = self.norm.forward(x)

        logits = self.lm_head.forward(x)

        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)

        return logits, loss

    def _compute_loss(self, logits, targets):
        logits_data = logits.data

        if logits_data.ndim == 3:
            batch_size, seq_len, vocab_size = logits_data.shape
        else:
            seq_len, vocab_size = logits_data.shape
            batch_size = 1
            logits_data = logits_data.reshape(1, seq_len, vocab_size)

        log_probs = logits_data - np.max(logits_data, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))

        targets_arr = np.array(targets).reshape(-1)

        if len(targets_arr) > seq_len:
            targets_arr = targets_arr[:seq_len]

        nll = 0.0
        for i, target in enumerate(targets_arr):
            if i < seq_len:
                nll -= log_probs[0, i, target]
        nll /= len(targets_arr)

        return Tensor(np.array([[nll]]), requires_grad=True)

    def parameters(self):
        params = self.token_embedding.parameters()
        params.extend(self.pos_encoding.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx
            if idx_cond.data.shape[1] > self.max_len:
                idx_cond = Tensor(
                    idx_cond.data[:, -self.max_len :], requires_grad=False
                )

            logits, _ = self.forward(idx_cond)

            logits_data = logits.data
            if logits_data.ndim == 3:
                logits = logits_data[0, -1, :] / temperature
            else:
                logits = logits_data[-1, :] / temperature

            if top_k is not None:
                top_indices = np.argsort(logits)[-top_k:]
                mask = np.full_like(logits, -float("inf"))
                mask[top_indices] = logits[top_indices]
                logits = mask

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            next_token = np.random.choice(len(probs), p=probs)

            idx = Tensor(
                np.concatenate([idx_cond.data, [[next_token]]], axis=1),
                requires_grad=False,
            )

        return idx


class GPT2(Module):
    def __init__(
        self, vocab_size=50257, ctx_len=1024, n_layer=12, n_head=12, n_embd=768
    ):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.transformer = Transformer(
            vocab_size=vocab_size,
            d_model=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            ff_dim=n_embd * 4,
            max_len=ctx_len,
            dropout=0.1,
        )

    def forward(self, x, targets=None):
        return self.transformer.forward(x, targets)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return self.transformer.generate(idx, max_new_tokens, temperature, top_k)

    def parameters(self):
        return self.transformer.parameters()


def create_gpt_config(config_name="124M"):
    configs = {
        "124M": {
            "vocab_size": 50257,
            "ctx_len": 1024,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
        },
        "355M": {
            "vocab_size": 50257,
            "ctx_len": 1024,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
        },
        "774M": {
            "vocab_size": 50257,
            "ctx_len": 1024,
            "n_layer": 36,
            "n_head": 16,
            "n_embd": 1280,
        },
    }
    return configs.get(config_name, configs["124M"])
