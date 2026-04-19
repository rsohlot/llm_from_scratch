"""Transformer language model using the autograd Tensor."""

import math

import numpy as np

from core.tensor import Tensor
from core.nn import (
    Module,
    Linear,
    LayerNorm,
    Embedding,
    PositionalEncoding,
    TransformerBlock,
    causal_mask,
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
        dropout=0.0,
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
        self.layers = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        seq_len = x.shape[-1]
        mask = causal_mask(seq_len)

        h = self.token_embedding.forward(x)
        h = self.pos_encoding.forward(h)
        for layer in self.layers:
            h = layer.forward(h, mask)
        h = self.norm.forward(h)
        logits = self.lm_head.forward(h)

        loss = None
        if targets is not None:
            t_arr = np.asarray(targets).reshape(-1)
            loss = logits.cross_entropy(t_arr)
        return logits, loss

    def parameters(self):
        params = self.token_embedding.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def _children(self):
        yield self.token_embedding
        yield self.pos_encoding
        for layer in self.layers:
            yield layer
        yield self.norm
        yield self.lm_head

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        ids = idx.data.astype(np.int64)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        for _ in range(max_new_tokens):
            cond = ids[:, -self.max_len :] if ids.shape[1] > self.max_len else ids
            logits, _ = self.forward(Tensor(cond, requires_grad=False))
            next_logits = logits.data[0, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k < next_logits.size:
                top_idx = np.argpartition(next_logits, -top_k)[-top_k:]
                mask = np.full_like(next_logits, -np.inf)
                mask[top_idx] = next_logits[top_idx]
                next_logits = mask
            shifted = next_logits - next_logits.max()
            exp_logits = np.exp(shifted)
            probs = exp_logits / exp_logits.sum()
            next_token = np.random.choice(len(probs), p=probs)
            ids = np.concatenate([ids, np.array([[next_token]], dtype=np.int64)], axis=1)
        self.train()
        return Tensor(ids, requires_grad=False)


class GPT2(Module):
    def __init__(
        self, vocab_size=50257, ctx_len=1024, n_layer=12, n_head=12, n_embd=768
    ):
        self.transformer = Transformer(
            vocab_size=vocab_size,
            d_model=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            ff_dim=n_embd * 4,
            max_len=ctx_len,
            dropout=0.0,
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
