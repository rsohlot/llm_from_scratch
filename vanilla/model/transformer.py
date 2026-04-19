"""Pure-Python Transformer language model."""

import math
import random

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
        d_model=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=16,
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
        if isinstance(x.data[0], list):
            indices = x.data[0]
        else:
            indices = x.data
        seq_len = len(indices)
        mask = causal_mask(seq_len)

        idx_tensor = Tensor(indices, requires_grad=False)
        h = self.token_embedding.forward(idx_tensor)
        h = self.pos_encoding.forward(h)
        for layer in self.layers:
            h = layer.forward(h, mask)
        h = self.norm.forward(h)
        logits = self.lm_head.forward(h)

        loss = None
        if targets is not None:
            loss = logits.cross_entropy(list(targets))
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
        for layer in self.layers:
            yield layer
        yield self.norm
        yield self.lm_head

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        if isinstance(idx.data[0], list):
            ids = list(idx.data[0])
        else:
            ids = list(idx.data)
        for _ in range(max_new_tokens):
            cond = ids[-self.max_len :] if len(ids) > self.max_len else ids
            logits, _ = self.forward(Tensor([cond], requires_grad=False))
            next_logits = logits.data[-1][:]
            if temperature != 1.0:
                next_logits = [x / temperature for x in next_logits]
            if top_k is not None and top_k < len(next_logits):
                threshold = sorted(next_logits, reverse=True)[top_k - 1]
                next_logits = [x if x >= threshold else -1e9 for x in next_logits]
            m = max(next_logits)
            exps = [math.exp(x - m) for x in next_logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            r = random.random()
            cum = 0.0
            chosen = len(probs) - 1
            for j, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen = j
                    break
            ids.append(chosen)
        self.train()
        return Tensor([ids], requires_grad=False)


def create_gpt_config(config_name="tiny"):
    configs = {
        "tiny": {
            "vocab_size": 64,
            "ctx_len": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
        },
    }
    return configs.get(config_name, configs["tiny"])
