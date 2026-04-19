from core.tensor import Tensor, tensor, zeros, randn
from core.nn import (
    Module,
    Linear,
    LayerNorm,
    Dropout,
    GELU,
    Softmax,
    Embedding,
    PositionalEncoding,
    Sequential,
)
import math
import random


class SimpleAttention(Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
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

        seq_len = len(Q.data)
        d_k = self.d_model // self.num_heads

        Q_split = []
        K_split = []
        V_split = []

        for h in range(self.num_heads):
            start = h * d_k
            end = start + d_k
            Q_split.append(
                [[Q.data[i][j] for j in range(start, end)] for i in range(seq_len)]
            )
            K_split.append(
                [[K.data[i][j] for j in range(start, end)] for i in range(seq_len)]
            )
            V_split.append(
                [[V.data[i][j] for j in range(start, end)] for i in range(seq_len)]
            )

        outputs = []
        for h in range(self.num_heads):
            Q_h = Tensor(Q_split[h], requires_grad=Q.requires_grad)
            K_h = Tensor(K_split[h], requires_grad=K.requires_grad)
            V_h = Tensor(V_split[h], requires_grad=V.requires_grad)

            scores = Q_h.matmul(K_h.T())
            scores = scores / math.sqrt(d_k)

            if mask:
                for i in range(len(scores.data)):
                    for j in range(len(scores.data[0])):
                        if j > i:
                            scores.data[i][j] = -1e9

            exp_scores = [[math.exp(min(x, 700)) for x in row] for row in scores.data]
            sum_exp = [sum(row) for row in exp_scores]
            attn = [
                [exp_scores[i][j] / sum_exp[i] for j in range(len(exp_scores[0]))]
                for i in range(len(exp_scores))
            ]

            context = Tensor(attn, requires_grad=scores.requires_grad).matmul(V_h)
            outputs.append(context.data)

        concat = []
        for i in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(outputs[h][i])
            concat.append(row)

        concat_tensor = Tensor(concat, requires_grad=query.requires_grad)
        output = self.W_o.forward(concat_tensor)

        return output

    def parameters(self):
        return (
            self.W_q.parameters()
            + self.W_k.parameters()
            + self.W_v.parameters()
            + self.W_o.parameters()
        )


class SimpleTransformerBlock(Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        self.attention = SimpleAttention(d_model, num_heads, dropout)
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
            self.layers.append(
                SimpleTransformerBlock(d_model, num_heads, ff_dim, dropout)
            )

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
        flat_logits = []
        for row in logits.data:
            flat_logits.extend(row)

        exp_logits = [math.exp(min(x, 700)) for x in flat_logits]
        sum_exp = sum(exp_logits)
        log_probs = [math.log(x / sum_exp) for x in exp_logits]

        nll = 0.0
        for target in targets:
            target_idx = int(target) if isinstance(target, float) else target
            if target_idx < len(log_probs):
                nll -= log_probs[target_idx]
        nll /= len(targets) if targets else 1

        return tensor([[nll]], requires_grad=True)

    def parameters(self):
        params = self.token_embedding.parameters()
        params.extend(self.pos_encoding.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx
            if len(idx_cond.data[0]) > self.max_len:
                idx_cond = tensor(
                    [idx_cond.data[0][-self.max_len :]], requires_grad=False
                )

            logits, _ = self.forward(idx_cond)

            logits = logits.data[-1]

            if temperature != 1.0:
                for i in range(len(logits)):
                    logits[i] = logits[i] / temperature

            if top_k is not None:
                sorted_idx = sorted(
                    range(len(logits)), key=lambda i: logits[i], reverse=True
                )
                for i in sorted_idx[top_k:]:
                    logits[i] = float("-inf")

            exp_logits = [math.exp(min(x, 700)) for x in logits]
            sum_exp = sum(exp_logits)
            probs = [x / sum_exp for x in exp_logits]

            next_token = random.choices(range(len(probs)), weights=probs)[0]

            next_token_tensor = tensor([[float(next_token)]], requires_grad=False)
            idx = tensor([idx.data[0] + [float(next_token)]], requires_grad=False)

        return idx

    def eval(self):
        pass

    def train(self):
        pass


class GPT2Small(Module):
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
