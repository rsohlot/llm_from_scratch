from core.tensor import Tensor, tensor, zeros, ones, randn
import math


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
        self.weight = Tensor(randn(in_features, out_features).data, requires_grad=True)
        self.weight.data = [[x * scale for x in row] for row in self.weight.data]

        if bias:
            self.bias = Tensor(zeros(1, out_features).data, requires_grad=True)
        else:
            self.bias = None

        self._output = None

    def forward(self, x):
        result = x.matmul(self.weight)
        if self.bias:
            seq_len = len(result.data)
            bias_data = [self.bias.data[0] for _ in range(seq_len)]
            result = result + Tensor(bias_data)
        self._output = result
        return result

    def parameters(self):
        params = [self.weight]
        if self.bias:
            params.append(self.bias)
        return params


class LayerNorm(Module):
    def __init__(self, features, eps=1e-5):
        self.features = features
        self.eps = eps
        self.gamma = Tensor(ones(1, features).data, requires_grad=True)
        self.beta = Tensor(zeros(1, features).data, requires_grad=True)

    def forward(self, x):
        mean = sum(x.data[0]) / len(x.data[0])
        var = sum((x.data[0][i] - mean) ** 2 for i in range(len(x.data[0]))) / len(
            x.data[0]
        )
        std = math.sqrt(var + self.eps)

        result = [
            [
                (x.data[0][i] - mean) / std * self.gamma.data[0][i]
                + self.beta.data[0][i]
                for i in range(len(x.data[0]))
            ]
        ]

        return Tensor(result, requires_grad=x.requires_grad)

    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p

    def forward(self, x):
        return x


class GELU(Module):
    def forward(self, x):
        result = []
        for row in x.data:
            new_row = []
            for val in row:
                new_row.append(
                    0.5
                    * val
                    * (
                        1
                        + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val**3))
                    )
                )
            result.append(new_row)
        return Tensor(result, requires_grad=x.requires_grad)


class Softmax(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        max_vals = [max(row) for row in x.data]
        exp_data = [
            [math.exp(val - max_vals[i]) for val in row] for i, row in enumerate(x.data)
        ]
        sum_exp = [sum(row) for row in exp_data]
        result = [[val / sum_exp[i] for val in row] for i, row in enumerate(exp_data)]
        return Tensor(result, requires_grad=x.requires_grad)


class CrossEntropyLoss(Module):
    def __init__(self):
        pass

    def forward(self, logits, targets):
        if len(logits.shape) == 2:
            logits = logits.view(1, logits.shape[0], logits.shape[1])

        exp_logits = [[math.exp(x) for x in row] for row in logits.data[0]]
        sum_exp = [sum(row) for row in exp_logits]
        log_probs = [
            [x - math.log(s) for x in row] for row, s in zip(exp_logits, sum_exp)
        ]

        nll = 0.0
        for i, target in enumerate(targets):
            target_idx = int(target) if isinstance(target, float) else target
            nll -= log_probs[i][target_idx]
        nll /= len(targets)

        return Tensor([[nll]], requires_grad=True)


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
            [
                [(hash((i, j)) % 1000) / 1000.0 * scale for j in range(embedding_dim)]
                for i in range(num_embeddings)
            ],
            requires_grad=True,
        )

    def forward(self, x):
        if isinstance(x.data[0], list):
            indices = [int(i) for i in x.data[0]]
        else:
            indices = [int(i) for i in x.data]

        result = []
        for idx in indices:
            if idx < self.num_embeddings:
                result.append(self.weight.data[idx])
            else:
                result.append([0.0] * self.embedding_dim)

        return Tensor(result, requires_grad=x.requires_grad)

    def parameters(self):
        return [self.weight]


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model

        pe = [[0.0 for _ in range(d_model)] for _ in range(max_len)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pe[pos][i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))

        self.pe = Tensor(pe, requires_grad=False)

    def forward(self, x):
        if isinstance(x.data[0], list):
            seq_len = len(x.data[0])
        else:
            seq_len = 1

        pe_slice = [self.pe.data[i][: self.d_model] for i in range(seq_len)]
        pe_tensor = Tensor(pe_slice, requires_grad=False)

        return x + pe_tensor


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

        self.softmax = Softmax()
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = 1
        seq_len = len(query.data[0]) if isinstance(query.data[0], list) else 1

        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        Q = self._reshape(Q)
        K = self._reshape(K)
        V = self._reshape(V)

        scores = Q.matmul(K.T())
        scores = scores / math.sqrt(self.d_k)

        if mask:
            for i in range(len(scores.data)):
                for j in range(len(scores.data[0])):
                    if j > i:
                        scores.data[i][j] = -1e9

        attn_weights = self.softmax.forward(scores)
        attn_weights = self.dropout.forward(attn_weights)

        context = attn_weights.matmul(V)
        context = self._reshape_back(context)

        output = self.W_o.forward(context)
        return output

    def _reshape(self, x):
        seq_len = len(x.data[0]) if isinstance(x.data[0], list) else 1
        result = []
        for i in range(seq_len):
            row = []
            for head in range(self.num_heads):
                for j in range(self.d_k):
                    row.append(x.data[i][head * self.d_k + j])
            result.append(row)
        return Tensor(result, requires_grad=x.requires_grad)

    def _reshape_back(self, x):
        seq_len = len(x.data) if isinstance(x.data[0], list) else 1
        result = []
        for i in range(seq_len):
            row = []
            for head in range(self.num_heads):
                for j in range(self.d_k):
                    row.append(x.data[i][head * self.d_k + j])
            result.append(row)
        return Tensor(result, requires_grad=x.requires_grad)

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
