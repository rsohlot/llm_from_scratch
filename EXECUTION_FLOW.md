# LLM from Scratch - Code Execution Flow

High-level walkthrough of how a training step flows through the code. Paths below use `numpy_impl/`; the `vanilla/` tree has identical module names and an identical pipeline (just with nested-list tensors and a tiny config).

---

## Overview

A Transformer decoder that predicts the next token given previous tokens.

```
Input: "the cat sat" → Output: P(next token | context)
```

---

## Execution Flow

```
main.py  ── entry point
  │
  ▼
[1] Tokenizer                      (data/tokenizer.py)
    text ──► token IDs
  │
  ▼
[2] Model construction             (model/transformer.py)
    Embedding → PositionalEncoding → N × TransformerBlock → LayerNorm → LM head
  │
  ▼
[3] Optimizer                      (core/optim.py)
    Adam wraps model.parameters()
  │
  ▼
[4] Training loop (epoch × step)
    sample batch → forward → loss → loss.backward() → optimizer.step()
  │
  ▼
[5] Forward pass                   (core/nn.py + model/transformer.py)

    Token IDs [T]
       │
       ▼
    Embedding → [T, d_model]
       │
       ▼
    + Positional encoding → [T, d_model]
       │
       ▼
    ┌── Pre-norm Transformer block (× num_layers) ──┐
    │   h = LayerNorm(x)                             │
    │   x = x + MultiHeadAttention(h, causal mask)   │
    │   x = x + FeedForward(LayerNorm(x))            │
    └────────────────────────────────────────────────┘
       │
       ▼
    Final LayerNorm → [T, d_model]
       │
       ▼
    LM head (Linear) → [T, vocab_size]   ← logits
  │
  ▼
[6] Loss                           (Tensor.cross_entropy)
    Fused log-softmax + negative-log-likelihood for stability.
    Returns a scalar Tensor connected to the graph.
  │
  ▼
[7] Backward pass                  (core/tensor.py)
    loss.backward() walks the graph in reverse topological order,
    invoking each op's _backward closure to accumulate gradients
    into param.grad.
  │
  ▼
[8] Optimizer step                 (core/optim.py)
    Adam: m, v momentum buffers; bias correction; weight update.
  │
  ▼
[9] Generation (inference)         (Transformer.generate)
    Autoregressive: repeatedly run forward on the growing prefix,
    sample the next token from softmax(logits[-1] / T), append.
```

---

## Key components

### 1. Tokenizer — `data/tokenizer.py`
`SimpleTokenizer` does character-level encoding (each unique char → integer ID). A byte-pair `Tokenizer` class is also present.

### 2. Embedding — `core/nn.py :: Embedding`
Learned lookup table shape `[vocab_size, d_model]`. Indexing by token IDs produces a dense `[T, d_model]` representation. The backward pass scatter-adds the upstream gradient into the rows that were looked up.

### 3. Positional encoding — `core/nn.py :: PositionalEncoding`
Fixed sinusoidal positions added to the embedding (not learned, `requires_grad=False`).

### 4. Multi-head self-attention — `core/nn.py :: MultiHeadAttention`
```
Q = x W_q,  K = x W_k,  V = x W_v         # each [T, d_model]
split into num_heads of d_k each
scores = Q @ Kᵀ / √d_k                    # [T, T] per head
scores = masked_fill(scores, causal_mask, -∞)
attn   = softmax(scores)
out    = attn @ V
concat heads, project with W_o
```
Causal mask: upper triangle of the `[T, T]` score matrix is set to `-∞` so a position cannot attend to the future.

### 5. Feed-forward — `core/nn.py :: TransformerBlock.ff`
`Linear(d_model → ff_dim)` → `GELU` → `Linear(ff_dim → d_model)`.

### 6. LayerNorm — `core/nn.py :: LayerNorm`
Normalizes across the last dim, then applies learnable `gamma, beta`. Implemented as a composite of autograd ops (mean/var/sub/pow/div), so the gradient is computed automatically.

### 7. Residual connections + pre-norm
Each block is `x + attn(LN(x))` and `x + ff(LN(x))` (pre-norm). The residual stream keeps gradients flowing through deep stacks.

### 8. Loss — `Tensor.cross_entropy`
Fused log-softmax + NLL. The backward produces `(softmax(logits) - one_hot(target)) / N`, scaled by the upstream gradient — no manual logits-to-softmax-to-log chain that could under/overflow.

### 9. Adam / AdamW — `core/optim.py`
Per-parameter first and second moment buffers, bias-corrected, with an optional decoupled weight-decay (AdamW).

---

## Shapes through the network

```
token IDs            [T]
  → embedding        [T, d_model]
  → + positional     [T, d_model]
  → N × block        [T, d_model]
  → final LayerNorm  [T, d_model]
  → LM head          [T, vocab_size]     (logits)
```

(In `numpy_impl/` the leading batch dim `B` is also supported: `[B, T]` → `[B, T, vocab_size]`. `vanilla/` keeps batch size = 1 implicit to stay compact.)

---

## Training vs inference

| | Training | Generation |
|---|---|---|
| Input | Fixed sequence from corpus | Growing prefix from prompt |
| Output | Scalar loss | Next-token probabilities |
| Gradient | Computed via `backward()` | None (`eval()` mode) |
| Weights | Updated by optimizer | Frozen |
| Dropout | Active (inverted) | Disabled |
| Loop | Batch → step → repeat | Sample → append → repeat |

---

## Summary

1. Tokenize text → integer IDs
2. Embed + add positional encoding
3. Pre-norm Transformer blocks: masked self-attention → feed-forward, with residuals
4. Final LayerNorm, project to vocabulary logits
5. Fused cross-entropy loss, backward pass through the graph
6. Optimizer step (Adam / AdamW)
7. Generate autoregressively by sampling from `softmax(logits / T)` with optional top-k
