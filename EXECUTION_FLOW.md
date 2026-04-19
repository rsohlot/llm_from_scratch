# LLM from Scratch - Code Execution Flow

This document explains the high-level architecture and execution flow of building an LLM from scratch.

---

## Overview: What Are We Building?

We're building a **Transformer-based Language Model** (like GPT-2) from scratch using only NumPy. The model learns to predict the next word given previous words.

```
Input: "the cat sat" → Output: P("on"), P("mat"), P("the"), ...
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           main_numpy.py                                 │
│                           (Entry Point)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Tokenizer                                                       │
│  data/tokenizer.py                                                       │
│                                                                         │
│  - Train on sample text                                                 │
│  - Build vocabulary (unique characters)                                │
│  - Encode: text → numbers                                              │
│  - Decode: numbers → text                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Model Creation                                                 │
│  model/transformer_numpy.py                                             │
│                                                                         │
│  Creates Transformer with:                                              │
│  - Token Embedding layer                                                │
│  - Positional Encoding                                                  │
│  - N Transformer Blocks (attention + FFN)                              │
│  - Final LayerNorm                                                      │
│  - LM Head (linear projection to vocab)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Optimizer Setup                                                │
│  core/optim_numpy.py                                                    │
│                                                                         │
│  - Adam optimizer                                                       │
│  - Stores all model parameters                                          │
│  - Handles weight updates                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Training Loop (Epochs)                                         │
│                                                                         │
│  For each epoch:                                                        │
│    For each batch:                                                      │
│      1. Get random sequence from data                                   │
│      2. Forward pass                                                    │
│      3. Compute loss                                                    │
│      4. Backward pass (gradients)                                       │
│      5. Optimizer step (update weights)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Forward Pass (Detailed)                                        │
│                                                                         │
│  Input: "the cat sat" (token IDs: [0, 1, 2])                          │
│                                                                         │
│  5a. TOKEN EMBEDDING                                                    │
│      core/nn_numpy.py → Embedding class                                │
│      - Converts token IDs to dense vectors                             │
│      - Shape: (seq_len) → (seq_len, d_model)                          │
│                                                                         │
│  5b. POSITIONAL ENCODING                                                │
│      core/nn_numpy.py → PositionalEncoding                            │
│      - Adds position information to embeddings                         │
│      - Uses sinusoidal functions                                        │
│                                                                         │
│  5c. TRANSFORMER BLOCKS (repeated N times)                             │
│                                                                         │
│      ┌──────────────────────────────────────────┐                      │
│      │  5c.1 ATTENTION BLOCK                    │                      │
│      │  core/nn_numpy.py → MultiHeadAttention  │                      │
│      │                                           │                      │
│      │  Input: x (batch, seq, d_model)          │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Linear → Q, K, V projections            │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Reshape for multi-head                  │                      │
│      │  (batch, seq, heads, d_k)                │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Attention Scores = Q @ K.T / sqrt(d_k)  │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Softmax → Attention Weights             │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Output = Weights @ V                    │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Linear → Final projection               │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Residual connection: x + attn_output    │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  LayerNorm                                │                      │
│      │  Output: (batch, seq, d_model)           │                      │
│      └──────────────────────────────────────────┘                      │
│                      │                                                   │
│                      ▼                                                   │
│      ┌──────────────────────────────────────────┐                      │
│      │  5c.2 FEED-FORWARD BLOCK                 │                      │
│      │  core/nn_numpy.py → TransformerBlock    │                      │
│      │                                           │                      │
│      │  Input: x (batch, seq, d_model)          │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  LayerNorm                                │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Linear (expand): d_model → ff_dim       │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  GELU activation                          │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Linear (contract): ff_dim → d_model     │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  Residual: x + ff_output                 │                      │
│      │         │                                 │                      │
│      │         ▼                                 │                      │
│      │  LayerNorm                                │                      │
│      │  Output: (batch, seq, d_model)           │                      │
│      └──────────────────────────────────────────┘                      │
│                                                                         │
│  5d. FINAL LAYER NORM                                                   │
│      - Normalize final output                                           │
│                                                                         │
│  5e. LANGUAGE MODEL HEAD                                                │
│      - Linear: d_model → vocab_size                                     │
│      - Output: (batch, seq, vocab_size)                                │
│      - These are LOGITS (raw scores)                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Loss Computation                                               │
│                                                                         │
│  Input: logits (raw scores), target (actual next tokens)              │
│                                                                         │
│  Process:                                                               │
│  1. Softmax on logits → probabilities                                  │
│  2. Get probability of correct token                                   │
│  3. Loss = -log(probability)                                           │
│                                                                         │
│  Example:                                                               │
│  - Target: "on" (token ID 3)                                          │
│  - P("on") = 0.15                                                      │
│  - Loss = -log(0.15) = 1.90                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Backward Pass (Training Only)                                 │
│                                                                         │
│  Process:                                                               │
│  1. loss.backward() called                                             │
│  2. Gradients computed for all parameters                              │
│  3. Stored in param.grad                                               │
└───────────────────────��─────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 8: Optimizer Step                                                 │
│  core/optim_numpy.py → Adam                                            │
│                                                                         │
│  For each parameter:                                                   │
│  1. Get gradient from param.grad                                       │
│  2. Update momentum (m) and velocity (v)                               │
│  3. Compute bias-corrected estimates                                   │
│  4. Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 9: Text Generation (Inference)                                   │
│                                                                         │
│  Process:                                                               │
│  1. Start with input: "In the beginning"                               │
│  2. Forward pass → get probabilities for next token                   │
│  3. Sample next token (or greedy/top-k)                                │
│  4. Append to input                                                     │
│  5. Repeat until max_new_tokens                                        │
│                                                                         │
│  This is AUTOREGRESSIVE generation!                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components Explained

### 1. Tokenizer (`data/tokenizer.py`)
- **What**: Converts text to numbers and back
- **How**: Character-level encoding (each unique character = unique ID)
- **Why**: Neural networks need numbers, not text

### 2. Embedding (`core/nn_numpy.py` → Embedding)
- **What**: Learned lookup table (vocab_size × d_model)
- **How**: Each token ID maps to a d_model-dimensional vector
- **Why**: Dense representations capture semantic meaning

### 3. Positional Encoding (`core/nn_numpy.py` → PositionalEncoding)
- **What**: Adds position information to embeddings
- **How**: Sinusoidal functions (sin for even, cos for odd indices)
- **Why**: Without this, model doesn't know word order

### 4. Multi-Head Attention (`core/nn_numpy.py` → MultiHeadAttention)
- **What**: Each token attends to ALL other tokens
- **How**: 
  - Q (Query): "What am I looking for?"
  - K (Key): "What do I contain?"
  - V (Value): "Here's my content"
  - Score = Q × K^T / √d_k
  - Output = softmax(scores) × V
- **Why**: Captures dependencies between any tokens (no distance limit!)

### 5. Feed-Forward Network (`core/nn_numpy.py` → TransformerBlock)
- **What**: Two linear layers with GELU
- **How**: Expand → GELU → Contract
- **Why**: Adds non-linearity and capacity

### 6. LayerNorm (`core/nn_numpy.py` → LayerNorm)
- **What**: Normalizes to mean=0, std=1
- **How**: (x - mean) / std × gamma + beta
- **Why**: Stabilizes training, faster convergence

### 7. Residual Connections (`core/nn_numpy.py` → TransformerBlock)
- **What**: Skip connection: output = input + block_output
- **Why**: Helps gradient flow, allows deeper networks

### 8. Loss Function (`model/transformer_numpy.py` → _compute_loss)
- **What**: Cross-entropy loss
- **How**: Loss = -log(probability of correct token)
- **Why**: Standard for classification (next token prediction)

### 9. Adam Optimizer (`core/optim_numpy.py` → Adam)
- **What**: Adaptive learning rate optimizer
- **How**: Uses momentum (m) and velocity (v) for updates
- **Why**: Faster convergence than vanilla SGD

---

## Data Shapes Through the Network

```
Input Text: "the cat sat"
           │
           ▼
Token IDs: [0, 1, 2]                    Shape: (3,)
           │
           ▼
Embedding: [[0.05, -0.01, ...],         Shape: (3, d_model)
            [0.12, 0.03, ...],
            [-0.02, 0.08, ...]]
           │
           ▼
+ Positional Encoding                        Shape: (3, d_model)
           │
           ▼
Transformer Block × N                        Shape: (3, d_model)
           │
           ▼
Final Norm                                    Shape: (3, d_model)
           │
           ▼
LM Head (linear)                             Shape: (3, vocab_size)
           │
           ▼
Softmax                                       Shape: (3, vocab_size)
           │
           ▼
For position 2 ("sat"):
P(the)=0.16, P(cat)=0.17, P(sat)=0.23, 
P(on)=0.24, P(mat)=0.20
```

---

## Training vs Inference

| Aspect | Training | Inference (Generation) |
|--------|----------|------------------------|
| Input | Fixed sequence | Starts with prompt |
| Output | Loss (for backprop) | Next token probabilities |
| Gradient | Computed | Not needed |
| Weights | Updated | Frozen |
| Process | One pass | Autoregressive loop |

---

## Summary

1. **Tokenize** text → numbers
2. **Embed** tokens + add positions
3. **Process** through transformer blocks:
   - Self-attention (word → context)
   - Feed-forward (process)
   - Residual + LayerNorm
4. **Project** to vocabulary size
5. **Compute** loss (prediction vs actual)
6. **Update** weights via optimizer
7. **Generate** text autoregressively

This is exactly how GPT, BERT, and all modern LLMs work!