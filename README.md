# Build Your Own LLM from Scratch

A GPT-style Transformer built from scratch in two flavors: pure Python (no deps) and NumPy. Both ship a small reverse-mode autograd engine and actually train — loss decreases end-to-end.

## Project Structure

```
llm_from_scratch/
├── vanilla/                   # Pure-Python implementation (no dependencies)
│   ├── core/
│   │   ├── tensor.py          # Autograd-enabled Tensor (nested-list backend)
│   │   ├── nn.py              # Linear, LayerNorm, GELU, Dropout, Embedding, MHA, ...
│   │   └── optim.py           # SGD, Adam, AdamW
│   ├── data/
│   │   └── tokenizer.py       # Character-level + BPE tokenizers
│   ├── model/
│   │   └── transformer.py     # Tiny Transformer language model
│   └── main.py                # Train + sample entry point
├── numpy_impl/                # NumPy-backed implementation (fast)
│   ├── core/{tensor,nn,optim}.py
│   ├── data/tokenizer.py
│   ├── model/transformer.py
│   └── main.py
├── EXECUTION_FLOW.md          # Architectural walkthrough
├── LICENSE                    # MIT
└── requirements.txt           # numpy (for numpy_impl only)
```

## Quick Start

```bash
# Pure Python — no dependencies. Tiny model, trains in seconds.
python vanilla/main.py

# NumPy-backed — larger model, faster training.
pip install -r requirements.txt
python numpy_impl/main.py
```

Both print decreasing training loss and sample text at the end.

## What's implemented

- **Autograd**: reverse-mode with topological sort; every op records parents and a `_backward` closure. Gradients flow through matmul, broadcast add/mul/div, pow, exp, log, softmax, GELU, fused log-softmax + cross-entropy, embedding lookup, column slicing + concat (for multi-head attention), and masked-fill (for causal masking).
- **Transformer**: pre-norm blocks with multi-head self-attention, causal mask, feed-forward with GELU, sinusoidal positional encoding, inverted-dropout Dropout, weight-tied-optional LM head.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

## Default configs

| | vanilla | numpy_impl |
|---|---|---|
| d_model | 16 | 64 |
| heads | 2 | 4 |
| layers | 1 | 2 |
| seq_len | 8 | 32 |
| epochs | 20 | 50 |
| params | ~3K | ~70K |

The vanilla config is kept tiny because pure-Python matmul is ~1000× slower than NumPy. It exists to prove the autograd mechanics, not to produce coherent text.

## Scaling up

To actually get readable generations you need more data and a bigger model. In `numpy_impl/main.py`, bump `d_model`, `num_layers`, `epochs`, and feed a larger text corpus. For anything real (GPT-2+), move to PyTorch/JAX — this repo is a learning exercise, not a training framework.

## Further reading

See [EXECUTION_FLOW.md](EXECUTION_FLOW.md) for the data flow through the network and what each layer does.
