# Build Your Own LLM from Scratch

A production-ready Transformer (GPT-style) implementation built from scratch using only NumPy.

## Project Structure

```
llm_from_scratch/
├── core/
│   ├── tensor_numpy.py    # Tensor operations (NumPy-backed)
│   ├── nn_numpy.py        # Neural network layers
│   └── optim_numpy.py     # Optimizers (SGD, Adam, AdamW)
├── data/
│   └── tokenizer.py       # Character-level tokenizer
├── model/
│   └── transformer_numpy.py # Full Transformer model
├── visual_learning/       # Interactive visualizations
├── main_numpy.py          # Training script
├── EXECUTION_FLOW.md      # Code execution flow explanation
└── requirements.txt       # Dependencies (NumPy only!)
```

## Quick Start

```bash
# Install dependencies
pip install numpy

# Run training
python main_numpy.py
```

## Understanding the Code

**Start here:** Read [EXECUTION_FLOW.md](EXECUTION_FLOW.md) for a detailed explanation of:
- How data flows through the network
- What each component does
- Training vs inference

**Interactive visualizations:**
```bash
python visual_learning/transformer_visual.py  # Step-by-step with actual numbers
```

## Architecture (Same as GPT-2)

- **Multi-head Self-Attention**: 8 heads, 256 dim
- **Feed-Forward Network**: GELU activation, 512 hidden dim
- **Layer Normalization**: Pre-norm architecture
- **Positional Encoding**: Sinusoidal
- **Token Embeddings**: Learned embeddings

## Performance

- Uses NumPy with SIMD optimizations (100-1000x faster than pure Python)
- Full gradient computation
- Adam optimizer with weight decay

## To Train on Your Data

1. Replace `SAMPLE_TEXT` in `main_numpy.py` with your dataset
2. Increase `epochs`, `batch_size`, `seq_len`
3. For better results, increase model dimensions:
   - `d_model=512`, `num_heads=16`, `num_layers=6`

## For Production-Scale

For actual GPT-3/4 scale training, you would need:
- **PyTorch/JAX** - for automatic differentiation & GPU acceleration
- **DeepSpeed/Megatron** - for distributed training
- **Gradient checkpointing** - to save memory
- **Mixed precision** - for faster training

This codebase demonstrates the **core algorithm** - all production systems use these same concepts!