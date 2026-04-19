"""Pure-Python LLM - trains a tiny Transformer and samples text."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.optim import Adam
from core.tensor import Tensor
from data.tokenizer import SimpleTokenizer
from model.transformer import Transformer


SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and brown. "
    "the lazy dog sleeps under the old oak tree. "
)


def main():
    random.seed(0)

    print("=" * 60)
    print("Building LLM from Scratch - Pure Python (no dependencies)")
    print("=" * 60)

    print("\n[1/5] Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.train(SAMPLE_TEXT)
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size}")

    print("\n[2/5] Encoding training data...")
    encoded = tokenizer.encode(SAMPLE_TEXT)
    print(f"   Total tokens: {len(encoded)}")

    print("\n[3/5] Creating tiny Transformer model...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=16,
        dropout=0.0,
    )
    params = model.parameters()
    total = sum(
        (len(p.data) * len(p.data[0])) if isinstance(p.data[0], list) else len(p.data)
        for p in params
    )
    print(f"   Parameter tensors: {len(params)}  |  scalars: {total}")

    print("\n[4/5] Setting up optimizer...")
    optimizer = Adam(params, lr=5e-3)

    print("\n[5/5] Training (this is pure Python, so slow)...")
    epochs = 20
    steps_per_epoch = 4
    seq_len = 8

    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            if len(encoded) < seq_len + 1:
                break
            start = random.randint(0, len(encoded) - seq_len - 1)
            input_seq = encoded[start : start + seq_len]
            target_seq = encoded[start + 1 : start + seq_len + 1]

            x = Tensor([input_seq], requires_grad=False)
            optimizer.zero_grad()
            _, loss = model.forward(x, target_seq)
            total_loss += loss.data[0][0]
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / steps_per_epoch
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}"
            )

    print("\n[Generating]")
    start_prompt = "the quick"
    input_ids = tokenizer.encode(start_prompt)
    x = Tensor([input_ids], requires_grad=False)
    result = model.generate(x, max_new_tokens=30, temperature=0.8, top_k=5)
    output_ids = [int(v) for v in result.data[0]]
    print(f"Input:  {start_prompt}")
    print(f"Output: {tokenizer.decode(output_ids)}")


if __name__ == "__main__":
    main()
