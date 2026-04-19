"""NumPy LLM - trains a small Transformer and samples text."""

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.optim import Adam
from core.tensor import Tensor
from data.tokenizer import SimpleTokenizer
from model.transformer import Transformer


SAMPLE_TEXT = """
In the beginning there was darkness. Then came light. The light brought with it hope and possibility.
For thousands of years, humanity has strived to understand the universe and our place within it.
We built cities, created art, discovered science, and dreamed of futures beyond our imagination.
Every day presents new opportunities to learn, grow, and make a positive impact on the world around us.
The journey of a thousand miles begins with a single step. Together, we can accomplish great things.
Knowledge is power. Education is the key to unlocking human potential. Let us learn and grow together.
"""


def main():
    np.random.seed(0)
    random.seed(0)

    print("=" * 60)
    print("Building LLM from Scratch - NumPy")
    print("=" * 60)

    print("\n[1/5] Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.train(SAMPLE_TEXT)
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size}")

    print("\n[2/5] Encoding training data...")
    encoded = tokenizer.encode(SAMPLE_TEXT)
    print(f"   Total tokens: {len(encoded)}")

    print("\n[3/5] Creating Transformer model...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        max_len=64,
        dropout=0.0,
    )
    params = model.parameters()
    total = sum(p.data.size for p in params)
    print(f"   Parameter tensors: {len(params)}  |  scalars: {total}")

    print("\n[4/5] Setting up optimizer...")
    optimizer = Adam(params, lr=3e-3)

    print("\n[5/5] Training...")
    epochs = 50
    steps_per_epoch = 8
    seq_len = 32

    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            if len(encoded) < seq_len + 1:
                break
            start = random.randint(0, len(encoded) - seq_len - 1)
            input_seq = encoded[start : start + seq_len]
            target_seq = encoded[start + 1 : start + seq_len + 1]

            x = Tensor(np.array([input_seq], dtype=np.int64))
            optimizer.zero_grad()
            _, loss = model.forward(x, target_seq)
            total_loss += float(loss.data)
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / steps_per_epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}"
            )

    print("\n[Generating]")
    start_prompt = "In the beginning"
    input_ids = tokenizer.encode(start_prompt)
    x = Tensor(np.array([input_ids], dtype=np.int64), requires_grad=False)
    result = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=10)
    output_ids = result.data[0].astype(int).tolist()
    print(f"Input:  {start_prompt}")
    print(f"Output: {tokenizer.decode(output_ids)}")


if __name__ == "__main__":
    main()
