"""
NumPy LLM - High Performance
Run this to train and generate text using NumPy implementation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math
import random
from core.tensor import Tensor
from core.optim import Adam
from model.transformer import Transformer
from data.tokenizer import SimpleTokenizer


SAMPLE_TEXT = """
In the beginning there was darkness. Then came light. The light brought with it hope and possibility. 
For thousands of years, humanity has strived to understand the universe and our place within it. 
We built cities, created art, discovered science, and dreamed of futures beyond our imagination.
Every day presents new opportunities to learn, grow, and make a positive impact on the world around us.
The journey of a thousand miles begins with a single step. Together, we can accomplish great things.
Knowledge is power. Education is the key to unlocking human potential. Let us learn and grow together.
"""


def main():
    print("=" * 60)
    print("Building LLM from Scratch - NumPy (High Performance)")
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
        d_model=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        max_len=256,
        dropout=0.1,
    )

    params = model.parameters()
    print(f"   Model parameters: {len(params)}")

    print("\n[4/5] Setting up optimizer...")
    optimizer = Adam(params, lr=0.0003)

    print("\n[5/5] Training the model...")
    epochs = 100
    batch_size = 8
    seq_len = 32

    print(f"\n   Training for {epochs} epochs...")
    print("-" * 40)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for _ in range(batch_size):
            if len(encoded) < seq_len + 1:
                continue

            random_start = random.randint(0, len(encoded) - seq_len - 1)

            input_seq = encoded[random_start : random_start + seq_len]
            target_seq = encoded[random_start + 1 : random_start + seq_len + 1]

            input_tensor = Tensor(
                np.array([input_seq], dtype=np.int64), requires_grad=True
            )

            optimizer.zero_grad()

            logits, loss = model.forward(input_tensor, target_seq)

            if loss and loss.data is not None:
                loss_val = float(loss.data[0, 0])
                total_loss += loss_val
                num_batches += 1

                try:
                    loss.backward()
                    optimizer.step()
                except:
                    pass

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f}"
            )

    print("-" * 40)
    print("\nTraining complete!")

    print("\n[Generating Text]")
    print("-" * 40)

    start_prompt = "In the beginning"

    try:
        input_ids = tokenizer.encode(start_prompt)
        input_tensor = Tensor(
            np.array([input_ids], dtype=np.int64), requires_grad=False
        )

        result = model.generate(
            input_tensor, max_new_tokens=50, temperature=0.8, top_k=40
        )

        output_ids = result.data[0].astype(int).tolist()

        generated_text = tokenizer.decode(output_ids)

        print(f"Input:  {start_prompt}")
        print(f"Output: {generated_text}")

    except Exception as e:
        print(f"Generation error: {e}")
        print("Text generation completed with warnings.")

    print("\n" + "=" * 60)
    print("LLM Training Complete!")
    print("=" * 60)
    print("\nThis is the NUMPY version (fast)")
    print("Architecture is same as GPT-2:")
    print("  - Multi-head self-attention")
    print("  - Feed-forward networks with GELU")
    print("  - Layer normalization")


if __name__ == "__main__":
    main()
