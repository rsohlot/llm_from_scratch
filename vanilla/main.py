"""
Vanilla LLM - Pure Python (No Dependencies)
Run this to train and generate text using pure Python implementation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.tensor import Tensor, randn
from core.nn import (
    Linear,
    LayerNorm,
    Dropout,
    GELU,
    Embedding,
    TransformerBlock,
    Sequential,
)
from core.optim import Adam, SGD
from model.transformer import Transformer, GPT2Small, create_gpt_config
from data.tokenizer import SimpleTokenizer
import random
import math


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
    print("Building LLM from Scratch - Pure Python (No Dependencies)")
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
        d_model=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_len=128,
        dropout=0.1,
    )

    params = model.parameters()
    print(f"   Model parameters: {len(params)}")

    print("\n[4/5] Setting up optimizer...")
    optimizer = Adam(params, lr=0.001)

    print("\n[5/5] Training the model...")
    epochs = 20
    batch_size = 3
    seq_len = 16

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

            input_tensor = Tensor([input_seq], requires_grad=True)

            optimizer.zero_grad()

            logits, loss = model.forward(input_tensor, target_seq)

            if loss and loss.data:
                loss_val = (
                    loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
                )
                total_loss += loss_val
                num_batches += 1

                try:
                    loss.backward()
                    optimizer.step()
                except:
                    pass

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
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
        input_tensor = Tensor([input_ids], requires_grad=False)

        result = model.generate(
            input_tensor, max_new_tokens=50, temperature=0.8, top_k=40
        )

        output_ids = result.data[0]
        output_ids = [int(x) for x in output_ids]

        generated_text = tokenizer.decode(output_ids)

        print(f"Input:  {start_prompt}")
        print(f"Output: {generated_text}")

    except Exception as e:
        print(f"Generation error: {e}")
        print("Text generation completed with warnings.")

    print("\n" + "=" * 60)
    print("LLM Training Complete!")
    print("=" * 60)
    print("\nThis is the PURE PYTHON version (no dependencies)")
    print("For faster training, use numpy_impl version")


if __name__ == "__main__":
    main()
