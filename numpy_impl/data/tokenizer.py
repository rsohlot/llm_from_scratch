from collections import Counter


class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def train(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_idx[c] for c in text]

    def decode(self, ids):
        return "".join([self.idx_to_char[i] for i in ids])

    def save(self, path):
        with open(path, "w") as f:
            f.write(str(self.vocab_size) + "\n")
            for char, idx in self.char_to_idx.items():
                f.write(f"{repr(char)}:{idx}\n")

    def load(self, path):
        with open(path, "r") as f:
            self.vocab_size = int(f.readline().strip())
            for line in f:
                parts = line.strip().split(":")
                char = eval(parts[0])
                idx = int(parts[1])
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
