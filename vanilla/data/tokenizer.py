from collections import Counter, defaultdict
import re


class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}

    def train(self, text):
        if isinstance(text, str):
            text = text.encode("utf-8")

        text = (
            text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
        )
        tokens = self._get_bytes_tokens(text)

        while len(set(tokens)) > self.vocab_size - 256:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            tokens = self._merge(tokens, best_pair)
            self.merges[best_pair] = True

        self._build_vocab()

    def _get_bytes_tokens(self, text):
        tokens = []
        for char in text:
            tokens.append(ord(char))
        return tokens

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs

    def _merge(self, tokens, pair):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(pair)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _build_vocab(self):
        self.token_to_id = {}
        self.id_to_token = {}

        for i in range(256):
            self.token_to_id[i] = i
            self.id_to_token[i] = i

        idx = 256
        for merge in self.merges:
            self.token_to_id[merge] = idx
            self.id_to_token[idx] = merge
            idx += 1

            if idx >= self.vocab_size:
                break

        self.eos_id = self.token_to_id.get(256, 0)
        self.pad_id = 0

    def encode(self, text):
        if isinstance(text, str):
            tokens = [ord(c) for c in text]
        else:
            tokens = list(text)

        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            pairs_in_vocab = {p: v for p, v in pairs.items() if p in self.token_to_id}
            if not pairs_in_vocab:
                break
            best_pair = max(pairs_in_vocab, key=pairs_in_vocab.get)
            tokens = self._merge(tokens, best_pair)

        return tokens

    def decode(self, ids):
        result = []
        for idx in ids:
            if isinstance(idx, list):
                idx = idx[0] if idx else 0
            idx = int(idx)

            if idx < 256:
                result.append(chr(idx))
            elif idx in self.id_to_token:
                pair = self.id_to_token[idx]
                if isinstance(pair, tuple):
                    for p in pair:
                        if p < 256:
                            result.append(chr(p))
                        elif p in self.id_to_token:
                            sub_pair = self.id_to_token.get(p)
                            if sub_pair:
                                for sp in sub_pair:
                                    result.append(chr(sp) if sp < 256 else "?")
                else:
                    result.append("?")
            else:
                result.append("?")

        return "".join(result)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"VOCAB_SIZE:{self.vocab_size}\n")
            for token, idx in self.token_to_id.items():
                if isinstance(token, tuple):
                    token_str = f"{token[0]},{token[1]}"
                else:
                    token_str = str(token)
                f.write(f"{token_str}:{idx}\n")

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("VOCAB_SIZE:"):
                    self.vocab_size = int(line.split(":")[1])
                elif ":" in line:
                    parts = line.split(":")
                    token_str = parts[0]
                    idx = int(parts[1])

                    if "," in token_str:
                        token = tuple(int(x) for x in token_str.split(","))
                    else:
                        token = int(token_str)

                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token


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
