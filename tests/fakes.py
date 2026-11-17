"""Small deterministic stand-ins for a Hugging Face tokenizer (no downloads)."""

import re

# Mirrors the Qwen2.5 behaviour that matters for SPARE: " ки" -> [" к", "и"].
_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>| к|и|\n| ?[A-Za-z0-9']+| ?[^\sA-Za-z0-9]| ")


class FakeTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    padding_side = "right"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "-": 2, "+": 3, "<|im_start|>": 4, "<|im_end|>": 5}

    @property
    def pad_token_id(self):
        return 0

    @property
    def unk_token_id(self):
        return 1

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def tokenize(self, text):
        tokens = _TOKEN_RE.findall(text)
        assert "".join(tokens) == text, "fake tokenizer must be lossless"
        return tokens

    def encode(self, text, add_special_tokens=True):
        return [self._id(t) for t in self.tokenize(text)]

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self.encode(text)}

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def apply_chat_template(self, messages, tokenize=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return self.encode(text) if tokenize else text
