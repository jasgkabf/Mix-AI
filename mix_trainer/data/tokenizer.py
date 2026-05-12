"""
MIX😌 BPE分词器 - Byte Pair Encoding
支持中英文混合文本，自监督学习合并规则
"""

import json
import os
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


class BPETokenizer:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<SEP>": 4,
        "<USER>": 5,
        "<ASSISTANT>": 6,
    }

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_order: List[Tuple[str, str]] = []

        for token, idx in self.SPECIAL_TOKENS.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self._base_vocab_built = False

    def _get_byte_repr(self, text: str) -> List[str]:
        tokens = []
        for ch in text:
            byte_repr = ch.encode("utf-8")
            if len(byte_repr) == 1 and 32 <= byte_repr[0] < 127:
                tokens.append(ch)
            else:
                for b in byte_repr:
                    tokens.append(f"<0x{b:02X}>")
        return tokens

    def build_vocab(self, texts: List[str], num_merges: Optional[int] = None):
        if num_merges is None:
            num_merges = self.vocab_size - len(self.SPECIAL_TOKENS)

        word_freqs: Dict[tuple, int] = defaultdict(int)

        for text in texts:
            words = text.strip().split()
            for word in words:
                chars = tuple(self._get_byte_repr(word))
                if chars:
                    word_freqs[chars] += 1

        for chars_tuple in word_freqs:
            for ch in chars_tuple:
                if ch not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[ch] = idx
                    self.id_to_token[idx] = ch

        self._base_vocab_built = True

        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]

            if new_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[new_token] = idx
                self.id_to_token[idx] = new_token

            self.merges[best_pair] = new_token
            self.merge_order.append(best_pair)

            new_word_freqs: Dict[tuple, int] = defaultdict(int)
            for word, freq in word_freqs.items():
                new_word = []
                j = 0
                while j < len(word):
                    if (
                        j < len(word) - 1
                        and word[j] == best_pair[0]
                        and word[j + 1] == best_pair[1]
                    ):
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            if (i + 1) % 500 == 0:
                print(f"  BPE合并进度: {i+1}/{num_merges} | 词表: {len(self.token_to_id)}")

        self.vocab_size = len(self.token_to_id)
        print(f"BPE分词器构建完成: 词表大小 {self.vocab_size}, 合并规则 {len(self.merge_order)} 条")
        return self.vocab_size

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        for pair in self.merge_order:
            if len(tokens) < 2:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    merged = pair[0] + pair[1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        words = text.strip().split()
        ids = []

        for word in words:
            chars = self._get_byte_repr(word)
            tokens = self._apply_merges(chars)
            for t in tokens:
                ids.append(self.token_to_id.get(t, self.SPECIAL_TOKENS["<UNK>"]))

        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id_val in ids:
            if id_val in self.id_to_token:
                token = self.id_to_token[id_val]
                if token in self.SPECIAL_TOKENS:
                    if token == "<SEP>":
                        tokens.append(" ")
                    continue
                tokens.append(token)
            else:
                tokens.append("")

        text = "".join(tokens)

        result = []
        i = 0
        current_bytes = bytearray()
        while i < len(text):
            if text[i:i+4].startswith("<0x") and i + 5 < len(text) and text[i+5] == ">":
                hex_str = text[i+3:i+5]
                try:
                    current_bytes.append(int(hex_str, 16))
                except ValueError:
                    if current_bytes:
                        try:
                            result.append(current_bytes.decode("utf-8"))
                        except UnicodeDecodeError:
                            result.append(current_bytes.decode("utf-8", errors="replace"))
                        current_bytes = bytearray()
                    result.append(text[i])
                i += 6
            else:
                if current_bytes:
                    try:
                        result.append(current_bytes.decode("utf-8"))
                    except UnicodeDecodeError:
                        result.append(current_bytes.decode("utf-8", errors="replace"))
                    current_bytes = bytearray()
                result.append(text[i])
                i += 1

        if current_bytes:
            try:
                result.append(current_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                result.append(current_bytes.decode("utf-8", errors="replace"))

        return "".join(result)

    def encode_conversation(self, user_msg: str, assistant_msg: str, max_len: int = 512) -> List[int]:
        text = f"<BOS><USER>{user_msg}<SEP><ASSISTANT>{assistant_msg}<EOS>"
        ids = self.encode(text)
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": [[p[0], p[1]] for p in self.merge_order],
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"分词器已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tokenizer.merge_order = [(p[0], p[1]) for p in data["merges"]]
        tokenizer.merges = {(p[0], p[1]): p[0] + p[1] for p in tokenizer.merge_order}
        tokenizer.SPECIAL_TOKENS = data.get("special_tokens", cls.SPECIAL_TOKENS)
        tokenizer._base_vocab_built = True

        print(f"分词器已加载: {path} (词表:{tokenizer.vocab_size})")
        return tokenizer
