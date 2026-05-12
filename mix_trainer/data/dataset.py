"""
MIX😌 训练数据集 - 自监督预训练格式
Next Token Prediction: 给定前N个token，预测第N+1个token
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Optional


class ConversationDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        all_items = self._load_all_data(data_dir)
        print(f"加载训练数据: {len(all_items)} 条对话")

        all_texts = []
        for item in all_items:
            user_msg = ""
            assistant_msg = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_msg = msg.get("content", "")
            if user_msg and assistant_msg:
                all_texts.append(user_msg)
                all_texts.append(assistant_msg)

        tokenizer.build_vocab(all_texts)

        for item in all_items:
            user_msg = ""
            assistant_msg = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_msg = msg.get("content", "")
            if user_msg and assistant_msg:
                ids = tokenizer.encode_conversation(
                    user_msg, assistant_msg, block_size + 1
                )
                if len(ids) >= 4:
                    self.examples.append(ids)

        self._flatten_and_chunk()

        print(f"构建训练样本: {len(self.examples)} 条 (block_size={block_size})")

    def _load_all_data(self, data_dir: str) -> List[dict]:
        all_items = []
        if not os.path.exists(data_dir):
            print(f"数据目录不存在: {data_dir}")
            return all_items

        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_items.extend(data)
                except Exception as e:
                    print(f"跳过文件 {filename}: {e}")
                    continue

        return all_items

    def _flatten_and_chunk(self):
        all_ids = []
        for ids in self.examples:
            all_ids.extend(ids)
            all_ids.append(self.tokenizer.SPECIAL_TOKENS.get("<SEP>", 4))

        chunked = []
        for i in range(0, len(all_ids) - self.block_size, self.block_size):
            chunk = all_ids[i : i + self.block_size + 1]
            if len(chunk) == self.block_size + 1:
                chunked.append(chunk)

        if chunked:
            self.examples = chunked

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        x = torch.tensor(item[:-1], dtype=torch.long)
        y = torch.tensor(item[1:], dtype=torch.long)
        return x, y
