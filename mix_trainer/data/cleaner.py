"""
MIX😌 训练数据清洗器 - 去重/过滤/标准化
"""

import os
import json
import hashlib
from typing import List, Dict


class DataCleaner:
    def __init__(self):
        self.seen_hashes = set()

    def clean(self, raw_data: List[Dict], output_dir: str = "data/cleaned") -> List[Dict]:
        print("清洗训练数据...")

        cleaned = self._remove_duplicates(raw_data)
        print(f"  去重后: {len(cleaned)} 条 (移除 {len(raw_data) - len(cleaned)} 条重复)")

        cleaned = self._remove_invalid(cleaned)
        print(f"  有效性过滤后: {len(cleaned)} 条")

        cleaned = self._normalize_content(cleaned)
        print(f"  内容标准化完成")

        self._save_cleaned_data(cleaned, output_dir)

        print(f"数据清洗完成，最终合格数据: {len(cleaned)} 条")
        return cleaned

    def _remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        result = []
        current_hashes = set()

        for item in data:
            item_hash = item.get("hash", "")
            if not item_hash:
                content = json.dumps(item.get("messages", []), ensure_ascii=False)
                item_hash = hashlib.md5(content.encode()).hexdigest()
                item["hash"] = item_hash

            if item_hash in self.seen_hashes or item_hash in current_hashes:
                continue
            current_hashes.add(item_hash)
            self.seen_hashes.add(item_hash)
            result.append(item)

        return result

    def _remove_invalid(self, data: List[Dict]) -> List[Dict]:
        valid = []
        for item in data:
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue

            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)

            if not user_msg or not assistant_msg:
                continue

            user_content = user_msg.get("content", "").strip()
            assistant_content = assistant_msg.get("content", "").strip()

            if len(user_content) < 2 or len(assistant_content) < 2:
                continue
            if len(user_content) > 2000 or len(assistant_content) > 2000:
                continue

            if self._is_low_quality(user_content) or self._is_low_quality(assistant_content):
                continue

            valid.append(item)

        return valid

    def _is_low_quality(self, text: str) -> bool:
        import re
        if re.search(r'(.)\1{10,}', text):
            return True
        words = text.split()
        if len(words) > 10:
            unique = set(words)
            if len(unique) / len(words) < 0.3:
                return True
        return False

    def _normalize_content(self, data: List[Dict]) -> List[Dict]:
        for item in data:
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                content = content.replace("\r\n", "\n")
                content = content.replace("\n{3,}", "\n\n")
                content = content.strip()
                msg["content"] = content
        return data

    def _save_cleaned_data(self, data: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"cleaned_{hashlib.md5(str(len(data)).encode()).hexdigest()[:8]}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  清洗数据已保存: {filename}")

    @staticmethod
    def load_cleaned_data(data_dir: str) -> List[Dict]:
        all_items = []
        if not os.path.exists(data_dir):
            return all_items

        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_items.extend(data)
                except Exception:
                    continue

        return all_items
