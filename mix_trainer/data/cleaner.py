"""
MIX😌 训练数据清洗器 - 去重/过滤/标准化/质量检查
含: 精确去重+相似度去重, 名称频率检查, 回复长度统计
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Tuple
from collections import Counter


class DataCleaner:
    def __init__(self, short_name: str = "MIX😌"):
        self.seen_hashes = set()
        self.short_name = short_name

    def clean(self, raw_data: List[Dict], output_dir: str = "data/cleaned") -> List[Dict]:
        print("清洗训练数据...")

        cleaned = self._remove_duplicates(raw_data)
        print(f"  精确去重后: {len(cleaned)} 条 (移除 {len(raw_data) - len(cleaned)} 条重复)")

        cleaned = self._remove_similar(cleaned)
        print(f"  相似度去重后: {len(cleaned)} 条")

        cleaned = self._remove_invalid(cleaned)
        print(f"  有效性过滤后: {len(cleaned)} 条")

        cleaned = self._truncate_long_replies(cleaned, max_len=50)
        print(f"  长回复截断后: {len(cleaned)} 条")

        cleaned = self._normalize_content(cleaned)
        print(f"  内容标准化完成")

        self._quality_report(cleaned)

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

    def _remove_similar(self, data: List[Dict], threshold: float = 0.8) -> List[Dict]:
        result = []
        seen_signatures = []

        for item in data:
            messages = item.get("messages", [])
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if not assistant_msg:
                result.append(item)
                continue

            content = assistant_msg.get("content", "").strip()
            signature = self._text_signature(content)

            is_similar = False
            for existing_sig in seen_signatures:
                similarity = self._jaccard_similarity(signature, existing_sig)
                if similarity > threshold:
                    is_similar = True
                    break

            if not is_similar:
                seen_signatures.append(signature)
                result.append(item)

        return result

    def _text_signature(self, text: str) -> set:
        cleaned = re.sub(r'[^\w]', '', text.lower())
        if len(cleaned) == 0:
            return set()
        ngrams = set()
        for i in range(len(cleaned) - 2):
            ngrams.add(cleaned[i:i+3])
        return ngrams

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / max(union, 1)

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
            if len(user_content) > 500 or len(assistant_content) > 500:
                continue

            if self._is_low_quality(user_content) or self._is_low_quality(assistant_content):
                continue

            valid.append(item)

        return valid

    def _truncate_long_replies(self, data: List[Dict], max_len: int = 100) -> List[Dict]:
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if len(content) > max_len:
                        truncated = content[:max_len]
                        last_punct = max(
                            truncated.rfind("。"),
                            truncated.rfind("！"),
                            truncated.rfind("？"),
                            truncated.rfind("."),
                            truncated.rfind("!"),
                            truncated.rfind("?"),
                        )
                        if last_punct > max_len * 0.5:
                            truncated = truncated[:last_punct + 1]
                        msg["content"] = truncated
        return data

    def _is_low_quality(self, text: str) -> bool:
        if re.search(r'(.)\1{10,}', text):
            return True
        words = text.split()
        if len(words) > 10:
            unique = set(words)
            if len(unique) / len(words) < 0.3:
                return True
        return False

    def _quality_report(self, data: List[Dict]):
        if not data:
            return

        reply_lengths = []
        name_count = 0
        cat_counts = Counter()

        for item in data:
            messages = item.get("messages", [])
            cat = item.get("category", "unknown")
            cat_counts[cat] += 1

            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    reply_lengths.append(len(content))
                    if self.short_name in content:
                        name_count += 1

        total = len(data)
        name_rate = name_count / max(total, 1) * 100
        avg_len = sum(reply_lengths) / max(len(reply_lengths), 1)

        print(f"\n  📊 数据质量报告:")
        print(f"    总条数: {total}")
        print(f"    回复平均长度: {avg_len:.0f}字")
        print(f"    名称'{self.short_name}'出现率: {name_rate:.1f}%")
        for cat, count in cat_counts.items():
            print(f"    {cat}: {count}条")

        if name_rate > 60:
            print(f"  ⚠️ 名称出现率过高({name_rate:.0f}%)! 数据可能过于雷同，模型会只会自我介绍")
        if avg_len > 50:
            print(f"  ⚠️ 回复平均长度过长({avg_len:.0f}字)! 建议控制在30字以内")

    def _normalize_content(self, data: List[Dict]) -> List[Dict]:
        for item in data:
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                content = content.replace("\r\n", "\n")
                content = re.sub(r'\n{3,}', '\n\n', content)
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
