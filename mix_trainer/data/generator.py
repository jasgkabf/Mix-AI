"""
MIX😌 训练数据生成器 - 调用第三方大模型API批量生成高质量训练语料
真实AI训练需要大量高质量数据，每轮生成500+条标准对话
"""

import os
import json
import time
import hashlib
import uuid
import requests
from typing import List, Dict, Optional


DATA_SPEC = {
    "greeting": {
        "label": "问候类",
        "per_call": 30,
        "total_target": 150,
        "sub_categories": [
            "基础问候(你好/嗨/哈喽)",
            "时间问候(早上好/中午好/晚上好/晚安)",
            "场景问候(在吗/最近怎么样/好久不见)",
            "情绪问候(今天心情不错/好累啊/开心)",
            "节日问候(新年好/生日快乐/圣诞快乐)",
        ],
    },
    "identity": {
        "label": "身份认知类",
        "per_call": 30,
        "total_target": 150,
        "sub_categories": [
            "名称记忆(你叫什么/你的名字/你叫啥)",
            "功能介绍(你能做什么/你会什么/你擅长什么)",
            "身份对比(你和ChatGPT区别/你和其他AI不同)",
            "自我介绍(介绍一下你自己/说说你的特点)",
            "能力边界(你不会什么/你的局限是什么)",
        ],
    },
    "daily_chat": {
        "label": "日常聊天类",
        "per_call": 30,
        "total_target": 200,
        "sub_categories": [
            "天气季节(今天好热/最近总下雨/冬天好冷)",
            "美食饮食(中午吃什么/推荐个菜/你会做饭吗)",
            "心情情感(我好开心/有点难过/压力好大)",
            "兴趣爱好(喜欢什么电影/推荐首歌/周末干嘛)",
            "生活日常(好无聊/刚起床/加班好累)",
            "社交人际(朋友过生日送什么/怎么认识新朋友)",
            "宠物动物(想养猫/狗狗好可爱/金鱼怎么养)",
        ],
    },
    "general_qa": {
        "label": "通用问答类",
        "per_call": 30,
        "total_target": 200,
        "sub_categories": [
            "科学知识(光速是多少/地球多大/水为什么沸腾)",
            "生活常识(怎么去油渍/感冒怎么办/手机省电)",
            "学习方法(怎么学英语/如何提高记忆力/读书方法)",
            "工作职场(如何提高效率/面试技巧/时间管理)",
            "科技数码(什么是AI/5G有什么用/区块链解释)",
            "健康养生(每天喝多少水/怎么改善睡眠/运动建议)",
            "文化历史(春节由来/中秋习俗/四大发明)",
            "地理旅行(去哪旅游好/签证怎么办/旅行必备)",
        ],
    },
}


def _build_prompt(category: str, sub_index: int, model_name: str, short_name: str,
                  personality: str, description: str, count: int, batch_index: int) -> str:
    spec = DATA_SPEC[category]
    sub_cats = spec["sub_categories"]
    current_sub = sub_cats[sub_index % len(sub_cats)]

    base = f"""你是专业的AI训练数据生成器。请为"{short_name}"微模型生成{count}组{spec['label']}对话数据。

模型信息:
- 全名: {model_name}
- 简称: {short_name}
- 性格: {personality}
- 定位: {description}

当前子类: {current_sub}
批次: 第{batch_index + 1}批

质量要求（非常重要）:
1. 每组对话必须是独立完整的问答，用户输入和模型回复都要具体、详细
2. 用户输入要多样化，模拟真实人类提问方式，包含口语化表达
3. 模型回复必须自然流畅，像真实对话一样，不能像百科词条
4. 模型回复中必须自然地提及自己是{short_name}（至少30%的回复要提到）
5. 回复长度: 用户输入10-50字，模型回复20-150字
6. 回复要有信息量，不能只说"好的""是的"这类空洞回答
7. 不同对话之间不能重复内容和句式
8. 模型回复要体现{personality}的性格特点

请严格按以下JSON格式输出，不要添加任何其他文字:
[
  {{"user": "用户说的话", "assistant": "{short_name}的回复"}},
  ...
]"""

    if category == "greeting":
        base += f"\n\n特别注意: 回复要热情亲切，{count}组问候方式各不相同"
    elif category == "identity":
        base += f"\n\n特别注意: 每条回复都必须明确说出'{short_name}'这个名称，不能含糊"
    elif category == "daily_chat":
        base += f"\n\n特别注意: 像朋友聊天一样自然，有温度有情感，避免官方腔调"
    elif category == "general_qa":
        base += f"\n\n特别注意: 回答要准确有料，同时保持{short_name}的个性风格"

    return base


class DataGenerator:
    def __init__(self, api_name: str, api_url: str, api_key: str, model: str,
                 model_name: str = "MIX😌混合聊天微模型",
                 short_name: str = "MIX😌",
                 personality: str = "友好、耐心、幽默、专业",
                 description: str = "多场景文本问答、日常聊天、智能互动专属AI微模型"):
        self.api_name = api_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.model_name = model_name
        self.short_name = short_name
        self.personality = personality
        self.description = description

    def _call_api(self, messages: List[Dict], temperature: float = 0.9, max_retries: int = 3) -> Optional[str]:
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"  API调用失败(第{attempt+1}次): {e}")
                time.sleep(3)

        return None

    def _parse_response(self, raw: str, category: str) -> List[Dict]:
        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                raw = json_match.group(0)

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            items = []
            for item in parsed:
                user_content = item.get("user", "").strip()
                assistant_content = item.get("assistant", "").strip()
                if not user_content or not assistant_content:
                    continue
                if len(user_content) < 2 or len(assistant_content) < 5:
                    continue

                msg_hash = hashlib.md5(
                    f"{user_content}:{assistant_content}".encode()
                ).hexdigest()

                items.append({
                    "id": str(uuid.uuid4()),
                    "category": category,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hash": msg_hash,
                })
            return items
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  JSON解析失败: {e}")
            return []

    def generate_batch(self, batch_index: int = 0, output_dir: str = "data/raw") -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)
        all_items = []

        print(f"\n  📊 数据生成规格 (第{batch_index + 1}轮):")
        total_target = 0
        for cat, spec in DATA_SPEC.items():
            print(f"    {spec['label']}: 目标{spec['total_target']}条, 每次{spec['per_call']}条, {len(spec['sub_categories'])}个子类")
            total_target += spec["total_target"]
        print(f"    总目标: {total_target}条/轮\n")

        for category in ["greeting", "identity", "daily_chat", "general_qa"]:
            spec = DATA_SPEC[category]
            print(f"  生成 {spec['label']} (目标{spec['total_target']}条)...")

            cat_items = []
            calls_needed = (spec["total_target"] + spec["per_call"] - 1) // spec["per_call"]

            for call_idx in range(calls_needed):
                sub_idx = (batch_index * calls_needed + call_idx) % len(spec["sub_categories"])
                prompt = _build_prompt(
                    category=category,
                    sub_index=sub_idx,
                    model_name=self.model_name,
                    short_name=self.short_name,
                    personality=self.personality,
                    description=self.description,
                    count=spec["per_call"],
                    batch_index=batch_index * calls_needed + call_idx,
                )

                messages = [
                    {"role": "system", "content": "你是一个专业的AI训练数据生成器，严格按照要求的JSON格式输出高质量的对话训练数据。每次生成的对话必须全新，不能与之前重复。"},
                    {"role": "user", "content": prompt},
                ]

                raw = self._call_api(messages)
                if raw:
                    items = self._parse_response(raw, category)
                    cat_items.extend(items)
                    print(f"    子类{call_idx+1}/{calls_needed} ({spec['sub_categories'][sub_idx][:8]}...): {len(items)}条")
                else:
                    print(f"    子类{call_idx+1}/{calls_needed}: 生成失败")

                if call_idx < calls_needed - 1:
                    time.sleep(1)

            all_items.extend(cat_items)
            print(f"    ✅ {spec['label']}: 共{len(cat_items)}条\n")

        filename = f"batch_{str(batch_index+1).zfill(3)}_{int(time.time())}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)

        print(f"  📁 原始数据已保存: {filename}")
        print(f"  📊 本轮总计: {len(all_items)}条")

        cat_summary = {}
        for item in all_items:
            cat = item["category"]
            cat_summary[cat] = cat_summary.get(cat, 0) + 1
        for cat, count in cat_summary.items():
            print(f"    {DATA_SPEC[cat]['label']}: {count}条")

        return all_items

    def test_connection(self) -> bool:
        messages = [{"role": "user", "content": "Hi"}]
        result = self._call_api(messages, temperature=0.1, max_retries=1)
        return result is not None
