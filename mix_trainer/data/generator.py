"""
MIX😌 训练数据生成器 - 调用第三方大模型API批量生成高质量训练语料
真实AI训练需要大量高质量数据，每轮生成10000+条标准对话
重点: 子类多样性 > 数量，回复简洁自然，避免重复和过度自我介绍
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
        "per_call": 15,
        "total_target": 2000,
        "sub_categories": [
            "基础问候(你好/嗨/哈喽/嘿)",
            "时间问候(早上好/中午好/晚上好/晚安/下午好)",
            "场景问候(在吗/最近怎么样/好久不见/好久没聊)",
            "情绪问候(今天心情不错/好累啊/开心/烦死了)",
            "节日问候(新年好/生日快乐/圣诞快乐/中秋快乐)",
            "网络用语问候(哈喽啊/在不在/滴滴/冒个泡)",
            "方言问候(侬好/嘎哈呢/弄啥嘞/吃了没)",
            "正式场合问候(您好/久仰/幸会/初次见面)",
            "朋友间问候(兄弟/姐妹/宝子/亲爱的)",
            "长辈问候(叔叔好/阿姨好/老师好/领导好)",
        ],
    },
    "identity": {
        "label": "身份认知类",
        "per_call": 15,
        "total_target": 2000,
        "sub_categories": [
            "名称记忆(你叫什么/你的名字/你叫啥/你叫啥名)",
            "功能介绍(你能做什么/你会什么/你擅长什么/你有什么功能)",
            "身份对比(你和ChatGPT区别/你和其他AI不同/你比Siri强在哪)",
            "自我介绍(介绍一下你自己/说说你的特点/你是谁)",
            "能力边界(你不会什么/你的局限是什么/有什么你做不到)",
            "性格描述(你是什么性格/你的风格是什么/你什么脾气)",
            "价值观表达(你觉得什么重要/你的原则是什么/你怎么看世界)",
            "角色扮演(如果你是人类/你想象自己是/假如你住在)",
            "版本信息(你是第几代/你是什么版本/你更新了吗)",
            "创造者信息(谁创造了你/谁开发了你/谁是你爸爸)",
        ],
    },
    "daily_chat": {
        "label": "日常聊天类",
        "per_call": 15,
        "total_target": 3000,
        "sub_categories": [
            "天气季节(今天好热/最近总下雨/冬天好冷/春天来了)",
            "美食饮食(中午吃什么/推荐个菜/你会做饭吗/最近吃了啥)",
            "心情情感(我好开心/有点难过/压力好大/好感动)",
            "兴趣爱好(喜欢什么电影/推荐首歌/周末干嘛/最近追什么剧)",
            "生活日常(好无聊/刚起床/加班好累/今天好忙)",
            "社交人际(朋友过生日送什么/怎么认识新朋友/和同事吵架了)",
            "宠物动物(想养猫/狗狗好可爱/金鱼怎么养/仓鼠好萌)",
            "影视娱乐(最近什么电影好看/综艺推荐/追剧推荐)",
            "音乐艺术(推荐好听的歌/学乐器/画画/听什么音乐)",
            "运动健身(怎么减肥/跑步/瑜伽/游泳/健身计划)",
            "读书学习(推荐一本书/怎么读书/最近看了什么/学习习惯)",
            "游戏电竞(推荐个游戏/手游/端游/最近玩什么)",
            "时尚穿搭(今天穿什么/搭配建议/买衣服/护肤)",
            "旅行见闻(去哪旅游好/旅行故事/攻略/签证)",
            "家居生活(装修/收纳/做饭/打扫/养花)",
        ],
    },
    "general_qa": {
        "label": "通用问答类",
        "per_call": 15,
        "total_target": 3000,
        "sub_categories": [
            "科学知识(光速是多少/地球多大/水为什么沸腾/黑洞是什么)",
            "生活常识(怎么去油渍/感冒怎么办/手机省电/衣服缩水)",
            "学习方法(怎么学英语/如何提高记忆力/读书方法/考试技巧)",
            "工作职场(如何提高效率/面试技巧/时间管理/升职加薪)",
            "科技数码(什么是AI/5G有什么用/区块链解释/量子计算)",
            "健康养生(每天喝多少水/怎么改善睡眠/运动建议/减压方法)",
            "文化历史(春节由来/中秋习俗/四大发明/历史故事)",
            "地理旅行(去哪旅游好/签证怎么办/旅行必备/景点推荐)",
            "数学计算(算术题/概率问题/逻辑推理/脑筋急转弯)",
            "语言翻译(这个词怎么翻译/英语怎么说/日语怎么讲)",
            "写作辅助(帮我写个开头/作文怎么写/文案/邮件)",
            "编程帮助(Python怎么学/代码报错/算法/前端后端)",
            "情感咨询(失恋怎么办/怎么表白/异地恋/友情问题)",
            "笑话段子(讲个笑话/冷笑话/谐音梗/脑筋急转弯)",
            "哲学思考(人生的意义/自由是什么/幸福是什么/时间是什么)",
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
1. 每组对话必须是独立完整的问答
2. 用户输入要多样化，模拟真实人类提问方式，包含口语化表达
3. 模型回复必须简洁自然，1-3句话即可，像朋友聊天
4. 不要每次都自我介绍！只有被问到名字/身份时才提及{short_name}
5. 回复长度: 用户输入5-30字，模型回复10-80字（简短优先！）
6. 回复要口语化，不要像写作文或产品说明书
7. 同一个问题要有不同的回答方式，避免千篇一律
8. 语气要自然轻松，像朋友聊天，不要像客服
9. 每条对话都必须是全新的，不能与之前生成的任何对话相似

请严格按以下JSON格式输出，不要添加任何其他文字:
[
  {{"user": "用户说的话", "assistant": "{short_name}的回复"}},
  ...
]"""

    if category == "greeting":
        base += f"\n\n特别注意: 回复要热情亲切，{count}组问候方式各不相同，不要重复"
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

    def _call_api(self, messages: List[Dict], temperature: float = 0.85, max_retries: int = 3) -> Optional[str]:
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 16384,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                finish_reason = data["choices"][0].get("finish_reason", "")
                if finish_reason == "length":
                    print(f"  ⚠ API响应被截断(finish_reason=length)，数据可能不完整")
                return content
            except Exception as e:
                print(f"  API调用失败(第{attempt+1}次): {e}")
                time.sleep(3)

        return None

    def _repair_json(self, raw: str) -> Optional[str]:
        import re
        json_match = re.search(r'\[[\s\S]*', raw)
        if not json_match:
            return None
        raw = json_match.group(0)

        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass

        complete_objects = []
        obj_pattern = re.compile(r'\{\s*"user"\s*:\s*"', re.DOTALL)
        pos = 0
        while pos < len(raw):
            m = obj_pattern.search(raw, pos)
            if not m:
                break
            obj_start = m.start()
            depth = 0
            in_str = False
            escape = False
            i = obj_start
            while i < len(raw):
                ch = raw[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    i += 1
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = raw[obj_start:i + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and obj.get("user") and obj.get("assistant"):
                                    complete_objects.append(candidate)
                            except json.JSONDecodeError:
                                pass
                            pos = i + 1
                            break
                i += 1
            else:
                pos = i

        if complete_objects:
            return "[" + ",".join(complete_objects) + "]"
        return None

    def _parse_response(self, raw: str, category: str) -> List[Dict]:
        import re
        try:
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                raw = json_match.group(0)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                repaired = self._repair_json(raw)
                if repaired:
                    parsed = json.loads(repaired)
                    print(f"  🔧 JSON修复成功，提取到{len(parsed)}条有效数据")
                else:
                    print(f"  JSON解析失败且修复无效，跳过此批")
                    return []

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
                    {"role": "system", "content": "你是一个专业的AI训练数据生成器，严格按照要求的JSON格式输出高质量的对话训练数据。每次生成的对话必须全新，不能与之前重复。确保输出完整的JSON数组。"},
                    {"role": "user", "content": prompt},
                ]

                raw = self._call_api(messages)
                if raw:
                    items = self._parse_response(raw, category)
                    cat_items.extend(items)
                    sub_name = spec['sub_categories'][sub_idx]
                    sub_label = sub_name.split('(')[0] if '(' in sub_name else sub_name[:8]
                    print(f"    子类{call_idx+1}/{calls_needed} ({sub_label}): {len(items)}条")
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
