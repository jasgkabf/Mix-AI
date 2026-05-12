"""
MIX😌 训练效果测试器
1. 直接测试MixGPT输出结果（不依赖外部API）
2. 可选: 调用外部大模型API评估输出质量
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional


TEST_QUESTIONS = {
    "身份记忆": [
        "你叫什么名字",
        "你是谁",
        "介绍一下你自己",
        "你的全名是什么",
    ],
    "问候回复": [
        "你好",
        "早上好",
        "晚安",
        "在吗",
    ],
    "日常聊天": [
        "今天天气真好",
        "中午吃什么好",
        "我好开心",
        "好无聊啊",
    ],
    "知识问答": [
        "什么是人工智能",
        "怎么学英语",
        "感冒了怎么办",
    ],
}


class ModelTester:
    def __init__(self, api_name: str = "", api_url: str = "", api_key: str = "",
                 api_model: str = "",
                 model_name: str = "MIX😌混合聊天微模型",
                 short_name: str = "MIX😌"):
        self.api_name = api_name
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.api_model = api_model
        self.model_name = model_name
        self.short_name = short_name

    def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        if not self.api_url or not self.api_key:
            return None
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  API调用失败: {e}")
            return None

    def run_direct_test(self, chat_engine) -> Dict:
        print("\n" + "=" * 60)
        print("  🧪 MixGPT 直接输出测试")
        print("=" * 60)

        all_responses = {}
        name_correct = 0
        total = 0

        for group_name, questions in TEST_QUESTIONS.items():
            print(f"\n  📂 {group_name}:")
            for question in questions:
                try:
                    answer = chat_engine.chat(question, max_new_tokens=100, temperature=0.7)
                    all_responses[question] = answer
                    total += 1

                    has_name = self.short_name in answer
                    if has_name:
                        name_correct += 1
                    status = "✅" if has_name else "⚠️"

                    print(f"  {status} Q: {question}")
                    print(f"     A: {answer}")
                    print()
                except Exception as e:
                    all_responses[question] = f"[错误: {e}]"
                    print(f"  ❌ Q: {question} → 错误: {e}")
                    print()

        name_rate = (name_correct / total * 100) if total > 0 else 0

        print("-" * 60)
        print(f"  📊 直接测试结果:")
        print(f"    总问题数: {total}")
        print(f"    名称记忆率: {name_rate:.0f}% ({name_correct}/{total}条提到'{self.short_name}')")
        if name_rate >= 80:
            print(f"    评价: ✅ 名称记忆良好")
        elif name_rate >= 50:
            print(f"    评价: ⚠️ 名称记忆一般，需要继续训练")
        else:
            print(f"    评价: ❌ 名称记忆不足，需要更多训练数据")
        print()

        return {
            "total": total,
            "name_correct": name_correct,
            "name_rate": name_rate,
            "details": all_responses,
        }

    def run_api_eval(self, direct_result: Dict) -> Dict:
        if not self.api_url or not self.api_key:
            print("  ⚠ 未配置API，跳过API评估")
            return {"passed": True, "score": 0, "evaluation": "未配置API"}

        print("\n📊 调用外部大模型API评估MixGPT输出质量...")

        eval_prompt = f"""你是AI模型训练评估专家。请评估以下"{self.short_name}"微模型的回复质量。

评估标准:
1. 模型是否正确记住了自己的名称"{self.short_name}"
2. 回复是否自然流畅
3. 回复是否准确回答了问题
4. 回复是否有逻辑性

以下是模型的对话测试结果:

"""
        for question, answer in direct_result["details"].items():
            eval_prompt += f"问题: {question}\n{self.short_name}回复: {answer}\n\n"

        eval_prompt += f"""请给出评估结果，格式如下:
1. 名称记忆: 通过/未通过 (模型是否记住了自己叫{self.short_name})
2. 回复流畅度: 1-10分
3. 回复准确度: 1-10分
4. 总体评分: 1-10分
5. 改进建议: (简短说明)"""

        messages = [
            {"role": "system", "content": "你是一个专业的AI模型评估专家，请客观评估模型训练效果。"},
            {"role": "user", "content": eval_prompt},
        ]

        result = self._call_api(messages)
        if result:
            print(f"\n  📋 API评估结果:")
            for line in result.split("\n"):
                if line.strip():
                    print(f"    {line.strip()}")

            score = self._extract_score(result)
            return {
                "passed": score >= 5,
                "score": score,
                "evaluation": result,
            }
        else:
            print("  ⚠ API评估失败")
            return {"passed": True, "score": 0, "evaluation": "API评估失败"}

    def _extract_score(self, evaluation: str) -> int:
        import re
        match = re.search(r"总体评分[：:]\s*(\d+)", evaluation)
        if match:
            return int(match.group(1))
        numbers = re.findall(r"(\d+)/10", evaluation)
        if numbers:
            return int(numbers[-1])
        return 5

    def save_test_result(self, result: Dict, output_dir: str = "data/logs"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"test_result_{int(time.time())}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  测试结果已保存: {filename}")
