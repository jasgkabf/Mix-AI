"""
MIX😌 微模型推理引擎 - 加载训练好的模型进行聊天
"""

import os
import json
import torch
from typing import Optional, List

from mix_trainer.model.gpt import MixGPT
from mix_trainer.model.config import ModelConfig
from mix_trainer.data.tokenizer import BPETokenizer


class MixChatEngine:
    def __init__(self, model_dir: str = "data/model"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, model_file: str = "model_best.pt") -> bool:
        config_path = os.path.join(self.model_dir, "model_config.json")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        model_path = os.path.join(self.model_dir, model_file)

        if not os.path.exists(config_path):
            print(f"❌ 模型配置不存在: {config_path}")
            return False
        if not os.path.exists(tokenizer_path):
            print(f"❌ 分词器不存在: {tokenizer_path}")
            return False
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        self.config = ModelConfig.from_dict(config_dict)

        self.tokenizer = BPETokenizer.load(tokenizer_path)

        self.model = MixGPT(self.config).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        n_params = self.model.get_num_params()
        print(f"✅ 模型加载成功: {n_params/1e6:.2f}M参数 | 设备: {self.device}")
        return True

    @torch.no_grad()
    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return "模型未加载，请先运行 mix train"

        prompt = f"<BOS><USER>{user_input}<SEP><ASSISTANT>"
        input_ids = self.tokenizer.encode(prompt)

        if len(input_ids) == 0:
            return "无法理解输入"

        idx = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        output_ids = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        generated_ids = output_ids[0].tolist()
        response = self.tokenizer.decode(generated_ids)

        if "<ASSISTANT>" in response:
            response = response.split("<ASSISTANT>")[-1]
        if "<EOS>" in response:
            response = response.split("<EOS>")[0]
        if "<USER>" in response:
            response = response.split("<USER>")[0]

        response = response.strip()
        return response if response else "..."

    def interactive_chat(self):
        print()
        print("=" * 50)
        print("  🤖 MIX😌 混合聊天微模型")
        print("  基于真实Transformer Decoder-only架构")
        print("  输入消息开始聊天，/quit 退出")
        print("=" * 50)
        print()

        if self.model is None:
            success = self.load()
            if not success:
                return

        n_params = self.model.get_num_params()
        print(f"  模型参数: {n_params/1e6:.2f}M | 设备: {self.device}")
        print(f"  词表大小: {self.tokenizer.vocab_size}")
        print("-" * 50)
        print()

        while True:
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！👋")
                break

            if not user_input:
                continue
            if user_input in ["/quit", "/exit"]:
                print("再见！👋")
                break
            if user_input == "/help":
                print("  /quit  - 退出聊天")
                print("  /help  - 显示帮助")
                print("  /info  - 模型信息")
                continue
            if user_input == "/info":
                print(f"  参数量: {self.model.get_num_params()/1e6:.2f}M")
                print(f"  词表: {self.tokenizer.vocab_size}")
                print(f"  上下文: {self.config.block_size}")
                print(f"  层数: {self.config.n_layer}")
                print(f"  维度: {self.config.n_embd}")
                continue

            response = self.chat(user_input)
            print(f"MIX😌: {response}")
            print()
