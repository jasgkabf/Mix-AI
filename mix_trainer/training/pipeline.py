"""
MIX😌 全自动闭环训练Pipeline
生成数据 → 清洗 → 训练 → API测试 → 循环强化
支持增量数据策略，每轮生成不同子类避免重复
"""

import os
import sys
import time
import yaml
import json
from typing import Optional

from mix_trainer.data.generator import DataGenerator
from mix_trainer.data.cleaner import DataCleaner
from mix_trainer.training.trainer import MixTrainer
from mix_trainer.utils.chat_engine import MixChatEngine
from mix_trainer.utils.model_tester import ModelTester


class TrainingPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            print(f"❌ 配置文件不存在: {self.config_path}")
            print("请先运行: python -m mix_trainer init")
            sys.exit(1)

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _reload_config(self):
        self.config = self._load_config()

    def _get_generator(self) -> DataGenerator:
        api = self.config["api"]
        model = self.config["model"]
        return DataGenerator(
            api_name=api["name"],
            api_url=api["baseUrl"],
            api_key=api["apiKey"],
            model=api["model"],
            model_name=model["name"],
            short_name=model["shortName"],
            personality=model["personality"],
            description=model["description"],
        )

    def _get_trainer(self) -> MixTrainer:
        training = self.config["training"]
        return MixTrainer(
            data_dir="data/cleaned",
            output_dir="data/model",
            model_size=training.get("modelSize", "small"),
            block_size=training.get("blockSize", 256),
            batch_size=training.get("batchSize", 16),
            epochs=training.get("epochs", 8),
            learning_rate=training.get("learningRate", 5e-4),
            weight_decay=training.get("weightDecay", 0.05),
            dropout=training.get("dropout", 0.2),
            grad_clip=training.get("gradClip", 0.5),
            label_smoothing=training.get("labelSmoothing", 0.1),
            early_stop_patience=training.get("earlyStopPatience", 3),
            early_stop_min_delta=training.get("earlyStopMinDelta", 0.01),
            lr_patience=training.get("lrPatience", 2),
            lr_factor=training.get("lrFactor", 0.5),
            warmup_steps=training.get("warmupSteps", 200),
            val_ratio=training.get("valRatio", 0.2),
            log_interval=training.get("logInterval", 5),
        )

    def _get_tester(self) -> ModelTester:
        api = self.config["api"]
        model = self.config["model"]
        return ModelTester(
            api_name=api["name"],
            api_url=api["baseUrl"],
            api_key=api["apiKey"],
            api_model=api["model"],
            model_name=model["name"],
            short_name=model["shortName"],
        )

    def run_once(self, loop_index: int = 0):
        print(f"\n{'='*60}")
        print(f"  🔄 第 {loop_index + 1} 轮闭环训练")
        print(f"{'='*60}\n")

        print("[1/4] 生成训练数据...")
        generator = self._get_generator()
        raw_data = generator.generate_batch(loop_index, "data/raw")
        print(f"  生成 {len(raw_data)} 条原始数据\n")

        if not raw_data:
            print("⚠ 未生成有效数据，跳过本轮")
            return

        print("[2/4] 清洗训练数据...")
        short_name = self.config.get("model", {}).get("shortName", "MIX😌")
        cleaner = DataCleaner(short_name=short_name)
        cleaned_data = cleaner.clean(raw_data, "data/cleaned")
        print(f"  清洗后 {len(cleaned_data)} 条合格数据\n")

        if not cleaned_data:
            print("⚠ 清洗后无有效数据，跳过训练")
            return

        print("[3/4] 训练MixGPT模型...")
        self._reload_config()
        trainer = self._get_trainer()
        trainer.train()

        print("\n[4/4] API自动测试训练效果...")
        self._test_trained_model()

        self._save_loop_state(loop_index + 1)

    def _test_trained_model(self):
        try:
            engine = MixChatEngine(model_dir="data/model")
            if not engine.load("model_best.pt"):
                print("  ⚠ 模型加载失败，跳过自动测试")
                return

            tester = self._get_tester()

            direct_result = tester.run_direct_test(engine)

            api_result = tester.run_api_eval(direct_result)

            combined = {
                "direct_test": direct_result,
                "api_eval": api_result,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            tester.save_test_result(combined)

            name_rate = direct_result.get("name_rate", 0)
            if name_rate >= 80:
                print(f"\n  ✅ 测试通过！名称记忆率: {name_rate:.0f}%")
            elif name_rate >= 50:
                print(f"\n  ⚠️ 测试一般，名称记忆率: {name_rate:.0f}%，继续训练可提升")
            else:
                print(f"\n  ❌ 测试未通过，名称记忆率: {name_rate:.0f}%，需要更多训练数据")

        except Exception as e:
            print(f"  ⚠ 自动测试出错: {e}")
            print("  可手动运行 mix test 测试")

    def run_loop(self, max_loops: int = 0, interval: int = 300):
        self.running = True
        loop_count = 0

        if max_loops == 0:
            max_loops = self.config.get("training", {}).get("maxLoops", 5)

        print("\n" + "=" * 60)
        print("  🚀 MIX😌 全自动闭环训练已启动")
        print(f"  循环间隔: {interval}秒 | 最大循环: {max_loops}轮")
        print("  按 Ctrl+C 安全停止，停止后可运行 mix chat 聊天")
        print("=" * 60 + "\n")

        try:
            while self.running:
                loop_count += 1
                self.run_once(loop_count - 1)

                if max_loops > 0 and loop_count >= max_loops:
                    print(f"\n✅ 已完成 {max_loops} 轮训练循环")
                    break

                if not self.running:
                    break

                print(f"\n⏳ 等待 {interval} 秒后开始下一轮循环...")
                print("按 Ctrl+C 安全停止，停止后运行 mix chat 与MixGPT聊天\n")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 训练已暂停！")
            print("  模型已保存在 data/model/ 目录")
            print("  运行以下命令与MixGPT聊天:")
            print("    python -m mix_trainer chat")
            self.running = False

    def _save_loop_state(self, loop_count: int):
        state = {
            "loopCount": loop_count,
            "lastLoopTime": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "running": self.running,
        }
        state_path = "data/logs/loop_state.json"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def stop(self):
        self.running = False
