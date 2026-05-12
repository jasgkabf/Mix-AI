"""
MIX😌 自监督预训练器 - Next Token Prediction
基于PyTorch + Transformer Decoder-only
支持CPU/GPU训练，支持DeepSpeed分布式训练
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from mix_trainer.model.gpt import MixGPT
from mix_trainer.model.config import ModelConfig, ModelSize
from mix_trainer.data.tokenizer import BPETokenizer
from mix_trainer.data.dataset import ConversationDataset


class MixTrainer:
    def __init__(
        self,
        data_dir: str = "data/cleaned",
        output_dir: str = "data/model",
        model_size: str = "small",
        block_size: int = 256,
        batch_size: int = 8,
        epochs: int = 50,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        grad_clip: float = 1.0,
        log_interval: int = 5,
        eval_interval: int = 1,
        save_interval: int = 5,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_size = ModelSize(model_size)
        self.block_size = block_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float("inf")
        self.train_history = []

        os.makedirs(output_dir, exist_ok=True)

    def setup(self):
        print("=" * 60)
        print("  🤖 MIX😌 混合聊天微模型 - 真实Transformer训练")
        print("  架构: Decoder-only GPT | 自监督: Next Token Prediction")
        print("  框架: PyTorch | 设备: " + str(self.device).upper())
        print("=" * 60)
        print()

        print("[1/4] 加载训练数据...")
        self.tokenizer = BPETokenizer()
        dataset = ConversationDataset(
            self.data_dir, self.tokenizer, self.block_size
        )

        if len(dataset) == 0:
            print("❌ 没有训练数据！请先运行 mix generate 和 mix clean")
            sys.exit(1)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            if val_size > 0
            else None
        )

        print(f"  训练集: {train_size} | 验证集: {val_size}")
        print()

        print("[2/4] 初始化GPT模型...")
        config = ModelConfig.from_preset(self.model_size, self.tokenizer.vocab_size)
        config.block_size = self.block_size
        config.dropout = self.dropout

        self.model = MixGPT(config).to(self.device)
        n_params = self.model.get_num_params()
        print(f"  模型参数量: {n_params/1e6:.2f}M")
        print()

        tokenizer_path = os.path.join(self.output_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        print("[3/4] 配置优化器...")
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=(0.9, 0.95),
            device_type=self.device.type,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5
        )
        print(f"  优化器: AdamW | 学习率: {self.learning_rate} | 调度: CosineAnnealing")
        print()

        config_path = os.path.join(self.output_dir, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        print("[4/4] 训练准备完成！")
        print(f"  训练轮次: {self.epochs} | 批量大小: {self.batch_size}")
        print(f"  梯度裁剪: {self.grad_clip} | 权重衰减: {self.weight_decay}")
        print("-" * 60)

    def train(self):
        self.setup()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            total_loss = 0.0
            batch_count = 0

            for batch_idx, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits, loss = self.model(x, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if (batch_idx + 1) % self.log_interval == 0:
                    avg_loss = total_loss / batch_count
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch {epoch+1}/{self.epochs} | "
                        f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                        f"耗时: {elapsed:.0f}s"
                    )

            self.scheduler.step()

            avg_train_loss = total_loss / max(batch_count, 1)

            avg_val_loss = avg_train_loss
            if self.val_loader is not None and (epoch + 1) % self.eval_interval == 0:
                avg_val_loss = self._evaluate()

            epoch_time = time.time() - epoch_start
            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "time": epoch_time,
            })

            print(
                f"  ✅ Epoch {epoch+1} 完成 | "
                f"训练Loss: {avg_train_loss:.4f} | "
                f"验证Loss: {avg_val_loss:.4f} | "
                f"耗时: {epoch_time:.0f}s"
            )

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self._save_model("model_best.pt", epoch, avg_train_loss, avg_val_loss)
                print(f"  💾 最佳模型已保存 (val_loss: {self.best_val_loss:.4f})")

            if (epoch + 1) % self.save_interval == 0:
                self._save_model(f"model_epoch_{epoch+1}.pt", epoch, avg_train_loss, avg_val_loss)

        self._save_model("model_final.pt", self.epochs, avg_train_loss, avg_val_loss)

        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)

        total_time = time.time() - start_time
        print()
        print("=" * 60)
        print(f"  🎉 训练完成！")
        print(f"  总耗时: {total_time/60:.1f}分钟")
        print(f"  最佳验证Loss: {self.best_val_loss:.4f}")
        print(f"  模型参数: {self.model.get_num_params()/1e6:.2f}M")
        print(f"  模型保存: {self.output_dir}")
        print(f"  运行聊天: python -m mix_trainer.cli.chat --model_dir {self.output_dir}")
        print("=" * 60)

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, loss = self.model(x, y)
                total_loss += loss.item()
                batch_count += 1

        return total_loss / max(batch_count, 1)

    def _save_model(self, filename: str, epoch: int, train_loss: float, val_loss: float):
        filepath = os.path.join(self.output_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": self.model.config.to_dict(),
            "best_val_loss": self.best_val_loss,
        }, filepath)
