# MIX😌 混合聊天微模型 - 全自动API闭环训练工具

> 基于Python + PyTorch + Transformer Decoder-only架构，**从零训练真实AI模型**
> 自监督预训练（Next Token Prediction），产出真实神经网络权重文件

## 🎯 项目定位

**真正从零训练AI模型**——不是API上下文注入，不是Prompt Engineering，而是用Python + PyTorch实现完整的Transformer Decoder-only架构，通过自监督预训练（Next Token Prediction）训练出真实的神经网络模型。

核心特性：
- 🧠 **真实Transformer架构**：Decoder-only GPT，含RoPE旋转位置编码、SwiGLU激活函数、RMSNorm
- 🔥 **自监督预训练**：Next Token Prediction，与GPT-2/GPT-3/LLaMA相同的训练范式
- 📝 **全自动数据生成**：调用第三方API批量生成高质量训练语料
- 🧹 **数据清洗流水线**：去重/过滤/标准化/BPE分词
- 🔄 **闭环循环训练**：生成→清洗→训练→循环强化
- 💬 **模型推理聊天**：加载训练好的权重文件，本地独立运行

## 🏗️ 核心架构

```
┌──────────────────────────────────────────────────────────┐
│                    MIX😌 CLI 控制台                        │
│  mix init | mix start | mix train | mix chat | ...       │
└──────────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │         全自动闭环训练引擎               │
    │                                         │
    │  [1] API数据生成 ──► 第三方大模型API    │
    │         │              (生成训练语料)    │
    │         ▼                               │
    │  [2] 数据清洗 ──► 去重/过滤/标准化      │
    │         │                               │
    │         ▼                               │
    │  [3] BPE分词 ──► 构建词表+编码文本      │
    │         │                               │
    │         ▼                               │
    │  [4] 自监督训练 ──► PyTorch GPU/CPU     │
    │     Next Token Prediction               │
    │     Decoder-only Transformer            │
    │         │                               │
    │         ▼                               │
    │  [5] 模型保存 ──► .pt权重文件           │
    │         │                               │
    │         ▼                               │
    │  [6] 循环控制器 ──► 回到[1]持续强化     │
    └─────────────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │         训练产出 (真实模型)              │
    │                                         │
    │  data/model/model_best.pt  ← 最佳权重   │
    │  data/model/tokenizer.json ← BPE分词器  │
    │  data/model/model_config.json ← 配置    │
    │  data/model/training_history.json       │
    └─────────────────────────────────────────┘
```

## 🧠 模型架构详解

### Decoder-only Transformer（与GPT-2/LLaMA同架构）

| 组件 | 实现 | 说明 |
|------|------|------|
| 注意力机制 | Causal Self-Attention | 因果自注意力，防止看到未来token |
| 位置编码 | Rotary Position Embedding (RoPE) | 旋转位置编码，LLaMA同款 |
| 激活函数 | SwiGLU | LLaMA同款，优于传统ReLU/GELU |
| 归一化 | RMSNorm | 替代LayerNorm，更高效 |
| MLP | Gate+Up+Down投影 | SwiGLU三投影结构 |
| 训练目标 | Next Token Prediction | 自监督预训练，预测下一个token |
| 采样策略 | Top-K + Top-P + 重复惩罚 | 多样化且连贯的文本生成 |

### 模型规模预设

| 预设 | 参数量 | 层数 | 维度 | 头数 | 上下文 | 适用场景 |
|------|--------|------|------|------|--------|---------|
| tiny | ~2M | 4 | 64 | 2 | 128 | 快速实验 |
| small | ~10M | 6 | 256 | 4 | 256 | **推荐** 4核4G服务器 |
| medium | ~50M | 8 | 512 | 8 | 512 | GPU服务器 |
| large | ~150M | 12 | 768 | 12 | 1024 | 高端GPU |

## 📋 服务器要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| CPU | 4核 | 8核+ |
| 内存 | 4GB | 8GB+ |
| GPU | 不需要（CPU可训练） | NVIDIA GPU加速 |
| 系统 | Linux | Ubuntu 20.04+ |
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | 最新版 |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/mix-trainer.git
cd mix-trainer
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 初始化配置

```bash
python -m mix_trainer.cli.main init
```

按提示输入：
- **API提供商名称**：如 DeepSeek、OpenAI、Qwen
- **API基础地址**：如 `https://api.deepseek.com/v1`
- **API密钥**：你的API Key
- **模型名称**：如 `deepseek-chat`
- **训练参数**：模型大小、轮次等

### 4. 启动全自动训练

```bash
python -m mix_trainer.cli.main start
```

### 5. 与训练好的模型聊天

```bash
python -m mix_trainer.cli.main chat
```

## 📖 CLI 命令详解

| 命令 | 说明 |
|------|------|
| `mix init` | 初始化项目，配置API和训练参数 |
| `mix config [show\|edit\|test]` | 查看/修改配置，测试API连接 |
| `mix generate [--count N]` | 调用API生成训练数据 |
| `mix clean` | 清洗训练数据（去重/过滤/标准化） |
| `mix train` | 训练Transformer模型（自监督预训练） |
| `mix start` | 启动全自动闭环训练循环 |
| `mix chat` | 加载训练好的模型，交互式聊天 |
| `mix info` | 查看模型训练成果详情 |
| `mix status` | 查看训练状态 |
| `mix reset` | 重置所有训练数据 |

> 所有命令通过 `python -m mix_trainer.cli.main <command>` 执行

## ⚙️ 配置文件说明

配置文件为 `config.yaml`，参考 `config.example.yaml`：

```yaml
api:
  name: "DeepSeek"
  baseUrl: "https://api.deepseek.com/v1"
  apiKey: "sk-xxxxxxxx"
  model: "deepseek-chat"

training:
  modelSize: "small"    # tiny/small/medium/large
  blockSize: 256        # 上下文窗口
  batchSize: 8          # 批量大小
  epochs: 50            # 训练轮次
  learningRate: 0.0003  # 学习率
  weightDecay: 0.1      # 权重衰减
  dropout: 0.1          # Dropout
  gradClip: 1.0         # 梯度裁剪
  loopInterval: 300     # 闭环间隔(秒)

model:
  name: "MIX😌混合聊天微模型"
  shortName: "MIX😌"
  personality: "友好、耐心、幽默、专业"
  description: "多场景文本问答、日常聊天、智能互动专属AI微模型"
```

## 📁 项目结构

```
mix-trainer/
├── mix_trainer/                    # 核心Python包
│   ├── model/
│   │   ├── config.py               # 模型配置(ModelConfig/ModelSize)
│   │   └── gpt.py                  # 🧠 GPT Decoder-only Transformer
│   │       ├── RotaryPositionEmbedding  (RoPE旋转位置编码)
│   │       ├── CausalSelfAttention      (因果自注意力)
│   │       ├── SwiGLUMLP                (SwiGLU前馈网络)
│   │       ├── TransformerBlock         (Transformer块)
│   │       └── MixGPT                   (完整GPT模型)
│   ├── data/
│   │   ├── tokenizer.py            # BPE分词器
│   │   ├── dataset.py              # 训练数据集(Next Token Prediction)
│   │   ├── generator.py            # API训练数据生成器
│   │   └── cleaner.py              # 数据清洗器
│   ├── training/
│   │   ├── trainer.py              # 🔥 自监督预训练器
│   │   └── pipeline.py             # 🔄 全自动闭环Pipeline
│   ├── utils/
│   │   └── chat_engine.py          # 💬 模型推理/聊天引擎
│   └── cli/
│       └── main.py                 # CLI命令行工具
├── data/
│   ├── raw/                        # 原始生成数据
│   ├── cleaned/                    # 清洗后数据
│   ├── model/                      # 🧠 训练产出(权重/分词器/配置)
│   └── logs/                       # 训练日志
├── config.example.yaml
├── requirements.txt
└── README.md
```

## 🔄 训练流程详解

### 自监督预训练：Next Token Prediction

```
输入序列:  <BOS> <USER> 你好 <SEP> <ASSISTANT> 你好！我是MIX😌
目标序列:  <USER> 你好 <SEP> <ASSISTANT> 你好！我是MIX😌 <EOS>

模型学习: 给定前面的token，预测下一个token
这是GPT-2/GPT-3/LLaMA等大模型使用的完全相同的训练范式
```

### 全自动闭环流程

```
mix start
    │
    ▼
[1] API生成数据 ──► 4类训练语料(问候/身份/日常/问答)
    │
    ▼
[2] 数据清洗 ──► 去重→过滤→标准化
    │
    ▼
[3] BPE分词 ──► 构建词表→编码文本→构建训练样本
    │
    ▼
[4] 自监督训练 ──► PyTorch训练Next Token Prediction
    │                 AdamW优化器 + Cosine学习率调度
    │                 梯度裁剪 + 权重衰减
    ▼
[5] 模型保存 ──► model_best.pt (真实神经网络权重)
    │
    ▼
[6] 循环控制 ──► 等待间隔→回到[1]持续强化
```

### 训练完成后

```bash
# 查看模型训练成果
python -m mix_trainer.cli.main info

# 与训练好的模型聊天（加载真实权重文件）
python -m mix_trainer.cli.main chat
```

训练产出文件：
- `data/model/model_best.pt` — 最佳模型权重（真实PyTorch模型）
- `data/model/tokenizer.json` — BPE分词器
- `data/model/model_config.json` — 模型配置
- `data/model/training_history.json` — 训练历史

## 🔌 支持的API提供商

任何兼容OpenAI Chat Completions API格式的第三方大模型均可使用：

| 提供商 | 基础地址 | 模型示例 |
|--------|---------|---------|
| OpenAI | `https://api.openai.com/v1` | gpt-3.5-turbo |
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-turbo |
| 智谱GLM | `https://open.bigmodel.cn/api/paas/v4` | glm-4-flash |
| 月之暗面 | `https://api.moonshot.cn/v1` | moonshot-v1-8k |

## 💡 核心优势

1. **真实AI训练** — Python + PyTorch + Transformer，从零训练神经网络权重
2. **与GPT同架构** — Decoder-only + RoPE + SwiGLU + RMSNorm，LLaMA同款设计
3. **自监督预训练** — Next Token Prediction，大模型标准训练范式
4. **全自动闭环** — 生成数据→清洗→训练→循环，全程无人值守
5. **CPU可训练** — tiny/small模型4核4G服务器即可训练
6. **GPU加速** — 有GPU自动使用CUDA加速训练
7. **产出真实模型** — .pt权重文件，可独立加载运行推理

## 📄 License

MIT
