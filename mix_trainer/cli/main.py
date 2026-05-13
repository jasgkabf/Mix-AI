"""
MIX😌 CLI 命令行工具 - 类似Claude Code的全局命令控制
"""

import os
import sys
import argparse
import yaml
import json


def get_project_root():
    return os.getcwd()


def load_config():
    config_path = os.path.join(get_project_root(), "config.yaml")
    if not os.path.exists(config_path):
        print("❌ 配置文件不存在，请先运行: mix init")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config):
    config_path = os.path.join(get_project_root(), "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)


def cmd_init(args):
    print()
    print("=" * 50)
    print("  🚀 MIX😌 混合聊天微模型 - 项目初始化")
    print("  基于真实Transformer Decoder-only训练")
    print("=" * 50)
    print()

    config = {
        "api": {
            "name": "",
            "baseUrl": "",
            "apiKey": "",
            "model": "",
        },
        "training": {
            "modelSize": "small",
            "blockSize": 256,
            "batchSize": 16,
            "epochs": 8,
            "learningRate": 0.0005,
            "weightDecay": 0.05,
            "dropout": 0.2,
            "gradClip": 0.5,
            "labelSmoothing": 0.1,
            "earlyStopPatience": 3,
            "earlyStopMinDelta": 0.01,
            "lrPatience": 2,
            "lrFactor": 0.5,
            "warmupSteps": 200,
            "valRatio": 0.2,
            "logInterval": 5,
            "loopInterval": 300,
            "maxLoops": 5,
        },
        "model": {
            "name": "MIX😌混合聊天微模型",
            "shortName": "MIX😌",
            "personality": "友好、耐心、幽默、专业",
            "description": "多场景文本问答、日常聊天、智能互动专属AI微模型",
        },
    }

    print("请配置第三方大模型API（用于生成训练数据）:")
    print()

    config["api"]["name"] = input("  API提供商名称 (如 DeepSeek, OpenAI, Qwen): ").strip()
    config["api"]["baseUrl"] = input("  API基础地址 (如 https://api.deepseek.com/v1): ").strip()
    config["api"]["apiKey"] = input("  API密钥: ").strip()
    config["api"]["model"] = input("  模型名称 (如 deepseek-chat): ").strip()

    print()
    print("训练参数配置 (直接回车使用默认值):")

    model_size = input(f"  模型大小 [tiny/small/medium] (默认: small): ").strip()
    if model_size in ["tiny", "small", "medium"]:
        config["training"]["modelSize"] = model_size

    epochs = input(f"  训练轮次 (默认: 50): ").strip()
    if epochs.isdigit():
        config["training"]["epochs"] = int(epochs)

    batch_size = input(f"  批量大小 (默认: 8): ").strip()
    if batch_size.isdigit():
        config["training"]["batchSize"] = int(batch_size)

    print()
    print("测试API连接...")
    from mix_trainer.data.generator import DataGenerator
    gen = DataGenerator(
        config["api"]["name"], config["api"]["baseUrl"],
        config["api"]["apiKey"], config["api"]["model"],
        config["model"]["name"], config["model"]["shortName"],
    )
    if gen.test_connection():
        print("✅ API连接成功！")
    else:
        print("⚠️ API连接失败，请检查配置")

    save_config(config)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/cleaned", exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print()
    print("✅ 初始化完成！")
    print()
    print("下一步操作:")
    print("  mix start     - 启动全自动闭环训练")
    print("  mix generate  - 仅生成训练数据")
    print("  mix train     - 仅训练模型")
    print("  mix chat      - 与训练好的模型聊天")
    print()


def cmd_config(args):
    config = load_config()
    action = args.action or "show"

    if action == "show":
        print("\n📋 当前配置:")
        print(f"  API: {config['api']['name']} ({config['api']['baseUrl']})")
        print(f"  模型: {config['api']['model']}")
        print(f"  密钥: {config['api']['apiKey'][:8]}{'*' * 20}")
        print(f"  训练模型大小: {config['training']['modelSize']}")
        print(f"  训练轮次: {config['training']['epochs']}")
        print(f"  批量大小: {config['training']['batchSize']}")
        print(f"  微模型名称: {config['model']['name']}")
    elif action == "test":
        from mix_trainer.data.generator import DataGenerator
        gen = DataGenerator(
            config["api"]["name"], config["api"]["baseUrl"],
            config["api"]["apiKey"], config["api"]["model"],
        )
        if gen.test_connection():
            print("✅ API连接正常")
        else:
            print("❌ API连接失败")
    elif action == "edit":
        print("请使用文本编辑器修改 config.yaml 文件")


def cmd_generate(args):
    config = load_config()
    from mix_trainer.data.generator import DataGenerator

    gen = DataGenerator(
        config["api"]["name"], config["api"]["baseUrl"],
        config["api"]["apiKey"], config["api"]["model"],
        config["model"]["name"], config["model"]["shortName"],
        config["model"]["personality"],
        config["model"]["description"],
    )

    count = args.count or 1
    print(f"\n📝 生成训练数据 ({count} 批)\n")

    total = 0
    for i in range(count):
        items = gen.generate_batch(i, "data/raw")
        total += len(items)
        print(f"  第 {i+1} 批: {len(items)} 条")

    print(f"\n✅ 共生成 {total} 条训练数据")
    print("运行 mix clean 清洗数据，或 mix start 自动完成全流程")


def cmd_clean(args):
    import glob
    from mix_trainer.data.cleaner import DataCleaner

    raw_dir = "data/raw"
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print("❌ 没有原始数据，请先运行 mix generate")
        return

    short_name = "MIX😌"
    try:
        config = load_config()
        short_name = config.get("model", {}).get("shortName", "MIX😌")
    except Exception:
        pass

    cleaner = DataCleaner(short_name=short_name)
    all_raw = []
    for filepath in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_raw.extend(data)
        except Exception:
            continue

    print(f"\n🧹 清洗训练数据 ({len(all_raw)} 条原始数据)\n")
    cleaned = cleaner.clean(all_raw, "data/cleaned")
    print(f"\n✅ 清洗完成: {len(all_raw)} → {len(cleaned)} 条")


def cmd_train(args):
    config = load_config()
    from mix_trainer.training.trainer import MixTrainer

    training = config["training"]
    trainer = MixTrainer(
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
    )
    trainer.train()


def cmd_start(args):
    config = load_config()
    from mix_trainer.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    interval = config["training"].get("loopInterval", 300)
    pipeline.run_loop(interval=interval)


def cmd_chat(args):
    from mix_trainer.utils.chat_engine import MixChatEngine

    engine = MixChatEngine(model_dir=args.model_dir or "data/model")
    if not engine.load(args.model_file or "model_best.pt"):
        print("❌ 模型加载失败，请先运行 mix train 训练模型")
        return
    engine.interactive_chat()


def cmd_info(args):
    model_dir = args.model_dir or "data/model"
    config_path = os.path.join(model_dir, "model_config.json")
    history_path = os.path.join(model_dir, "training_history.json")

    print()
    print("=" * 50)
    print("  🧠 MIX😌 微模型训练成果")
    print("=" * 50)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"\n  模型架构: Decoder-only GPT Transformer")
        print(f"  模型大小: {cfg.get('model_size', '?')}")
        print(f"  层数: {cfg.get('n_layer', '?')}")
        print(f"  维度: {cfg.get('n_embd', '?')}")
        print(f"  头数: {cfg.get('n_head', '?')}")
        print(f"  词表: {cfg.get('vocab_size', '?')}")
        print(f"  上下文: {cfg.get('block_size', '?')}")
    else:
        print("\n  ⚠ 模型尚未训练")
        print("  运行 mix start 开始训练")
        return

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if history:
            last = history[-1]
            print(f"\n  训练轮次: {last.get('epoch', '?')}")
            print(f"  训练Loss: {last.get('train_loss', '?'):.4f}")
            print(f"  验证Loss: {last.get('val_loss', '?'):.4f}")

    best_path = os.path.join(model_dir, "model_best.pt")
    if os.path.exists(best_path):
        size_mb = os.path.getsize(best_path) / (1024 * 1024)
        print(f"\n  最佳模型: {size_mb:.1f} MB")

    print(f"\n  运行聊天: mix chat")
    print()


def cmd_status(args):
    state_path = "data/logs/loop_state.json"
    model_dir = "data/model"

    print("\n📊 MIX😌 训练状态\n")

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"  循环状态: {'运行中' if state.get('running') else '已停止'}")
        print(f"  循环次数: {state.get('loopCount', 0)}")
        print(f"  最近训练: {state.get('lastLoopTime', '-')}")
    else:
        print("  循环状态: 未启动")

    config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"\n  模型: {cfg.get('model_size', '?')} ({cfg.get('n_layer', '?')}层, {cfg.get('n_embd', '?')}维)")
    else:
        print("\n  模型: 尚未训练")

    import glob
    raw_count = len(glob.glob("data/raw/*.json"))
    cleaned_count = len(glob.glob("data/cleaned/*.json"))
    print(f"\n  原始数据文件: {raw_count}")
    print(f"  清洗数据文件: {cleaned_count}")
    print()


def cmd_test(args):
    config = load_config()
    from mix_trainer.utils.chat_engine import MixChatEngine
    from mix_trainer.utils.model_tester import ModelTester

    engine = MixChatEngine(model_dir=args.model_dir or "data/model")
    if not engine.load(args.model_file or "model_best.pt"):
        print("❌ 模型加载失败，请先运行 mix train")
        return

    api = config["api"]
    model = config["model"]
    tester = ModelTester(
        api_name=api.get("name", ""),
        api_url=api.get("baseUrl", ""),
        api_key=api.get("apiKey", ""),
        api_model=api.get("model", ""),
        model_name=model["name"],
        short_name=model["shortName"],
    )

    direct_result = tester.run_direct_test(engine)

    api_result = tester.run_api_eval(direct_result)

    combined = {
        "direct_test": direct_result,
        "api_eval": api_result,
    }
    tester.save_test_result(combined)

    name_rate = direct_result.get("name_rate", 0)
    if name_rate >= 80:
        print(f"✅ 测试通过！名称记忆率: {name_rate:.0f}%")
    elif name_rate >= 50:
        print(f"⚠️ 测试一般，名称记忆率: {name_rate:.0f}%，继续训练: mix start")
    else:
        print(f"❌ 测试未通过，名称记忆率: {name_rate:.0f}%，需要更多训练数据")


def cmd_reset(args):
    import shutil
    confirm = input("⚠️ 确定要重置所有训练数据？(yes/no): ").strip()
    if confirm != "yes":
        print("已取消")
        return

    for d in ["data/raw", "data/cleaned", "data/model", "data/logs"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    print("✅ 所有训练数据已重置")


def main():
    parser = argparse.ArgumentParser(
        prog="mix",
        description="MIX😌混合聊天微模型 - 基于Transformer的真实AI训练工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    subparsers.add_parser("init", help="初始化项目，配置API和训练参数")

    config_p = subparsers.add_parser("config", help="配置管理 (show/edit/test)")
    config_p.add_argument("action", nargs="?", default="show", help="show/edit/test")

    gen_p = subparsers.add_parser("generate", help="生成训练数据")
    gen_p.add_argument("--count", type=int, default=1, help="生成批数")

    subparsers.add_parser("clean", help="清洗训练数据")
    subparsers.add_parser("train", help="训练模型")
    subparsers.add_parser("start", help="启动全自动闭环训练")

    chat_p = subparsers.add_parser("chat", help="与微模型聊天")
    chat_p.add_argument("--model_dir", default="data/model")
    chat_p.add_argument("--model_file", default="model_best.pt")

    info_p = subparsers.add_parser("info", help="查看模型训练成果")
    info_p.add_argument("--model_dir", default="data/model")

    test_p = subparsers.add_parser("test", help="API自动测试MixGPT训练效果")
    test_p.add_argument("--model_dir", default="data/model")
    test_p.add_argument("--model_file", default="model_best.pt")

    subparsers.add_parser("status", help="查看训练状态")
    subparsers.add_parser("reset", help="重置所有数据")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "init": cmd_init,
        "config": cmd_config,
        "generate": cmd_generate,
        "clean": cmd_clean,
        "train": cmd_train,
        "start": cmd_start,
        "chat": cmd_chat,
        "info": cmd_info,
        "test": cmd_test,
        "status": cmd_status,
        "reset": cmd_reset,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
