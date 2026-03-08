# Qwen3-ASR-0.6B Text Rewrite

基于 Qwen3-0.6B 的 ASR 文本后处理（Text Rewrite）模型，用于将 ASR 输出的口语化文本转换为规范书面文本。

功能包括：
- **ITN（逆文本规范化）**：数字、日期、时间等的格式化（"三点半" → "3:30"）
- **标点恢复**：根据语意添加标点符号
- **同音字纠错（CSC）**：纠正常见的同音错别字（"油箱" → "邮箱"）
- **术语规范化**：大小写修正（"chatgpt" → "ChatGPT"）

## 目录结构

```
scripts/            # Python 评测脚本和 benchmark
docs/               # 评测报告和 LoRA 微调文档
data/rewrite_sft/   # SFT 训练数据（train/valid/test.jsonl）
models/             # 模型权重（BF16 + INT8）— 不入 git
adapters/           # LoRA 适配器 checkpoint — 不入 git
rust-reference/     # Rust 推理代码参考片段（来自 QwenASR 引擎）
```

## 模型

- **基础模型**：Qwen3-0.6B（1024d, 28L）
- **微调方式**：LoRA（r=8, alpha=32, 目标模块 q/k/v/o/gate/up/down proj）
- **融合模型**：`models/Qwen3-0.6B-rewrite-lora/model.safetensors`（BF16, 1.1GB）
- **量化模型**：`models/Qwen3-0.6B-rewrite-lora/model_int8.qint8`（INT8, 570MB）

## 相关项目

- [QwenASR](https://github.com/chaoqunzhao_microsoft/QwenASR) — Rust ASR 推理引擎（本项目的 rewrite 功能原始集成于此）
