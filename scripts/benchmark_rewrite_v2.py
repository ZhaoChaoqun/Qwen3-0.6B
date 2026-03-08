#!/usr/bin/env python3
"""Benchmark text rewrite with updated prompt (v2) across multiple models."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]

SYSTEM_PROMPT = """你是资深的AI技术文档编辑助手。请对输入后的ASR语音文本进行重写，遵循以下规则：
1. 修正错别字（如同音字错误）。
2. 添加正确的标点符号。
3. 执行ITN（逆文本正则化）：将中文数字转换为阿拉伯数字（包括日期、金额、IP地址、版本号）。
4. 排版规范：在中文和英文/数字之间必须添加一个空格。
5. 术语规范：正确处理技术术语的大小写（如 java -> Java, github -> GitHub）。
严禁输出任何解释或思考过程，直接输出最终文本即可。"""

FEW_SHOT = [
    ("我今天上午十点去了医院花了三百二十块钱",
     "我今天上午 10 点去了医院，花了 320 块钱。"),
    ("服务器ip地址是一九二点一六八点零点一端口号是八零八零",
     "服务器 IP 地址是 192.168.0.1，端口号是 8080。"),
    ("我想用rust写一个http server处理json数据",
     "我想用 Rust 写一个 HTTP Server，处理 JSON 数据。"),
]

TEST_CASES = [
    # 原有 case
    ("今天下午三点半我们去开会讨论一下关于二零二五年第一季度的销售数据",
     "ITN + 标点: 数字转换 + 添加标点",
     "今天下午 3:30，我们去开会讨论一下关于 2025 年第一季度的销售数据。"),

    ("语音识别道自动驾驶",
     "CSC: '道'→'到'",
     "语音识别到自动驾驶。"),

    ("人工智能正在深刻地改变我们的生活方式从语音识别到自动驾驶从医疗诊断到金融分析",
     "标点: 长句添加标点",
     "人工智能正在深刻地改变我们的生活方式，从语音识别到自动驾驶，从医疗诊断到金融分析。"),

    ("我今天买了三个苹果花了十五块钱",
     "ITN + 标点: 数字 + 标点",
     "我今天买了 3 个苹果，花了 15 块钱。"),

    ("请帮我定一个明天上午九点的会议",
     "CSC + ITN: '定'→'订' + 数字",
     "请帮我订一个明天上午 9 点的会议。"),

    ("他说他明天回在家",
     "CSC: '回'→'会'",
     "他说他明天会在家。"),

    ("这个软件的性能非常号",
     "CSC: '号'→'好'",
     "这个软件的性能非常好。"),

    # 新增: 排版空格
    ("我们用python3开发了一个api接口",
     "排版: 中英空格 + 术语大小写",
     "我们用 Python3 开发了一个 API 接口。"),

    # 新增: 技术术语大小写
    ("这个项目部署在aws上用了docker和kubernetes",
     "术语: 大小写 + 标点",
     "这个项目部署在 AWS 上，用了 Docker 和 Kubernetes。"),

    # 新增: IP/端口 ITN
    ("数据库地址是一零点零点零点一端口三三零六",
     "ITN: IP + 端口",
     "数据库地址是 10.0.0.1，端口 3306。"),

    # 新增: 版本号
    ("我们把node js从十二点一八升级到了二十点零",
     "ITN: 版本号 + 术语",
     "我们把 Node.js 从 12.18 升级到了 20.0。"),

    # 新增: 混合场景
    ("昨天github上有个rust项目拿到了一万两千个star",
     "混合: 术语 + ITN + 标点",
     "昨天 GitHub 上有个 Rust 项目拿到了 12000 个 Star。"),
]


def build_messages(input_text: str, model_name: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_in, ex_out in FEW_SHOT:
        messages.append({"role": "user", "content": ex_in})
        messages.append({"role": "assistant", "content": ex_out})
    messages.append({"role": "user", "content": input_text})
    return messages


def run_model(model_name: str):
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    is_qwen3 = "Qwen3" in model_name

    for i, (input_text, desc, expected) in enumerate(TEST_CASES):
        messages = build_messages(input_text, model_name)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if is_qwen3 else {}),
        )
        inputs = tokenizer([text], return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )
        elapsed = time.time() - t0

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        status = "✓" if result != input_text else "✗"
        print(f"\n[{i+1}/{len(TEST_CASES)}] {desc}")
        print(f"  输入: {input_text}")
        print(f"  期望: {expected}")
        print(f"  输出: {result}")
        print(f"  状态: {status}  ({elapsed:.2f}s)")

    del model
    del tokenizer


def main():
    for model_name in MODELS:
        try:
            run_model(model_name)
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All done.")


if __name__ == "__main__":
    main()
