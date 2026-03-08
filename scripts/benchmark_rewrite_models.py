#!/usr/bin/env python3
"""Benchmark text rewrite (ITN + Punctuation + CSC) across multiple models using transformers."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]

SYSTEM_PROMPT = "你是语音识别后处理助手。对输入文本：1)添加标点符号 2)修正错别字(如同音字错误) 3)将中文数字转为阿拉伯数字。只输出处理后的文本，不做任何解释。"

FEW_SHOT = [
    ("我今天上午十点去了医院花了三百二十块钱", "我今天上午10点去了医院，花了320块钱。"),
    ("他说这个项目的完成率以经达到了百分之九十五", "他说这个项目的完成率已经达到了95%。"),
]

TEST_CASES = [
    ("今天下午三点半我们去开会讨论一下关于二零二五年第一季度的销售数据",
     "ITN + 标点: 数字转换 + 添加标点",
     "今天下午3:30，我们去开会讨论一下关于2025年第一季度的销售数据。"),

    ("语音识别道自动驾驶",
     "CSC: '道'→'到'",
     "语音识别到自动驾驶。"),

    ("人工智能正在深刻地改变我们的生活方式从语音识别到自动驾驶从医疗诊断到金融分析",
     "标点: 长句添加标点",
     "人工智能正在深刻地改变我们的生活方式，从语音识别到自动驾驶，从医疗诊断到金融分析。"),

    ("我今天买了三个苹果花了十五块钱",
     "ITN + 标点: 数字 + 标点",
     "我今天买了3个苹果，花了15块钱。"),

    ("请帮我定一个明天上午九点的会议",
     "CSC + ITN: '定'→'订' + 数字",
     "请帮我订一个明天上午9点的会议。"),

    ("他说他明天回在家",
     "CSC: '回'→'会'",
     "他说他明天会在家。"),

    ("这个软件的性能非常号",
     "CSC: '号'→'好'",
     "这个软件的性能非常好。"),
]


def build_messages(input_text: str, model_name: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_in, ex_out in FEW_SHOT:
        messages.append({"role": "user", "content": ex_in})
        messages.append({"role": "assistant", "content": ex_out})

    # Qwen3 needs /no_think to disable thinking mode
    if "Qwen3" in model_name:
        messages.append({"role": "user", "content": f"{input_text} /no_think"})
    else:
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

    for i, (input_text, desc, expected) in enumerate(TEST_CASES):
        messages = build_messages(input_text, model_name)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if "Qwen3" in model_name else {}),
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

        # Extract only newly generated tokens
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

    print(f"\n{'='*80}")
    print("All done.")


if __name__ == "__main__":
    main()
