#!/usr/bin/env python3
"""Benchmark LoRA-finetuned Qwen3-0.6B on full 100 test cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_cases_100 import test_cases as ALL_CASES
from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "你是一个文本格式化工具。将用户的口语化ASR语音文本转换为规范的书面文本。\n\n"
    "规则：\n"
    "1. 纠正同音错别字（如\"油箱→邮箱\"、\"以经→已经\"），去除口语赘词（如\"那个\"、\"呃\"）。\n"
    "2. 根据语意添加标点符号，合理断句。\n"
    "3. 数字格式化：日期、时间、金额、百分比转阿拉伯数字（三点半→3:30，百分之五→5%）。\n"
    "4. 中文与英文/数字之间加一个空格。\n"
    "5. 术语大小写：excel→Excel, chatgpt→ChatGPT, iphone→iPhone, cicd→CI/CD。\n\n"
    "重要约束：\n"
    "- 只做格式修正，严禁改写句意、回答问题或添加/删除信息内容。\n"
    "- 直接输出处理后的文本，无需任何解释。"
)


def parse_expected(desc_str: str) -> str:
    """从 '期望: xxx - 说明' 格式中提取期望输出."""
    if desc_str.startswith("期望: "):
        rest = desc_str[len("期望: "):]
        idx = rest.rfind(" - ")
        if idx > 0:
            return rest[:idx]
    return desc_str


def main():
    print("Loading model + LoRA adapter...")
    model, tokenizer = load("Qwen/Qwen3-0.6B", adapter_path="adapters/rewrite-lora")

    total = len(ALL_CASES)
    changed_count = 0
    unchanged_count = 0

    print(f"\nRunning {total} test cases\n{'='*80}")

    for i, (input_text, desc) in enumerate(ALL_CASES):
        expected = parse_expected(desc)
        test_point = desc.split(" - ")[-1] if " - " in desc else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        result = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)

        changed = result != input_text
        if changed:
            changed_count += 1
        else:
            unchanged_count += 1

        status = "+" if changed else "-"
        print(f"\n[{i+1}/{total}] {status} {test_point}")
        print(f"  输入: {input_text}")
        print(f"  期望: {expected}")
        print(f"  输出: {result}")

    print(f"\n{'='*80}")
    print(f"Results: {changed_count} changed, {unchanged_count} unchanged out of {total}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
