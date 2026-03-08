#!/usr/bin/env python3
"""Benchmark text rewrite with V4 prompt across multiple models.

V4 prompt 改进要点:
- 角色弱化: "文本格式化工具" 而非 "润色助手"，减少过度改写
- 新增强约束: "不要回答问题、不要改写句意、不要添加或删除信息"
- 新增 few-shot: 指令式文本保留原意、CI/CD 带斜杠术语、CSC 油箱→邮箱
- 保留 V3 优点: 口语去噪、盘古之白、术语大小写

Uses 100 test cases from test_cases_100.py, with a --quick mode for representative cases.
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]

SYSTEM_PROMPT = """你是一个文本格式化工具。将用户的口语化ASR语音文本转换为规范的书面文本。

规则：
1. 纠正同音错别字（如"油箱→邮箱"、"以经→已经"），去除口语赘词（如"那个"、"呃"）。
2. 根据语意添加标点符号，合理断句。
3. 数字格式化：日期、时间、金额、百分比转阿拉伯数字（三点半→3:30，百分之五→5%）。
4. 中文与英文/数字之间加一个空格。
5. 术语大小写：excel→Excel, chatgpt→ChatGPT, iphone→iPhone, cicd→CI/CD。

重要约束：
- 只做格式修正，严禁改写句意、回答问题或添加/删除信息内容。
- 直接输出处理后的文本，无需任何解释。"""

FEW_SHOT = [
    # 1. 商务: 术语+ITN+标点+口语去噪
    ("那个麻烦帮我把这个excel表格里的数据整理一下然后做成一个ppt汇报",
     "麻烦帮我把这个 Excel 表格里的数据整理一下，然后做成一个 PPT 汇报。"),
    # 2. 指令式文本: 保留原意，只做格式修正（不要去回答！）
    ("请帮我总结一下这篇文章的核心观点不超过三百字",
     "请帮我总结一下这篇文章的核心观点，不超过 300 字。"),
    # 3. CSC + ITN: 同音字纠正 + 数字
    ("我已经把文件发到你油箱了下午两点半之前请查收",
     "我已经把文件发到你邮箱了，下午 2:30 之前请查收。"),
    # 4. 技术术语: CI/CD 带斜杠 + ITN
    ("cicd流水线跑了二十分钟部署到了aws上用的docker和kubernetes",
     "CI/CD 流水线跑了 20 分钟，部署到了 AWS 上，用的 Docker 和 Kubernetes。"),
]

# 从 test_cases_100.py 导入完整 100 条
from test_cases_100 import test_cases as ALL_CASES

# 快速模式: 20 条代表性用例 (覆盖 V3 发现的所有问题 + 各场景能力)
QUICK_INDICES = [
    # 商务: 会议ITN(0), CSC嘛→吗(8), CSC油箱→邮箱(13), 百分比+金额ITN(16), 大数金额(20), 日期(21), CSC以经(29)
    0, 8, 13, 16, 20, 21, 29,
    # 知识: 书名号(30), 术语Transformer(32)
    30, 32,
    # 技术: Excel(50), Python/Bug(55), React/Node(59), iPhone(60), iOS版本(62), QPS/P99(65), CI/CD(67), VSCode(69)
    50, 55, 59, 60, 62, 65, 67, 69,
    # AI交互: 总结文章(70), ChatGPT/Claude(77)
    70, 77,
    # 数字: 百分比(91), Python版本号(95), PyTorch/CUDA(96), 手机号(98)
    91, 95, 96, 98,
]


def parse_expected(desc_str: str) -> str:
    """从 '期望: xxx - 说明' 格式中提取期望输出."""
    if desc_str.startswith("期望: "):
        rest = desc_str[len("期望: "):]
        idx = rest.rfind(" - ")
        if idx > 0:
            return rest[:idx]
    return desc_str


def build_messages(input_text: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_in, ex_out in FEW_SHOT:
        messages.append({"role": "user", "content": ex_in})
        messages.append({"role": "assistant", "content": ex_out})
    messages.append({"role": "user", "content": input_text})
    return messages


def run_model(model_name: str, cases: list[tuple[int, str, str]]):
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    is_qwen3 = "Qwen3" in model_name
    total = len(cases)

    for seq, (case_idx, input_text, desc) in enumerate(cases):
        expected = parse_expected(desc)
        messages = build_messages(input_text)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if is_qwen3 else {}),
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
        test_point = desc.split(" - ")[-1] if " - " in desc else ""
        print(f"\n[{seq+1}/{total}] #{case_idx+1} {test_point}")
        print(f"  输入: {input_text}")
        print(f"  期望: {expected}")
        print(f"  输出: {result}")
        print(f"  状态: {status}  ({elapsed:.2f}s)")

    del model
    del tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark V4 prompt")
    parser.add_argument("--full", action="store_true", help="Run all 100 cases (default: quick)")
    parser.add_argument("--model", type=str, help="Run only a specific model (substring match)")
    args = parser.parse_args()

    if args.full:
        cases = [(i, inp, desc) for i, (inp, desc) in enumerate(ALL_CASES)]
        print(f"Running FULL benchmark: {len(cases)} cases")
    else:
        indices = [i for i in QUICK_INDICES if i < len(ALL_CASES)]
        cases = [(i, ALL_CASES[i][0], ALL_CASES[i][1]) for i in indices]
        print(f"Running QUICK benchmark: {len(cases)} cases (use --full for all 100)")

    models = MODELS
    if args.model:
        models = [m for m in MODELS if args.model.lower() in m.lower()]
        if not models:
            print(f"No model matching '{args.model}' found.")
            return

    for model_name in models:
        try:
            run_model(model_name, cases)
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All done.")


if __name__ == "__main__":
    main()
