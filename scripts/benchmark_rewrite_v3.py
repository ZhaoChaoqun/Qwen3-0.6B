#!/usr/bin/env python3
"""Benchmark text rewrite with V3 prompt across multiple models.

V3 prompt: 语音输入重写与润色助手 (通用场景, 去口语化, 盘古之白)
Uses 100 test cases from test_cases_100.py, with a --quick mode for 20 representative cases.
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

SYSTEM_PROMPT = """你是专业的语音输入重写与润色助手。你的任务是将用户的口语ASR文本转换为格式完美、逻辑通顺的书面文本。请严格遵循以下规则：

1. **精准纠错**：修正同音错别字，并智能去除口语赘词（如"那个"、"呃"、"然后就是"），保持句子精炼。
2. **标点重构**：根据语意添加准确的标点符号，支持长句的合理断句。
3. **格式规范化 (ITN)**：
   - 将日期、时间、金额、百分比、度量衡转为阿拉伯数字（如：三点半 -> 3:30，百分之五 -> 5%）。
   - 数量词在正式语境下转数字（如：三个方案 -> 3 个方案）。
4. **中西文排版**：在中文与英文/数字之间必须添加一个空格（盘古之白）。
5. **术语大小写**：正确处理常见软件、硬件及技术术语的大小写（如 excel -> Excel, chatgpt -> ChatGPT, iphone -> iPhone）。

严禁输出任何解释、寒暄或思考过程。直接输出处理后的最终文本。"""

FEW_SHOT = [
    # 场景：商务/工具 (Excel, PPT, 数据) + 口语去噪
    ("那个麻烦帮我把这个excel表格里的数据整理一下然后做成一个ppt汇报",
     "麻烦帮我把这个 Excel 表格里的数据整理一下，然后做成一个 PPT 汇报。"),
    # 场景：AI 交互 / 技术概念 (Prompt, API, JSON)
    ("请扮演一个产品经理帮我写一段关于调用chatgpt接口处理json数据的prompt",
     "请扮演一个产品经理，帮我写一段关于调用 ChatGPT 接口处理 JSON 数据的 Prompt。"),
    # 场景：生活/复杂数字 (时间, 金额, App)
    ("我定了明天下午两点半的闹钟要在京东上抢那个三千九百块的显卡",
     "我定了明天下午 2:30 的闹钟，要在京东上抢那个 3900 块的显卡。"),
]

# 从 test_cases_100.py 导入完整 100 条
from test_cases_100 import test_cases as ALL_CASES

# 快速模式: 20 条代表性用例 (覆盖各场景和能力)
QUICK_INDICES = [
    # 商务: 会议日程(0), 邮件CSC(8), 即时通讯CSC(13), 周报ITN(16), 商务大数(20), 日期(21)
    0, 8, 13, 16, 20, 21,
    # 知识: 书名号(30), 术语(32), 引号(37)
    30, 32, 37,
    # 技术: Excel(50), Python/Bug(55), React/Node(59), iPhone(60), iOS版本(62), QPS(65), CI/CD(67), VSCode(69)
    50, 55, 59, 60, 62, 65, 67,
    # AI交互: Prompt(70), ChatGPT(77), RAG(80)
    # (这里补到20条)
    70,
]


def parse_expected(desc_str: str) -> str:
    """从 '期望: xxx - 说明' 格式中提取期望输出."""
    if desc_str.startswith("期望: "):
        rest = desc_str[len("期望: "):]
        # 找最后一个 " - " 作为分隔符
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
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
        # 提取描述中的测试点说明
        test_point = desc.split(" - ")[-1] if " - " in desc else ""
        print(f"\n[{seq+1}/{total}] #{case_idx+1} {test_point}")
        print(f"  输入: {input_text}")
        print(f"  期望: {expected}")
        print(f"  输出: {result}")
        print(f"  状态: {status}  ({elapsed:.2f}s)")

    del model
    del tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark V3 prompt")
    parser.add_argument("--full", action="store_true", help="Run all 100 cases (default: 20 quick cases)")
    parser.add_argument("--model", type=str, help="Run only a specific model (substring match)")
    args = parser.parse_args()

    if args.full:
        cases = [(i, inp, desc) for i, (inp, desc) in enumerate(ALL_CASES)]
        print(f"Running FULL benchmark: {len(cases)} cases")
    else:
        cases = [(i, ALL_CASES[i][0], ALL_CASES[i][1]) for i in QUICK_INDICES if i < len(ALL_CASES)]
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
