#!/usr/bin/env python3
"""Test text_rewrite API: load independent Qwen3-0.6B for ITN + punctuation + CSC."""

import ctypes
import time
from pathlib import Path

DYLIB = Path(__file__).parent.parent / "target/release/libqwen_asr.dylib"
ASR_MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
REWRITE_MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def setup_lib():
    lib = ctypes.cdll.LoadLibrary(str(DYLIB))
    lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    lib.qwen_asr_load_model.restype = ctypes.c_void_p
    lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free.restype = None
    lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free_string.restype = None
    lib.qwen_asr_load_rewrite_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_load_rewrite_model.restype = ctypes.c_int32
    lib.qwen_asr_text_rewrite.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_text_rewrite.restype = ctypes.c_void_p
    return lib


def get_str(lib, ptr):
    if not ptr:
        return ""
    s = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
    lib.qwen_asr_free_string(ptr)
    return s


def main():
    print(f"dylib: {DYLIB}")
    print(f"ASR model: {ASR_MODEL_DIR}")
    print(f"Rewrite model: {REWRITE_MODEL_DIR}\n")

    lib = setup_lib()

    # Load ASR engine
    engine = lib.qwen_asr_load_model(str(ASR_MODEL_DIR).encode(), 0, 1)
    if not engine:
        print("ERROR: failed to load ASR model")
        return

    # Load independent rewrite model
    ret = lib.qwen_asr_load_rewrite_model(engine, str(REWRITE_MODEL_DIR).encode())
    if ret != 0:
        print("ERROR: failed to load rewrite model")
        lib.qwen_asr_free(engine)
        return

    test_cases = [
        # (input, description)
        ("今天下午三点半我们去开会讨论一下关于二零二五年第一季度的销售数据",
         "ITN + 标点: 数字转换 + 添加标点"),

        ("语音识别道自动驾驶",
         "CSC: 纠正'道'→'到'"),

        ("人工智能正在深刻地改变我们的生活方式从语音识别到自动驾驶从医疗诊断到金融分析",
         "标点: 长句添加标点"),

        ("我今天买了三个苹果花了十五块钱",
         "ITN + 标点: 数字 + 标点"),

        ("请帮我定一个明天上午九点的会议",
         "CSC + ITN: '定'→'订' + 数字"),

        ("他说他明天回在家",
         "CSC: '回'→'会'"),

        ("这个软件的性能非常号",
         "CSC: '号'→'好'"),
    ]

    print(f"{'='*80}")
    print(f"Text Rewrite 测试 (Independent Qwen3-0.6B)")
    print(f"{'='*80}\n")

    for i, (input_text, desc) in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {desc}")
        print(f"  输入: {input_text}")

        t0 = time.time()
        ptr = lib.qwen_asr_text_rewrite(engine, input_text.encode("utf-8"))
        elapsed = time.time() - t0
        result = get_str(lib, ptr)

        if result:
            print(f"  输出: {result}")
            changed = "✓ 有改动" if result != input_text else "✗ 无改动"
            print(f"  状态: {changed}  ({elapsed:.2f}s)")
        else:
            print(f"  输出: (null / 失败)")
        print()

    lib.qwen_asr_free(engine)
    print("Done.")


if __name__ == "__main__":
    main()
