#!/usr/bin/env python3
"""Benchmark LoRA-fused Qwen3-0.6B rewrite model via Rust C API (100 cases)."""

import ctypes
import sys
import time
from pathlib import Path

DYLIB = Path(__file__).parent.parent / "target/release/libqwen_asr.dylib"
ASR_MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
FUSED_MODEL_DIR = Path(__file__).parent.parent / "models/Qwen3-0.6B-rewrite-lora"

sys.path.insert(0, str(Path(__file__).parent))
from test_cases_100 import test_cases as ALL_CASES


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


def parse_expected(desc_str: str) -> str:
    if desc_str.startswith("期望: "):
        rest = desc_str[len("期望: "):]
        idx = rest.rfind(" - ")
        if idx > 0:
            return rest[:idx]
    return desc_str


def main():
    print(f"dylib: {DYLIB}")
    print(f"ASR:   {ASR_MODEL_DIR}")
    print(f"Fused: {FUSED_MODEL_DIR}\n")

    lib = setup_lib()

    # Load ASR model
    engine = lib.qwen_asr_load_model(str(ASR_MODEL_DIR).encode(), 0, 1)
    if not engine:
        print("ERROR: failed to load ASR model")
        return

    # Load fused rewrite model
    ret = lib.qwen_asr_load_rewrite_model(engine, str(FUSED_MODEL_DIR).encode())
    if ret != 0:
        print("ERROR: failed to load fused rewrite model")
        lib.qwen_asr_free(engine)
        return

    total = len(ALL_CASES)
    changed_count = 0
    total_time = 0

    print(f"Running {total} test cases\n{'='*80}")

    for i, (input_text, desc) in enumerate(ALL_CASES):
        expected = parse_expected(desc)
        test_point = desc.split(" - ")[-1] if " - " in desc else ""

        t0 = time.time()
        ptr = lib.qwen_asr_text_rewrite(engine, input_text.encode("utf-8"))
        elapsed = time.time() - t0
        total_time += elapsed

        result = get_str(lib, ptr)
        changed = result != input_text
        if changed:
            changed_count += 1

        status = "+" if changed else "-"
        print(f"\n[{i+1}/{total}] {status} {test_point} ({elapsed:.2f}s)")
        print(f"  输入: {input_text}")
        print(f"  期望: {expected}")
        print(f"  输出: {result}")

    lib.qwen_asr_free(engine)

    print(f"\n{'='*80}")
    print(f"Results: {changed_count} changed, {total - changed_count} unchanged out of {total}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
