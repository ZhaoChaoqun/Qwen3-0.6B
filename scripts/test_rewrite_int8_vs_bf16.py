#!/usr/bin/env python3
"""Compare text_rewrite output: INT8 vs BF16 using Qwen3-0.6B."""

import ctypes
import os
import time
from pathlib import Path

DYLIB = Path(__file__).parent.parent / "target/release/libqwen_asr.dylib"
ASR_MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
REWRITE_MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
QINT8_FILE = REWRITE_MODEL_DIR / "model_int8.qint8"
QINT8_FILE_TMP = REWRITE_MODEL_DIR / "model_int8.qint8.bak"

# Import full 100 test cases
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_cases_100 import test_cases as ALL_CASES
TEST_CASES = [(inp, desc) for inp, desc in ALL_CASES]


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


def run_rewrite(lib, engine, label):
    results = []
    total_time = 0
    for i, (input_text, desc) in enumerate(TEST_CASES):
        t0 = time.time()
        ptr = lib.qwen_asr_text_rewrite(engine, input_text.encode("utf-8"))
        elapsed = time.time() - t0
        total_time += elapsed
        result = get_str(lib, ptr)
        results.append(result)
        changed = "+" if result != input_text else "-"
        print(f"  [{i+1:2d}] {changed} ({elapsed:.2f}s) {result}")
    print(f"  Total: {total_time:.2f}s\n")
    return results


def main():
    print(f"dylib: {DYLIB}")
    print(f"ASR:   {ASR_MODEL_DIR}")
    print(f"RW:    {REWRITE_MODEL_DIR}\n")

    lib = setup_lib()

    # ---- BF16 run: temporarily hide qint8 file ----
    print("=" * 80)
    print("Phase 1: BF16 rewrite")
    print("=" * 80)

    has_qint8 = QINT8_FILE.exists()
    if has_qint8:
        os.rename(QINT8_FILE, QINT8_FILE_TMP)
        print(f"  (moved qint8 aside)\n")

    engine = lib.qwen_asr_load_model(str(ASR_MODEL_DIR).encode(), 0, 1)
    if not engine:
        print("ERROR: failed to load ASR model")
        if has_qint8:
            os.rename(QINT8_FILE_TMP, QINT8_FILE)
        return

    ret = lib.qwen_asr_load_rewrite_model(engine, str(REWRITE_MODEL_DIR).encode())
    if ret != 0:
        print("ERROR: failed to load BF16 rewrite model")
        lib.qwen_asr_free(engine)
        if has_qint8:
            os.rename(QINT8_FILE_TMP, QINT8_FILE)
        return

    bf16_results = run_rewrite(lib, engine, "BF16")
    lib.qwen_asr_free(engine)

    # ---- INT8 run: restore qint8 file ----
    print("=" * 80)
    print("Phase 2: INT8 rewrite")
    print("=" * 80)

    if has_qint8:
        os.rename(QINT8_FILE_TMP, QINT8_FILE)
        print(f"  (restored qint8)\n")

    engine = lib.qwen_asr_load_model(str(ASR_MODEL_DIR).encode(), 0, 1)
    if not engine:
        print("ERROR: failed to load ASR model")
        return

    ret = lib.qwen_asr_load_rewrite_model(engine, str(REWRITE_MODEL_DIR).encode())
    if ret != 0:
        print("ERROR: failed to load INT8 rewrite model")
        lib.qwen_asr_free(engine)
        return

    int8_results = run_rewrite(lib, engine, "INT8")
    lib.qwen_asr_free(engine)

    # ---- Comparison ----
    print("=" * 80)
    print("Comparison: BF16 vs INT8")
    print("=" * 80)
    match_count = 0
    diffs = []
    for i, (inp, desc) in enumerate(TEST_CASES):
        bf16 = bf16_results[i]
        int8 = int8_results[i]
        if bf16 == int8:
            match_count += 1
        else:
            diffs.append((i, inp, desc, bf16, int8))

    if diffs:
        print(f"\nDifferences ({len(diffs)}):\n")
        for i, inp, desc, bf16, int8 in diffs:
            short_desc = desc.split(" - ")[-1] if " - " in desc else desc[:30]
            print(f"  #{i+1:3d} {short_desc}")
            print(f"    Input: {inp}")
            print(f"    BF16:  {bf16}")
            print(f"    INT8:  {int8}")
            print()
    else:
        print("\n  All outputs identical!\n")

    print(f"{'='*80}")
    print(f"Match: {match_count}/{len(TEST_CASES)} ({100*match_count/len(TEST_CASES):.0f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
