// Extracted rewrite C API functions from QwenASR c_api.rs
// This is reference code — NOT a standalone compilable file.
// It shows the C FFI bindings for the rewrite functionality.

use std::ffi::{CStr, CString, c_char};

// ========================================================================
// Text Rewrite API (ITN + Punctuation + CSC)
// ========================================================================

/// Load an independent LLM (e.g. Qwen3-0.6B-Instruct) for text rewrite.
/// `model_dir` should point to a directory containing safetensors + vocab.json.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_load_rewrite_model(
    engine: *mut QwenAsrEngine,
    model_dir: *const c_char,
) -> i32 {
    if engine.is_null() || model_dir.is_null() {
        return -1;
    }
    let eng = &mut *engine;
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    if eng.ctx.load_rewrite_model(dir) {
        0
    } else {
        -1
    }
}

/// Rewrite ASR output text: add punctuation, correct errors, normalize numbers.
/// Requires `qwen_asr_load_rewrite_model` to have been called first.
/// Returns a heap-allocated C string. Caller must free with `qwen_asr_free_string`.
/// Returns null on failure or if engine/text is null.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_text_rewrite(
    engine: *mut QwenAsrEngine,
    input_text: *const c_char,
) -> *mut c_char {
    if engine.is_null() || input_text.is_null() {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let text = match CStr::from_ptr(input_text).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    if text.is_empty() {
        return std::ptr::null_mut();
    }

    match transcribe::text_rewrite(&mut eng.ctx, text) {
        Some(result) => match CString::new(result) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}
