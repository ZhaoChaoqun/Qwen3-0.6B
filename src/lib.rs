//! Qwen3 text rewrite engine — C FFI for iOS/macOS integration.
//!
//! Provides text post-processing: ITN + punctuation + CSC + terminology normalization.

#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

pub mod rewrite_ctx;
pub mod text_rewrite;

use rewrite_ctx::RewriteCtx;

/// Opaque handle to the rewrite engine.
pub struct RewriteEngine {
    ctx: RewriteCtx,
}

/// Load rewrite model from a directory path. Returns null on failure.
///
/// `n_threads`: number of CPU threads (0 = auto-detect).
/// `verbosity`: 0 = quiet, 1 = info, 2 = debug.
#[no_mangle]
pub unsafe extern "C" fn qwen3_rewrite_load(
    model_dir: *const c_char,
    n_threads: i32,
    verbosity: i32,
) -> *mut RewriteEngine {
    if model_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    qwen_core::kernels::set_verbose(verbosity);

    let threads = if n_threads <= 0 {
        qwen_core::kernels::get_num_cpus()
    } else {
        n_threads as usize
    };
    qwen_core::kernels::set_threads(threads);

    match RewriteCtx::load(dir) {
        Some(ctx) => Box::into_raw(Box::new(RewriteEngine { ctx })),
        None => std::ptr::null_mut(),
    }
}

/// Rewrite input text: add punctuation, correct errors, normalize numbers.
/// Returns a heap-allocated C string. Caller must free with `qwen3_rewrite_free_string`.
/// Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn qwen3_rewrite_text(
    engine: *mut RewriteEngine,
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

    match text_rewrite::text_rewrite(&mut eng.ctx, text) {
        Some(result) => match CString::new(result) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Enable or disable GPU acceleration (1 = GPU, 0 = CPU-only).
#[no_mangle]
pub unsafe extern "C" fn qwen3_rewrite_set_use_gpu(engine: *mut RewriteEngine, use_gpu: i32) {
    if !engine.is_null() {
        (*engine).ctx.use_gpu = use_gpu != 0;
    }
}

/// Free a string returned by `qwen3_rewrite_text`.
#[no_mangle]
pub unsafe extern "C" fn qwen3_rewrite_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free the engine.
#[no_mangle]
pub unsafe extern "C" fn qwen3_rewrite_free(engine: *mut RewriteEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}
