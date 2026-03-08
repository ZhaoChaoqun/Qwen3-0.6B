// Extracted text_rewrite() function from QwenASR transcribe.rs
// This is reference code — NOT a standalone compilable file.
// It shows how the rewrite decoder is used for ASR post-processing.

use crate::context::QwenCtx;

// Token constants used by the rewrite function:
// const TOKEN_IM_START: i32 = 151644;  // <|im_start|>
// const TOKEN_IM_END: i32 = 151645;    // <|im_end|>
// const TOKEN_ENDOFTEXT: i32 = 151643; // <|endoftext|>

/// Rewrite ASR output text using an independent LLM decoder (text-to-text).
/// Adds punctuation, corrects errors, and normalizes numbers (ITN).
///
/// Requires `ctx.load_rewrite_model()` to have been called first.
/// Falls back to returning the original text if rewrite model is not loaded.
pub fn text_rewrite(ctx: &mut QwenCtx, input_text: &str) -> Option<String> {
    if !ctx.has_rewrite_model() {
        eprintln!("text_rewrite: rewrite model not loaded, returning original text");
        return Some(input_text.to_string());
    }

    // Load tokenizer from rewrite model dir (has the same vocab)
    let rw_dir = ctx.rewrite_model_dir.as_ref().unwrap();
    let tokenizer = load_tokenizer(rw_dir)
        .or_else(|| load_tokenizer(&ctx.model_dir))?;
    let dim = ctx.rewrite_cfg().dec_hidden;

    // Build ChatML prompt tokens:
    // <|im_start|>system\n{system_prompt}<|im_end|>\n
    // Few-shot examples as user/assistant turns
    // <|im_start|>user\n{input_text} /no_think<|im_end|>\n
    // <|im_start|>assistant\n
    let system_prompt = "你是一个文本格式化工具。将用户的口语化ASR语音文本转换为规范的书面文本。\n\n规则：\n1. 纠正同音错别字（如\"油箱→邮箱\"、\"以经→已经\"），去除口语赘词（如\"那个\"、\"呃\"）。\n2. 根据语意添加标点符号，合理断句。\n3. 数字格式化：日期、时间、金额、百分比转阿拉伯数字（三点半→3:30，百分之五→5%）。\n4. 中文与英文/数字之间加一个空格。\n5. 术语大小写：excel→Excel, chatgpt→ChatGPT, iphone→iPhone, cicd→CI/CD。\n\n重要约束：\n- 只做格式修正，严禁改写句意、回答问题或添加/删除信息内容。\n- 直接输出处理后的文本，无需任何解释。";

    let few_shot_examples: &[(&str, &str)] = &[];

    let mut prompt_tokens: Vec<i32> = Vec::new();

    // <|im_start|>system\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("system")?);
    prompt_tokens.push(198); // \n

    // system prompt text
    prompt_tokens.extend_from_slice(&tokenizer.encode(system_prompt)?);

    // <|im_end|>\n
    prompt_tokens.push(TOKEN_IM_END);
    prompt_tokens.push(198); // \n

    // Few-shot examples
    for &(example_in, example_out) in few_shot_examples {
        // <|im_start|>user\n{example_in}<|im_end|>\n
        prompt_tokens.push(TOKEN_IM_START);
        prompt_tokens.extend_from_slice(&tokenizer.encode("user")?);
        prompt_tokens.push(198);
        prompt_tokens.extend_from_slice(&tokenizer.encode(example_in)?);
        prompt_tokens.push(TOKEN_IM_END);
        prompt_tokens.push(198);

        // <|im_start|>assistant\n{example_out}<|im_end|>\n
        prompt_tokens.push(TOKEN_IM_START);
        prompt_tokens.extend_from_slice(&tokenizer.encode("assistant")?);
        prompt_tokens.push(198);
        prompt_tokens.extend_from_slice(&tokenizer.encode(example_out)?);
        prompt_tokens.push(TOKEN_IM_END);
        prompt_tokens.push(198);
    }

    // <|im_start|>user\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("user")?);
    prompt_tokens.push(198); // \n

    // user input text (no /no_think suffix; thinking disabled via empty <think> block)
    let user_text = input_text;
    prompt_tokens.extend_from_slice(&tokenizer.encode(&user_text)?);

    // <|im_end|>\n
    prompt_tokens.push(TOKEN_IM_END);
    prompt_tokens.push(198); // \n

    // <|im_start|>assistant\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("assistant")?);
    prompt_tokens.push(198); // \n

    // Disable thinking mode: insert empty <think>\n\n</think>\n\n block
    // (matches enable_thinking=False in Qwen3 chat template used during LoRA training)
    prompt_tokens.push(151667); // <think>
    prompt_tokens.push(198);    // \n
    prompt_tokens.push(198);    // \n
    prompt_tokens.push(151668); // </think>
    prompt_tokens.push(198);    // \n
    prompt_tokens.push(198);    // \n

    let total_seq = prompt_tokens.len();
    if total_seq == 0 {
        return Some(input_text.to_string());
    }

    // Build embeddings from token IDs using rewrite decoder
    let mut input_embeds = vec![0.0f32; total_seq * dim];
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        ctx.rewrite_tok_embed_to_f32(&mut input_embeds[i * dim..(i + 1) * dim], tok, dim);
    }

    // Reset rewrite KV cache
    ctx.reset_rewrite_kv_cache();
    if let Some(ref mut kv) = ctx.rewrite_kv_cache {
        kv.shrink_to(256);
    }

    // Prefill
    let prefill_len = total_seq - 1;
    ctx.rewrite_prefill(&input_embeds, prefill_len);

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = ctx.rewrite_forward(last_embed);

    // Autoregressive decode
    let max_new_tokens = (input_text.len() as f32 * 1.5) as usize + 64;
    let max_new_tokens = max_new_tokens.min(2048);
    let mut text_bytes: Vec<u8> = Vec::new();
    let mut n_generated = 0;

    while n_generated < max_new_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        let piece_bytes = tokenizer.decode_bytes(token);

        // Stop if this token contains a newline: rewrite output should be single-line text.
        // After the correct output the model may hallucinate (e.g. Markdown image links).
        // Keep bytes before the newline (e.g. "。\n" → keep "。").
        if !text_bytes.is_empty() {
            if let Some(nl_pos) = piece_bytes.iter().position(|&b| b == b'\n') {
                text_bytes.extend_from_slice(&piece_bytes[..nl_pos]);
                break;
            }
        }

        text_bytes.extend_from_slice(piece_bytes);

        token = ctx.rewrite_forward_token(token);
    }

    // Reset rewrite KV cache after done
    ctx.reset_rewrite_kv_cache();

    let text = String::from_utf8_lossy(&text_bytes).trim().to_string();
    if text.is_empty() {
        Some(input_text.to_string())
    } else {
        Some(text)
    }
}
