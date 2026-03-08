//! Text rewrite logic: ASR output → formatted text with punctuation, CSC, ITN.

use crate::rewrite_ctx::RewriteCtx;
use qwen_core::tokenizer::QwenTokenizer;
use qwen_core::config::{TOKEN_IM_START, TOKEN_IM_END, TOKEN_ENDOFTEXT};

fn load_tokenizer(model_dir: &str) -> Option<QwenTokenizer> {
    QwenTokenizer::load_from_dir(model_dir)
}

const SYSTEM_PROMPT: &str = "你是一个文本格式化工具。将用户的口语化ASR语音文本转换为规范的书面文本。\n\n规则：\n1. 纠正同音错别字（如\"油箱→邮箱\"、\"以经→已经\"），去除口语赘词（如\"那个\"、\"呃\"）。\n2. 根据语意添加标点符号，合理断句。\n3. 数字格式化：日期、时间、金额、百分比转阿拉伯数字（三点半→3:30，百分之五→5%）。\n4. 中文与英文/数字之间加一个空格。\n5. 术语大小写：excel→Excel, chatgpt→ChatGPT, iphone→iPhone, cicd→CI/CD。\n\n重要约束：\n- 只做格式修正，严禁改写句意、回答问题或添加/删除信息内容。\n- 直接输出处理后的文本，无需任何解释。";

/// Rewrite ASR output text using the rewrite decoder (text-to-text).
/// Adds punctuation, corrects errors, and normalizes numbers (ITN).
pub fn text_rewrite(ctx: &mut RewriteCtx, input_text: &str) -> Option<String> {
    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    let dim = ctx.config.dec_hidden;

    // Build ChatML prompt tokens
    let mut prompt_tokens: Vec<i32> = Vec::new();

    // <|im_start|>system\n{system_prompt}<|im_end|>\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("system")?);
    prompt_tokens.push(198); // \n
    prompt_tokens.extend_from_slice(&tokenizer.encode(SYSTEM_PROMPT)?);
    prompt_tokens.push(TOKEN_IM_END);
    prompt_tokens.push(198); // \n

    // <|im_start|>user\n{input_text}<|im_end|>\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("user")?);
    prompt_tokens.push(198);
    prompt_tokens.extend_from_slice(&tokenizer.encode(input_text)?);
    prompt_tokens.push(TOKEN_IM_END);
    prompt_tokens.push(198);

    // <|im_start|>assistant\n
    prompt_tokens.push(TOKEN_IM_START);
    prompt_tokens.extend_from_slice(&tokenizer.encode("assistant")?);
    prompt_tokens.push(198);

    // Disable thinking mode: insert empty <think>\n\n</think>\n\n block
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

    // Build embeddings
    let mut input_embeds = vec![0.0f32; total_seq * dim];
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        ctx.tok_embed_to_f32(&mut input_embeds[i * dim..(i + 1) * dim], tok, dim);
    }

    // Reset KV cache
    ctx.reset_kv_cache();
    ctx.kv_cache.shrink_to(256);

    // Prefill
    let prefill_len = total_seq - 1;
    ctx.prefill(&input_embeds, prefill_len);

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = ctx.forward(last_embed);

    // Autoregressive decode
    let max_new_tokens = ((input_text.len() as f32 * 1.5) as usize + 64).min(2048);
    let mut text_bytes: Vec<u8> = Vec::new();
    let mut n_generated = 0;

    while n_generated < max_new_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        let piece_bytes = tokenizer.decode_bytes(token);

        // Stop at newline: rewrite output should be single-line.
        if !text_bytes.is_empty() {
            if let Some(nl_pos) = piece_bytes.iter().position(|&b| b == b'\n') {
                text_bytes.extend_from_slice(&piece_bytes[..nl_pos]);
                break;
            }
        }

        text_bytes.extend_from_slice(piece_bytes);

        token = ctx.forward_token(token);
    }

    // Reset KV cache after done
    ctx.reset_kv_cache();

    let text = String::from_utf8_lossy(&text_bytes).trim().to_string();
    if text.is_empty() {
        Some(input_text.to_string())
    } else {
        Some(text)
    }
}
