// Extracted rewrite-related fields and methods from QwenASR context.rs
// These are reference code snippets — NOT a standalone compilable file.
// They show how the rewrite decoder is integrated into the QwenCtx engine state.

// ---- Fields added to QwenCtx struct ----
//
// pub struct QwenCtx {
//     ...
//     // Independent rewrite decoder (Qwen3-0.6B / Qwen3-1.7B etc.)
//     pub rewrite_config: Option<QwenConfig>,
//     pub rewrite_decoder: Option<Decoder>,
//     pub rewrite_decoder_int8: Option<DecoderInt8>,
//     pub rewrite_kv_cache: Option<KvCache>,
//     pub rewrite_dec_bufs: Option<DecoderBuffers>,
//     pub rewrite_rope_cache: Option<RopeCache>,
//     pub rewrite_safetensors: Option<MultiSafetensors>,
//     pub rewrite_qint8: Option<QuantFile>,
//     pub rewrite_model_dir: Option<String>,
//
//     // Rewrite GPU state (Metal acceleration for rewrite decoder)
//     #[cfg(feature = "metal")]
//     pub rewrite_gpu_decoder: Option<crate::decoder_gpu::DecoderGpuWeights>,
//     #[cfg(feature = "metal")]
//     pub rewrite_gpu_kv_cache: Option<crate::decoder_gpu::GpuKvCache>,
//     #[cfg(feature = "metal")]
//     pub rewrite_gpu_rope_cache: Option<crate::decoder_gpu::GpuRopeCache>,
//     ...
// }

// ---- Methods ----

impl QwenCtx {
    /// Load an independent LLM (e.g. Qwen3-0.6B) for text rewrite.
    /// Prefers INT8 (model_int8.qint8) if available, falls back to BF16 safetensors.
    /// Returns true on success.
    pub fn load_rewrite_model(&mut self, model_dir: &str) -> bool {
        if kernels::verbose() >= 1 {
            eprintln!("Loading rewrite model from {}", model_dir);
        }

        // Try INT8 first
        let qint8_path = format!("{}/model_int8.qint8", model_dir);
        if let Some(qf) = QuantFile::open(&qint8_path) {
            if kernels::verbose() >= 1 {
                eprintln!("Found rewrite INT8 model: {}", qint8_path);
            }

            // Detect config from qint8 tensor metadata
            let lm_head_shape_i64: Option<Vec<i64>> = qf.find("thinker.lm_head.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let embed_shape_i64: Option<Vec<i64>> = qf.find("thinker.model.embed_tokens.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let gate_shape_i64: Option<Vec<i64>> = qf.find("thinker.model.layers.0.mlp.gate_proj.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let info = crate::config::DetectInfo {
                has_enc_layer_18: false,
                lm_head_shape: lm_head_shape_i64.as_deref(),
                embed_tokens_shape: embed_shape_i64.as_deref(),
                gate_proj_shape: gate_shape_i64.as_deref(),
            };
            let cfg = QwenConfig::detect(&info);

            if kernels::verbose() >= 1 {
                eprintln!("Rewrite model: {}d {}L (INT8)", cfg.dec_hidden, cfg.dec_layers);
            }

            let decoder_int8 = match DecoderInt8::load(&qf, &cfg) {
                Some(d) => d,
                None => {
                    eprintln!("rewrite: failed to load INT8 decoder weights");
                    return false;
                }
            };

            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
            let dec_bufs = DecoderBuffers::new(&cfg);
            let rope_cache = RopeCache::new();

            self.rewrite_config = Some(cfg);
            self.rewrite_decoder = None;
            self.rewrite_decoder_int8 = Some(decoder_int8);
            self.rewrite_kv_cache = Some(kv_cache);
            self.rewrite_dec_bufs = Some(dec_bufs);
            self.rewrite_rope_cache = Some(rope_cache);
            self.rewrite_safetensors = None;
            self.rewrite_qint8 = Some(qf);
            self.rewrite_model_dir = Some(model_dir.to_string());

            if kernels::verbose() >= 1 {
                eprintln!("Rewrite model loaded (INT8).");
            }

            #[cfg(feature = "metal")]
            self.upload_rewrite_to_gpu();

            return true;
        }

        // Fall back to BF16 safetensors
        let ms = match MultiSafetensors::open(model_dir) {
            Some(m) => m,
            None => {
                eprintln!("rewrite: failed to open safetensors in {}", model_dir);
                return false;
            }
        };

        let info = crate::config::DetectInfo {
            has_enc_layer_18: false,
            lm_head_shape: ms.find("lm_head.weight").map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms.find("model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms.find("model.layers.0.mlp.gate_proj.weight").map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model: {}d {}L (BF16)", cfg.dec_hidden, cfg.dec_layers);
        }

        let decoder = match Decoder::load_with_prefix(&ms, &cfg, "") {
            Some(d) => d,
            None => {
                eprintln!("rewrite: failed to load decoder weights");
                return false;
            }
        };

        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
        let dec_bufs = DecoderBuffers::new(&cfg);
        let rope_cache = RopeCache::new();

        self.rewrite_config = Some(cfg);
        self.rewrite_decoder = Some(decoder);
        self.rewrite_decoder_int8 = None;
        self.rewrite_kv_cache = Some(kv_cache);
        self.rewrite_dec_bufs = Some(dec_bufs);
        self.rewrite_rope_cache = Some(rope_cache);
        self.rewrite_safetensors = Some(ms);
        self.rewrite_qint8 = None;
        self.rewrite_model_dir = Some(model_dir.to_string());

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model loaded (BF16).");
        }

        #[cfg(feature = "metal")]
        self.upload_rewrite_to_gpu();

        true
    }

    /// Upload rewrite decoder weights to GPU (Metal).
    #[cfg(feature = "metal")]
    fn upload_rewrite_to_gpu(&mut self) {
        if !self.use_gpu {
            return;
        }
        let dev = match &self.gpu_device {
            crate::device::ComputeDevice::Metal(dev) => dev.clone(),
            _ => return,
        };
        let cfg = match &self.rewrite_config {
            Some(c) => c.clone(),
            None => return,
        };

        if kernels::verbose() >= 1 {
            eprintln!("[metal] Uploading rewrite decoder weights to GPU...");
        }

        let gpu_dec = if let Some(ref d) = self.rewrite_decoder_int8 {
            crate::decoder_gpu::DecoderGpuWeights::from_decoder_int8(d, &cfg, &dev)
        } else if let Some(ref d) = self.rewrite_decoder {
            crate::decoder_gpu::DecoderGpuWeights::from_decoder(d, &cfg, &dev)
        } else {
            return;
        };

        match gpu_dec {
            Ok(gd) => {
                match crate::decoder_gpu::GpuKvCache::new(
                    cfg.dec_layers, cfg.dec_kv_heads, cfg.dec_head_dim, 256, &dev
                ) {
                    Ok(kv) => {
                        let rope = crate::decoder_gpu::GpuRopeCache::new(&dev, cfg.dec_head_dim);
                        self.rewrite_gpu_decoder = Some(gd);
                        self.rewrite_gpu_kv_cache = Some(kv);
                        self.rewrite_gpu_rope_cache = Some(rope);
                        if kernels::verbose() >= 1 {
                            eprintln!("[metal] Rewrite decoder GPU weights + KV cache ready.");
                        }
                    }
                    Err(e) => {
                        eprintln!("[metal] Failed to create rewrite GPU KV cache: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("[metal] Failed to upload rewrite decoder weights: {}, falling back to CPU", e);
            }
        }
    }

    pub fn has_rewrite_model(&self) -> bool {
        self.rewrite_decoder.is_some() || self.rewrite_decoder_int8.is_some()
    }

    pub fn rewrite_cfg(&self) -> &QwenConfig {
        self.rewrite_config.as_ref().expect("rewrite model not loaded")
    }

    pub fn rewrite_tok_embed_to_f32(&self, dst: &mut [f32], token_id: i32, dim: usize) {
        if let Some(ref d) = self.rewrite_decoder_int8 {
            tok_embed_int8_to_f32(dst, &d.tok_embeddings, token_id, dim);
        } else {
            let d = self.rewrite_decoder.as_ref().expect("rewrite decoder not loaded");
            tok_embed_bf16_to_f32(dst, d.tok_embeddings_bf16, token_id, dim);
        }
    }

    /// Prefill rewrite decoder (BF16 or INT8, GPU-accelerated if available).
    pub fn rewrite_prefill(&mut self, input_embeds: &[f32], seq_len: usize) {
        #[cfg(feature = "metal")]
        if self.use_gpu {
            if let (Some(ref gpu_dec), Some(ref mut gpu_kv), Some(ref mut gpu_rope)) =
                (&self.rewrite_gpu_decoder, &mut self.rewrite_gpu_kv_cache, &mut self.rewrite_gpu_rope_cache)
            {
                let cfg = self.rewrite_config.as_ref().unwrap();
                let kv = self.rewrite_kv_cache.as_mut().unwrap();
                let rope = self.rewrite_rope_cache.as_mut().unwrap();
                let bufs = self.rewrite_dec_bufs.as_mut().unwrap();
                crate::decoder_gpu::decoder_prefill_full_gpu(
                    gpu_dec, cfg, gpu_kv, gpu_rope, kv, rope, bufs, input_embeds, seq_len);
                return;
            }
        }

        let cfg = self.rewrite_config.as_ref().unwrap();
        let kv = self.rewrite_kv_cache.as_mut().unwrap();
        let rope = self.rewrite_rope_cache.as_mut().unwrap();
        let bufs = self.rewrite_dec_bufs.as_mut().unwrap();
        if let Some(ref d) = self.rewrite_decoder_int8 {
            decoder_prefill_int8(d, cfg, kv, rope, bufs, input_embeds, seq_len);
        } else {
            let d = self.rewrite_decoder.as_ref().unwrap();
            decoder_prefill(d, cfg, kv, rope, bufs, input_embeds, seq_len);
        }
    }

    /// Single-token forward on rewrite decoder (BF16 or INT8, GPU-accelerated if available).
    pub fn rewrite_forward(&mut self, input_embed: &[f32]) -> i32 {
        #[cfg(feature = "metal")]
        if self.use_gpu {
            if let Some(tok) = self.rewrite_forward_gpu(input_embed) {
                return tok;
            }
        }

        let cfg = self.rewrite_config.as_ref().unwrap();
        let kv = self.rewrite_kv_cache.as_mut().unwrap();
        let rope = self.rewrite_rope_cache.as_mut().unwrap();
        let bufs = self.rewrite_dec_bufs.as_mut().unwrap();
        if let Some(ref d) = self.rewrite_decoder_int8 {
            decoder_forward_int8(d, cfg, kv, rope, bufs, input_embed)
        } else {
            let d = self.rewrite_decoder.as_ref().unwrap();
            decoder_forward(d, cfg, kv, rope, bufs, input_embed)
        }
    }

    #[cfg(feature = "metal")]
    fn rewrite_forward_gpu(&mut self, input_embed: &[f32]) -> Option<i32> {
        let gpu_dec = self.rewrite_gpu_decoder.as_ref()?;
        let gpu_kv = self.rewrite_gpu_kv_cache.as_mut()?;
        let gpu_rope = self.rewrite_gpu_rope_cache.as_mut()?;
        let cfg = self.rewrite_config.as_ref()?;
        let cpu_rope = self.rewrite_rope_cache.as_mut()?;

        match crate::decoder_gpu::decoder_forward_gpu(gpu_dec, cfg, gpu_kv, gpu_rope, cpu_rope, input_embed) {
            Ok(hidden) => {
                let gpu_kv_len = gpu_kv.len;
                let dim = cfg.dec_hidden;
                let eps = cfg.dec_rms_norm_eps;

                let norm_weights = if let Some(ref d) = self.rewrite_decoder_int8 {
                    &d.norm
                } else {
                    &self.rewrite_decoder.as_ref().unwrap().norm
                };
                let mut x_normed = vec![0.0f32; dim];
                kernels::rms_norm(&mut x_normed, &hidden, norm_weights, 1, dim, eps);

                let lm_out_dim = cfg.lm_head_dim();
                let next_token = if let Some(ref d) = self.rewrite_decoder_int8 {
                    let lm = d.lm_head.as_ref().unwrap_or(&d.tok_embeddings);
                    let mut logits = vec![0.0f32; lm_out_dim];
                    kernels::linear_nobias_int8(&mut logits, &x_normed,
                                                lm.data, lm.scales, 1, dim, lm_out_dim);
                    logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0)
                } else {
                    let d = self.rewrite_decoder.as_ref().unwrap();
                    let lm_weight = d.lm_head_bf16.unwrap_or(d.tok_embeddings_bf16);
                    kernels::argmax_matvec_bf16(&x_normed, lm_weight, dim, lm_out_dim) as i32
                };

                if let Some(ref mut kv) = self.rewrite_kv_cache {
                    kv.len = gpu_kv_len;
                }
                Some(next_token)
            }
            Err(e) => {
                eprintln!("[metal] rewrite GPU decode failed: {}, falling back to CPU", e);
                None
            }
        }
    }

    pub fn rewrite_forward_token(&mut self, token_id: i32) -> i32 {
        let dim = self.rewrite_cfg().dec_hidden;
        let mut tmp = vec![0.0f32; dim];
        self.rewrite_tok_embed_to_f32(&mut tmp, token_id, dim);
        self.rewrite_forward(&tmp)
    }

    pub fn reset_rewrite_kv_cache(&mut self) {
        if let Some(ref mut kv) = self.rewrite_kv_cache {
            kv.len = 0;
        }
        #[cfg(feature = "metal")]
        if let Some(ref mut gpu_kv) = self.rewrite_gpu_kv_cache {
            gpu_kv.reset();
        }
    }
}
