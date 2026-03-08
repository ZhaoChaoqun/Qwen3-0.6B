//! Rewrite engine context — a lightweight decoder-only context for text rewrite.

use qwen_core::config::{QwenConfig, DetectInfo};
use qwen_core::decoder::*;
use qwen_core::quantize::QuantFile;
use qwen_core::safetensors::MultiSafetensors;
use qwen_core::kernels;

pub struct RewriteCtx {
    pub config: QwenConfig,
    pub decoder: Option<Decoder>,
    pub decoder_int8: Option<DecoderInt8>,
    pub kv_cache: KvCache,
    pub dec_bufs: DecoderBuffers,
    pub rope_cache: RopeCache,
    pub _safetensors: Option<MultiSafetensors>,
    pub _qint8: Option<QuantFile>,
    pub model_dir: String,
    pub use_gpu: bool,

    #[cfg(feature = "metal")]
    pub gpu_device: qwen_core::device::ComputeDevice,
    #[cfg(feature = "metal")]
    pub gpu_decoder: Option<qwen_core::decoder_gpu::DecoderGpuWeights>,
    #[cfg(feature = "metal")]
    pub gpu_kv_cache: Option<qwen_core::decoder_gpu::GpuKvCache>,
    #[cfg(feature = "metal")]
    pub gpu_rope_cache: Option<qwen_core::decoder_gpu::GpuRopeCache>,
}

impl RewriteCtx {
    /// Load a rewrite model from directory. Prefers INT8 if available.
    pub fn load(model_dir: &str) -> Option<Self> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading rewrite model from {}", model_dir);
        }

        // Try INT8 first
        let qint8_path = format!("{}/model_int8.qint8", model_dir);
        if let Some(qf) = QuantFile::open(&qint8_path) {
            return Self::load_int8(qf, model_dir);
        }

        // Fall back to BF16 safetensors
        let ms = MultiSafetensors::open(model_dir)?;
        Self::load_bf16(ms, model_dir)
    }

    fn load_int8(qf: QuantFile, model_dir: &str) -> Option<Self> {
        if kernels::verbose() >= 1 {
            eprintln!("Found rewrite INT8 model");
        }

        let lm_head_shape: Option<Vec<i64>> = qf.find("thinker.lm_head.weight")
            .map(|t| t.shape.iter().map(|&s| s as i64).collect());
        let embed_shape: Option<Vec<i64>> = qf.find("thinker.model.embed_tokens.weight")
            .map(|t| t.shape.iter().map(|&s| s as i64).collect());
        let gate_shape: Option<Vec<i64>> = qf.find("thinker.model.layers.0.mlp.gate_proj.weight")
            .map(|t| t.shape.iter().map(|&s| s as i64).collect());
        let info = DetectInfo {
            has_enc_layer_18: false,
            lm_head_shape: lm_head_shape.as_deref(),
            embed_tokens_shape: embed_shape.as_deref(),
            gate_proj_shape: gate_shape.as_deref(),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model: {}d {}L (INT8)", cfg.dec_hidden, cfg.dec_layers);
        }

        let decoder_int8 = DecoderInt8::load(&qf, &cfg)?;

        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
        let dec_bufs = DecoderBuffers::new(&cfg);
        let rope_cache = RopeCache::new();

        let mut ctx = RewriteCtx {
            config: cfg,
            decoder: None,
            decoder_int8: Some(decoder_int8),
            kv_cache,
            dec_bufs,
            rope_cache,
            _safetensors: None,
            _qint8: Some(qf),
            model_dir: model_dir.to_string(),
            use_gpu: true,
            #[cfg(feature = "metal")]
            gpu_device: qwen_core::device::ComputeDevice::best(),
            #[cfg(feature = "metal")]
            gpu_decoder: None,
            #[cfg(feature = "metal")]
            gpu_kv_cache: None,
            #[cfg(feature = "metal")]
            gpu_rope_cache: None,
        };

        #[cfg(feature = "metal")]
        ctx.upload_to_gpu();

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model loaded (INT8).");
        }

        Some(ctx)
    }

    fn load_bf16(ms: MultiSafetensors, model_dir: &str) -> Option<Self> {
        let info = DetectInfo {
            has_enc_layer_18: false,
            lm_head_shape: ms.find("lm_head.weight").map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms.find("model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms.find("model.layers.0.mlp.gate_proj.weight").map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model: {}d {}L (BF16)", cfg.dec_hidden, cfg.dec_layers);
        }

        let decoder = Decoder::load_with_prefix(&ms, &cfg, "")?;

        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
        let dec_bufs = DecoderBuffers::new(&cfg);
        let rope_cache = RopeCache::new();

        let mut ctx = RewriteCtx {
            config: cfg,
            decoder: Some(decoder),
            decoder_int8: None,
            kv_cache,
            dec_bufs,
            rope_cache,
            _safetensors: Some(ms),
            _qint8: None,
            model_dir: model_dir.to_string(),
            use_gpu: true,
            #[cfg(feature = "metal")]
            gpu_device: qwen_core::device::ComputeDevice::best(),
            #[cfg(feature = "metal")]
            gpu_decoder: None,
            #[cfg(feature = "metal")]
            gpu_kv_cache: None,
            #[cfg(feature = "metal")]
            gpu_rope_cache: None,
        };

        #[cfg(feature = "metal")]
        ctx.upload_to_gpu();

        if kernels::verbose() >= 1 {
            eprintln!("Rewrite model loaded (BF16).");
        }

        Some(ctx)
    }

    #[cfg(feature = "metal")]
    fn upload_to_gpu(&mut self) {
        if !self.use_gpu {
            return;
        }
        let dev = match &self.gpu_device {
            qwen_core::device::ComputeDevice::Metal(dev) => dev.clone(),
            _ => return,
        };
        let cfg = self.config.clone();

        if kernels::verbose() >= 1 {
            eprintln!("[metal] Uploading rewrite decoder weights to GPU...");
        }

        let gpu_dec = if let Some(ref d) = self.decoder_int8 {
            qwen_core::decoder_gpu::DecoderGpuWeights::from_decoder_int8(d, &cfg, &dev)
        } else if let Some(ref d) = self.decoder {
            qwen_core::decoder_gpu::DecoderGpuWeights::from_decoder(d, &cfg, &dev)
        } else {
            return;
        };

        match gpu_dec {
            Ok(gd) => {
                match qwen_core::decoder_gpu::GpuKvCache::new(
                    cfg.dec_layers, cfg.dec_kv_heads, cfg.dec_head_dim, 256, &dev
                ) {
                    Ok(kv) => {
                        let rope = qwen_core::decoder_gpu::GpuRopeCache::new(&dev, cfg.dec_head_dim);
                        self.gpu_decoder = Some(gd);
                        self.gpu_kv_cache = Some(kv);
                        self.gpu_rope_cache = Some(rope);
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
                eprintln!("[metal] Failed to upload rewrite decoder weights: {}", e);
            }
        }
    }

    pub fn tok_embed_to_f32(&self, dst: &mut [f32], token_id: i32, dim: usize) {
        if let Some(ref d) = self.decoder_int8 {
            tok_embed_int8_to_f32(dst, &d.tok_embeddings, token_id, dim);
        } else {
            let d = self.decoder.as_ref().expect("decoder not loaded");
            tok_embed_bf16_to_f32(dst, d.tok_embeddings_bf16, token_id, dim);
        }
    }

    pub fn prefill(&mut self, input_embeds: &[f32], seq_len: usize) {
        #[cfg(feature = "metal")]
        if self.use_gpu {
            if let (Some(ref gpu_dec), Some(ref mut gpu_kv), Some(ref mut gpu_rope)) =
                (&self.gpu_decoder, &mut self.gpu_kv_cache, &mut self.gpu_rope_cache)
            {
                let cfg = &self.config;
                let kv = &mut self.kv_cache;
                let rope = &mut self.rope_cache;
                let bufs = &mut self.dec_bufs;
                qwen_core::decoder_gpu::decoder_prefill_full_gpu(
                    gpu_dec, cfg, gpu_kv, gpu_rope, kv, rope, bufs, input_embeds, seq_len);
                return;
            }
        }

        if let Some(ref d) = self.decoder_int8 {
            decoder_prefill_int8(d, &self.config, &mut self.kv_cache, &mut self.rope_cache,
                                 &mut self.dec_bufs, input_embeds, seq_len);
        } else {
            let d = self.decoder.as_ref().unwrap();
            decoder_prefill(d, &self.config, &mut self.kv_cache, &mut self.rope_cache,
                           &mut self.dec_bufs, input_embeds, seq_len);
        }
    }

    pub fn forward(&mut self, input_embed: &[f32]) -> i32 {
        #[cfg(feature = "metal")]
        if self.use_gpu {
            if let Some(tok) = self.forward_gpu(input_embed) {
                return tok;
            }
        }

        if let Some(ref d) = self.decoder_int8 {
            decoder_forward_int8(d, &self.config, &mut self.kv_cache, &mut self.rope_cache,
                                 &mut self.dec_bufs, input_embed)
        } else {
            let d = self.decoder.as_ref().unwrap();
            decoder_forward(d, &self.config, &mut self.kv_cache, &mut self.rope_cache,
                           &mut self.dec_bufs, input_embed)
        }
    }

    #[cfg(feature = "metal")]
    fn forward_gpu(&mut self, input_embed: &[f32]) -> Option<i32> {
        let gpu_dec = self.gpu_decoder.as_ref()?;
        let gpu_kv = self.gpu_kv_cache.as_mut()?;
        let gpu_rope = self.gpu_rope_cache.as_mut()?;
        let cfg = &self.config;
        let cpu_rope = &mut self.rope_cache;

        match qwen_core::decoder_gpu::decoder_forward_gpu(gpu_dec, cfg, gpu_kv, gpu_rope, cpu_rope, input_embed) {
            Ok(hidden) => {
                let gpu_kv_len = gpu_kv.len;
                let dim = cfg.dec_hidden;
                let eps = cfg.dec_rms_norm_eps;

                let norm_weights = if let Some(ref d) = self.decoder_int8 {
                    &d.norm
                } else {
                    &self.decoder.as_ref().unwrap().norm
                };
                let mut x_normed = vec![0.0f32; dim];
                kernels::rms_norm(&mut x_normed, &hidden, norm_weights, 1, dim, eps);

                let lm_out_dim = cfg.lm_head_dim();
                let next_token = if let Some(ref d) = self.decoder_int8 {
                    let lm = d.lm_head.as_ref().unwrap_or(&d.tok_embeddings);
                    let mut logits = vec![0.0f32; lm_out_dim];
                    kernels::linear_nobias_int8(&mut logits, &x_normed,
                                                lm.data, lm.scales, 1, dim, lm_out_dim);
                    logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0)
                } else {
                    let d = self.decoder.as_ref().unwrap();
                    let lm_weight = d.lm_head_bf16.unwrap_or(d.tok_embeddings_bf16);
                    kernels::argmax_matvec_bf16(&x_normed, lm_weight, dim, lm_out_dim) as i32
                };

                self.kv_cache.len = gpu_kv_len;
                Some(next_token)
            }
            Err(e) => {
                eprintln!("[metal] rewrite GPU decode failed: {}, falling back to CPU", e);
                None
            }
        }
    }

    pub fn forward_token(&mut self, token_id: i32) -> i32 {
        let dim = self.config.dec_hidden;
        let mut tmp = vec![0.0f32; dim];
        self.tok_embed_to_f32(&mut tmp, token_id, dim);
        self.forward(&tmp)
    }

    pub fn reset_kv_cache(&mut self) {
        self.kv_cache.len = 0;
        #[cfg(feature = "metal")]
        if let Some(ref mut gpu_kv) = self.gpu_kv_cache {
            gpu_kv.reset();
        }
    }
}
