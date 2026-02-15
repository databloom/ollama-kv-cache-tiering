//! FFI bindings to llama.cpp.
//!
//! This module defines the C FFI interface for interacting with llama.cpp.
//! The actual linking is handled by build.rs which compiles llama.cpp from source.
//!
//! For the initial implementation, we use a mock/stub that simulates
//! llama.cpp behavior for integration testing without requiring the C library.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    #[error("Tokenization failed: {0}")]
    TokenizeFailed(String),

    #[error("Decode failed: {0}")]
    DecodeFailed(String),

    #[error("Context creation failed: {0}")]
    ContextFailed(String),
}

/// Token ID type.
pub type TokenId = i32;

/// Model parameters (mirrors llama_model_params).
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Number of GPU layers to offload.
    pub n_gpu_layers: i32,

    /// Use memory mapping for the model file.
    pub use_mmap: bool,

    /// Use memory locking.
    pub use_mlock: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1, // all layers
            use_mmap: true,
            use_mlock: false,
        }
    }
}

/// Context parameters (mirrors llama_context_params).
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// Context size in tokens.
    pub n_ctx: u32,

    /// Batch size for prompt processing.
    pub n_batch: u32,

    /// Number of threads for computation.
    pub n_threads: u32,

    /// Use flash attention.
    pub flash_attn: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 32768,
            n_batch: 512,
            n_threads: 4,
            flash_attn: true,
        }
    }
}

/// Stub model handle.
///
/// In a real implementation, this would wrap `*mut llama_model` from the C library.
pub struct LlamaModel {
    /// Model file path.
    pub path: String,

    /// Vocabulary size.
    pub n_vocab: usize,

    /// Number of layers.
    pub n_layers: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Number of KV heads (for GQA).
    pub n_kv_heads: usize,

    /// Hidden dimension per head.
    pub head_dim: usize,
}

/// Stub context handle.
///
/// In a real implementation, this would wrap `*mut llama_context`.
pub struct LlamaContext {
    /// Context size.
    pub n_ctx: u32,

    /// Current token position.
    pub pos: usize,
}

impl LlamaModel {
    /// Load a model from a GGUF file (stub).
    pub fn load(path: &str, _params: ModelParams) -> Result<Self, LlamaError> {
        // Stub: in real implementation, calls llama_load_model_from_file.
        Ok(Self {
            path: path.to_string(),
            n_vocab: 152064,  // Qwen2.5 vocab size
            n_layers: 40,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
        })
    }

    /// Create a new context for this model (stub).
    pub fn new_context(&self, _params: ContextParams) -> Result<LlamaContext, LlamaError> {
        Ok(LlamaContext {
            n_ctx: 32768,
            pos: 0,
        })
    }

    /// Tokenize a string into token IDs (stub).
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<TokenId>, LlamaError> {
        // Stub: produce approximately 1 token per 4 characters.
        let n_tokens = (text.len() / 4).max(1);
        let mut tokens: Vec<TokenId> = (0..n_tokens as i32).collect();
        if add_bos {
            tokens.insert(0, 1); // BOS token
        }
        Ok(tokens)
    }

    /// Decode tokens back to text (stub).
    pub fn detokenize(&self, tokens: &[TokenId]) -> Result<String, LlamaError> {
        // Stub: return placeholder text.
        Ok(format!("[decoded {} tokens]", tokens.len()))
    }
}

impl LlamaContext {
    /// Process a batch of tokens (stub).
    ///
    /// In a real implementation, this calls llama_decode and fills the KV cache.
    pub fn decode(&mut self, tokens: &[TokenId]) -> Result<(), LlamaError> {
        self.pos += tokens.len();
        Ok(())
    }

    /// Sample the next token (stub).
    ///
    /// In a real implementation, this applies sampling parameters (temperature,
    /// top-p, etc.) to the logits from the last decode call.
    pub fn sample(&self) -> Result<TokenId, LlamaError> {
        // Stub: return a dummy token.
        Ok(42)
    }

    /// Get the current KV cache usage in tokens.
    pub fn kv_cache_used(&self) -> usize {
        self.pos
    }

    /// Clear the KV cache.
    pub fn kv_cache_clear(&mut self) {
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_stub() {
        let model = LlamaModel::load("test.gguf", ModelParams::default()).unwrap();
        assert_eq!(model.n_layers, 40);
    }

    #[test]
    fn test_tokenize_stub() {
        let model = LlamaModel::load("test.gguf", ModelParams::default()).unwrap();
        let tokens = model.tokenize("Hello, world!", true).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 1); // BOS
    }

    #[test]
    fn test_context_decode() {
        let model = LlamaModel::load("test.gguf", ModelParams::default()).unwrap();
        let mut ctx = model.new_context(ContextParams::default()).unwrap();

        ctx.decode(&[1, 2, 3]).unwrap();
        assert_eq!(ctx.kv_cache_used(), 3);

        let token = ctx.sample().unwrap();
        assert_eq!(token, 42);
    }
}
