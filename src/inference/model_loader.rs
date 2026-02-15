//! GGUF model loading and configuration.
//!
//! Reads model metadata from GGUF files to determine architecture
//! parameters (layers, heads, dimensions) needed for KV cache sizing.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum ModelLoaderError {
    #[error("Model file not found: {0}")]
    FileNotFound(String),

    #[error("Invalid GGUF format: {0}")]
    InvalidFormat(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Metadata extracted from a GGUF model file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model architecture name (e.g., "llama", "qwen2").
    pub architecture: String,

    /// Number of transformer layers.
    pub n_layers: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Number of KV heads (for GQA/MQA, may differ from n_heads).
    pub n_kv_heads: usize,

    /// Dimension per attention head.
    pub head_dim: usize,

    /// Vocabulary size.
    pub n_vocab: usize,

    /// Context length the model was trained with.
    pub context_length: usize,

    /// File size in bytes.
    pub file_size: u64,

    /// Quantization type string.
    pub quantization: String,
}

impl ModelMetadata {
    /// Compute the KV cache size per token in bytes (FP16, K+V).
    ///
    /// Per token per layer: 2 * n_kv_heads * head_dim * sizeof(fp16)
    /// Total: per_token_per_layer * n_layers
    pub fn kv_bytes_per_token(&self) -> usize {
        let per_layer = 2 * self.n_kv_heads * self.head_dim * 2; // K+V, FP16
        per_layer * self.n_layers
    }

    /// Compute total KV cache size for a given context length.
    pub fn kv_cache_size(&self, context_length: usize) -> usize {
        self.kv_bytes_per_token() * context_length
    }

    /// Compute how many tokens of context fit in the given VRAM budget.
    pub fn context_for_vram(&self, vram_bytes: usize) -> usize {
        let per_token = self.kv_bytes_per_token();
        if per_token == 0 {
            return 0;
        }
        vram_bytes / per_token
    }
}

/// Load model metadata from a GGUF file.
///
/// Currently uses a stub implementation that returns metadata based on
/// common model architectures. A full implementation would parse the
/// GGUF binary header.
pub fn load_metadata(path: &Path) -> Result<ModelMetadata, ModelLoaderError> {
    if !path.exists() {
        return Err(ModelLoaderError::FileNotFound(
            path.display().to_string(),
        ));
    }

    let file_size = std::fs::metadata(path)?.len();

    // Determine architecture from filename heuristics.
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    let metadata = if filename.contains("qwen2.5-coder-32b") || filename.contains("qwen2.5-32b") {
        ModelMetadata {
            architecture: "qwen2".to_string(),
            n_layers: 64,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            n_vocab: 152064,
            context_length: 32768,
            file_size,
            quantization: "Q4_K_M".to_string(),
        }
    } else if filename.contains("qwen2.5-coder-14b") || filename.contains("qwen2.5-14b") {
        ModelMetadata {
            architecture: "qwen2".to_string(),
            n_layers: 48,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            n_vocab: 152064,
            context_length: 32768,
            file_size,
            quantization: "Q4_K_M".to_string(),
        }
    } else {
        // Default: assume a Llama-style 7B model.
        ModelMetadata {
            architecture: "llama".to_string(),
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 32,
            head_dim: 128,
            n_vocab: 32000,
            context_length: 4096,
            file_size,
            quantization: "Q4_K_M".to_string(),
        }
    };

    info!(
        arch = metadata.architecture,
        layers = metadata.n_layers,
        heads = metadata.n_heads,
        kv_heads = metadata.n_kv_heads,
        head_dim = metadata.head_dim,
        kv_per_token = metadata.kv_bytes_per_token(),
        "Loaded model metadata"
    );

    Ok(metadata)
}

/// Create metadata for testing without a real model file.
pub fn stub_metadata_14b() -> ModelMetadata {
    ModelMetadata {
        architecture: "qwen2".to_string(),
        n_layers: 48,
        n_heads: 40,
        n_kv_heads: 8,
        head_dim: 128,
        n_vocab: 152064,
        context_length: 32768,
        file_size: 9 * 1024 * 1024 * 1024,
        quantization: "Q4_K_M".to_string(),
    }
}

pub fn stub_metadata_32b() -> ModelMetadata {
    ModelMetadata {
        architecture: "qwen2".to_string(),
        n_layers: 64,
        n_heads: 40,
        n_kv_heads: 8,
        head_dim: 128,
        n_vocab: 152064,
        context_length: 32768,
        file_size: 19 * 1024 * 1024 * 1024,
        quantization: "Q4_K_M".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_bytes_per_token_14b() {
        let meta = stub_metadata_14b();
        // 2 * 8 * 128 * 2 * 48 = 196,608 bytes per token
        let expected = 2 * 8 * 128 * 2 * 48;
        assert_eq!(meta.kv_bytes_per_token(), expected);
    }

    #[test]
    fn test_context_for_vram() {
        let meta = stub_metadata_14b();
        let per_token = meta.kv_bytes_per_token();
        let vram = 4 * 1024 * 1024 * 1024; // 4 GB
        let tokens = meta.context_for_vram(vram);
        assert_eq!(tokens, vram / per_token);
        assert!(tokens > 20000); // should support good context at 4GB
    }
}
