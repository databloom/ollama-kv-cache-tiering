//! Runtime configuration for kv-cache-tier.
//!
//! Configuration can be loaded from a YAML/JSON file or constructed programmatically.
//! All tier-related knobs (capacities, thresholds, eviction weights) live here.

use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

/// Command-line arguments.
#[derive(Parser, Debug, Clone)]
#[command(name = "kv-cache-tier", about = "Tiered KV-cache LLM inference server")]
pub struct Cli {
    /// Path to configuration file (JSON).
    #[arg(short, long, default_value = "config.json")]
    pub config: PathBuf,

    /// HTTP listen address.
    #[arg(long, default_value = "0.0.0.0:8080")]
    pub listen: String,

    /// Enable verbose logging.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration.
    pub server: ServerConfig,

    /// Model configuration.
    pub model: ModelConfig,

    /// Tier configuration.
    pub tiers: TierConfig,

    /// Eviction policy tuning.
    pub eviction: EvictionConfig,

    /// Compression settings.
    pub compression: CompressionConfig,

    /// Prefetching settings.
    pub prefetch: PrefetchConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            tiers: TierConfig::default(),
            eviction: EvictionConfig::default(),
            compression: CompressionConfig::default(),
            prefetch: PrefetchConfig::default(),
        }
    }
}

/// HTTP server settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Listen address (e.g. "0.0.0.0:8080").
    pub listen: String,

    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".to_string(),
            max_concurrent_requests: 4,
            request_timeout_secs: 300,
        }
    }
}

/// Model-related settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,

    /// Number of GPU layers to offload (-1 = all).
    pub n_gpu_layers: i32,

    /// Context size in tokens.
    pub context_size: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Number of KV heads (for GQA/MQA).
    pub n_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Number of layers in the model.
    pub n_layers: usize,

    /// KV block size in tokens.
    pub block_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.gguf"),
            n_gpu_layers: -1,
            context_size: 32768,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            n_layers: 40,
            block_size: 256,
        }
    }
}

/// Tier capacity and path configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// GPU VRAM budget for KV cache in bytes (0 = auto-detect).
    pub gpu_vram_budget: usize,

    /// Host RAM budget for KV cache in bytes.
    pub host_ram_budget: usize,

    /// Path for local SSD storage.
    pub local_ssd_path: PathBuf,

    /// Maximum bytes on local SSD.
    pub local_ssd_budget: usize,

    /// Path for NFS/HDD storage (optional).
    pub nfs_path: Option<PathBuf>,

    /// Maximum bytes on NFS.
    pub nfs_budget: usize,

    /// High watermark: start eviction when tier usage exceeds this fraction.
    pub high_watermark: f64,

    /// Low watermark: stop eviction when tier usage drops below this fraction.
    pub low_watermark: f64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            gpu_vram_budget: 0, // auto-detect
            host_ram_budget: 8 * 1024 * 1024 * 1024, // 8 GB
            local_ssd_path: PathBuf::from("/tmp/kv-cache"),
            local_ssd_budget: 20 * 1024 * 1024 * 1024, // 20 GB
            nfs_path: None,
            nfs_budget: 0,
            high_watermark: 0.85,
            low_watermark: 0.70,
        }
    }
}

/// Eviction policy weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionConfig {
    /// Weight for inverse attention score (higher = prefer evicting low-attention blocks).
    pub alpha: f64,

    /// Weight for age / time since last access.
    pub beta: f64,

    /// Weight for tier preference (prefer evicting from faster tiers first).
    pub gamma: f64,

    /// Exponential moving average decay for attention scores.
    pub attention_ema_decay: f64,

    /// Minimum number of blocks to keep hot on GPU.
    pub min_hot_blocks: usize,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            beta: 0.3,
            gamma: 0.1,
            attention_ema_decay: 0.9,
            min_hot_blocks: 8, // 2048 tokens at block_size=256
        }
    }
}

/// Compression settings per tier transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Quantize to Q8 when moving GPU → RAM.
    pub gpu_to_ram_quantize: bool,

    /// Quantize to Q4 when moving RAM → Disk.
    pub ram_to_disk_quantize: bool,

    /// Apply zstd compression when writing to disk.
    pub disk_zstd_compression: bool,

    /// zstd compression level (1-22).
    pub zstd_level: i32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            gpu_to_ram_quantize: true,
            ram_to_disk_quantize: true,
            disk_zstd_compression: true,
            zstd_level: 3,
        }
    }
}

/// Prefetch strategy settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchConfig {
    /// Number of tokens in the sliding hot window.
    pub hot_window_tokens: usize,

    /// Number of blocks to prefetch ahead.
    pub prefetch_ahead: usize,

    /// Enable attention-pattern-based prefetching.
    pub attention_based: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            hot_window_tokens: 2048,
            prefetch_ahead: 4,
            attention_based: false,
        }
    }
}

impl Config {
    /// Load configuration from a JSON file, falling back to defaults for missing fields.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            let config: Config = serde_json::from_str(&data)?;
            Ok(config)
        } else {
            tracing::warn!("Config file not found at {:?}, using defaults", path);
            Ok(Config::default())
        }
    }

    /// Compute the size of a single KV block in bytes (FP16, both K and V).
    pub fn kv_block_bytes(&self) -> usize {
        // K and V, each: block_size * n_kv_heads * head_dim * 2 bytes (FP16)
        // Across all layers
        let per_layer = self.model.block_size * self.model.n_kv_heads * self.model.head_dim * 2 * 2;
        per_layer * self.model.n_layers
    }

    /// Compute how many tokens can fit in a given byte budget at FP16.
    pub fn tokens_for_budget(&self, budget_bytes: usize) -> usize {
        let block_bytes = self.kv_block_bytes();
        if block_bytes == 0 {
            return 0;
        }
        let blocks = budget_bytes / block_bytes;
        blocks * self.model.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.eviction.alpha, 0.6);
        assert_eq!(cfg.model.block_size, 256);
    }

    #[test]
    fn test_kv_block_bytes() {
        let cfg = Config::default();
        // block_size(256) * n_kv_heads(8) * head_dim(128) * 2(fp16) * 2(K+V) * n_layers(40)
        let expected = 256 * 8 * 128 * 2 * 2 * 40;
        assert_eq!(cfg.kv_block_bytes(), expected);
    }
}
