//! Tiered KV cache management.
//!
//! This module contains the core cache data structures and algorithms:
//! - [`block`]: KvBlock, BlockTable, Tier definitions
//! - [`pager`]: Tier manager that orchestrates promotion/eviction
//! - [`evictor`]: Eviction policy (attention-score + LRU hybrid)
//! - [`prefetcher`]: Prefetch predictions for proactive tier promotion
//! - [`compressor`]: Quantization and zstd compression/decompression

pub mod block;
pub mod compressor;
pub mod evictor;
pub mod pager;
pub mod prefetcher;
