//! KV block types and block table management.
//!
//! A KV block holds a fixed number of token KV pairs for all layers.
//! Blocks are the unit of movement between tiers.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Identifies which storage tier a block currently resides in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// Tier 0: GPU VRAM (hot).
    Gpu,
    /// Tier 1: Host RAM (warm).
    Ram,
    /// Tier 2: Local SSD (cool).
    LocalDisk,
    /// Tier 3: NFS / remote HDD (cold).
    Nfs,
}

impl Tier {
    /// Returns the numeric tier level (lower = faster).
    pub fn level(&self) -> u8 {
        match self {
            Tier::Gpu => 0,
            Tier::Ram => 1,
            Tier::LocalDisk => 2,
            Tier::Nfs => 3,
        }
    }

    /// Returns the next slower tier for eviction, or None if already coldest.
    pub fn demote(&self) -> Option<Tier> {
        match self {
            Tier::Gpu => Some(Tier::Ram),
            Tier::Ram => Some(Tier::LocalDisk),
            Tier::LocalDisk => Some(Tier::Nfs),
            Tier::Nfs => None,
        }
    }

    /// Returns the next faster tier for promotion, or None if already hottest.
    pub fn promote(&self) -> Option<Tier> {
        match self {
            Tier::Gpu => None,
            Tier::Ram => Some(Tier::Gpu),
            Tier::LocalDisk => Some(Tier::Ram),
            Tier::Nfs => Some(Tier::LocalDisk),
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Gpu => write!(f, "GPU"),
            Tier::Ram => write!(f, "RAM"),
            Tier::LocalDisk => write!(f, "SSD"),
            Tier::Nfs => write!(f, "NFS"),
        }
    }
}

/// The quantization / storage format of a block's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheFormat {
    /// Full precision FP16 (native GPU format).
    Fp16,
    /// 8-bit quantized.
    Q8,
    /// 4-bit quantized.
    Q4,
    /// 4-bit quantized + zstd compressed (on-disk format).
    Q4Zstd,
}

impl CacheFormat {
    /// Bytes per element for this format (approximate).
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            CacheFormat::Fp16 => 2.0,
            CacheFormat::Q8 => 1.0,
            CacheFormat::Q4 => 0.5,
            CacheFormat::Q4Zstd => 0.33, // ~1.5x compression on top of Q4
        }
    }
}

/// Unique identifier for a KV block.
pub type BlockId = u64;

/// Global monotonic block ID counter.
static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(0);

/// Allocate a new unique block ID.
pub fn new_block_id() -> BlockId {
    NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed)
}

/// A single KV cache block.
///
/// Each block holds `block_size` tokens worth of K and V tensors across all layers.
/// Blocks are the unit of tier movement — they are evicted, promoted, and compressed
/// as whole units.
#[derive(Debug)]
pub struct KvBlock {
    /// Unique identifier for this block.
    pub id: BlockId,

    /// Sequence ID this block belongs to.
    pub sequence_id: u64,

    /// Starting token position within the sequence.
    pub token_start: usize,

    /// Number of tokens stored in this block (may be < block_size for the last block).
    pub token_count: usize,

    /// Current storage tier.
    pub tier: Tier,

    /// Current data format.
    pub format: CacheFormat,

    /// Cumulative attention score (exponential moving average).
    pub attention_score: f64,

    /// Timestamp of last access.
    pub last_access: Instant,

    /// Number of times this block has been accessed.
    pub access_count: u64,

    /// If in RAM, pointer to the host buffer.
    /// Stored as a raw pointer + length for zero-copy operations.
    pub ram_data: Option<Vec<u8>>,

    /// If on disk (local SSD or NFS), path to the block file.
    pub disk_path: Option<PathBuf>,

    /// If on GPU, the device ID and offset within the GPU allocator.
    pub gpu_location: Option<GpuLocation>,

    /// Size of the data in bytes (in current format).
    pub data_size: usize,
}

/// Describes where a block lives in GPU memory.
#[derive(Debug, Clone)]
pub struct GpuLocation {
    /// GPU device index.
    pub device_id: usize,

    /// Offset within the KV cache allocation on that device.
    pub offset: usize,

    /// Size in bytes on GPU.
    pub size: usize,
}

impl KvBlock {
    /// Create a new block that will initially live on GPU.
    pub fn new_gpu(
        sequence_id: u64,
        token_start: usize,
        token_count: usize,
        gpu_location: GpuLocation,
        data_size: usize,
    ) -> Self {
        Self {
            id: new_block_id(),
            sequence_id,
            token_start,
            token_count,
            tier: Tier::Gpu,
            format: CacheFormat::Fp16,
            attention_score: 1.0, // start with neutral score
            last_access: Instant::now(),
            access_count: 0,
            ram_data: None,
            disk_path: None,
            gpu_location: Some(gpu_location),
            data_size,
        }
    }

    /// Create a new block in host RAM.
    pub fn new_ram(
        sequence_id: u64,
        token_start: usize,
        token_count: usize,
        data: Vec<u8>,
        format: CacheFormat,
    ) -> Self {
        let data_size = data.len();
        Self {
            id: new_block_id(),
            sequence_id,
            token_start,
            token_count,
            tier: Tier::Ram,
            format,
            attention_score: 1.0,
            last_access: Instant::now(),
            access_count: 0,
            ram_data: Some(data),
            disk_path: None,
            gpu_location: None,
            data_size,
        }
    }

    /// Record an access, updating timestamp and counter.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Update the attention score using exponential moving average.
    pub fn update_attention(&mut self, new_score: f64, decay: f64) {
        self.attention_score = decay * self.attention_score + (1.0 - decay) * new_score;
    }

    /// Token range covered by this block.
    pub fn token_range(&self) -> std::ops::Range<usize> {
        self.token_start..self.token_start + self.token_count
    }

    /// Whether this block's data is currently resident in the given tier.
    pub fn is_resident_in(&self, tier: Tier) -> bool {
        self.tier == tier
    }
}

/// The block table maps sequence positions to blocks.
///
/// For each active sequence (conversation), the block table tracks which blocks
/// cover which token ranges. This is the central index for the cache.
#[derive(Debug)]
pub struct BlockTable {
    /// Sequence ID.
    pub sequence_id: u64,

    /// Ordered list of block IDs covering token positions [0..n).
    pub blocks: Vec<BlockId>,

    /// Total tokens in this sequence.
    pub total_tokens: usize,

    /// Block size (tokens per block).
    pub block_size: usize,
}

impl BlockTable {
    /// Create a new empty block table.
    pub fn new(sequence_id: u64, block_size: usize) -> Self {
        Self {
            sequence_id,
            blocks: Vec::new(),
            total_tokens: 0,
            block_size,
        }
    }

    /// Add a block to the end of the sequence.
    pub fn push(&mut self, block_id: BlockId, token_count: usize) {
        self.blocks.push(block_id);
        self.total_tokens += token_count;
    }

    /// Get the block ID that covers a given token position.
    pub fn block_for_token(&self, token_pos: usize) -> Option<BlockId> {
        if token_pos >= self.total_tokens {
            return None;
        }
        let idx = token_pos / self.block_size;
        self.blocks.get(idx).copied()
    }

    /// Number of blocks in this sequence.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the block table is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get all block IDs in a token range.
    pub fn blocks_in_range(&self, start: usize, end: usize) -> Vec<BlockId> {
        let start_idx = start / self.block_size;
        let end_idx = (end.saturating_sub(1)) / self.block_size + 1;
        self.blocks[start_idx..end_idx.min(self.blocks.len())].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_ordering() {
        assert_eq!(Tier::Gpu.level(), 0);
        assert_eq!(Tier::Nfs.level(), 3);
    }

    #[test]
    fn test_tier_transitions() {
        assert_eq!(Tier::Gpu.demote(), Some(Tier::Ram));
        assert_eq!(Tier::Nfs.demote(), None);
        assert_eq!(Tier::Nfs.promote(), Some(Tier::LocalDisk));
        assert_eq!(Tier::Gpu.promote(), None);
    }

    #[test]
    fn test_block_table_lookup() {
        let mut table = BlockTable::new(1, 256);
        table.push(100, 256);
        table.push(101, 256);
        table.push(102, 128); // partial last block

        assert_eq!(table.block_for_token(0), Some(100));
        assert_eq!(table.block_for_token(255), Some(100));
        assert_eq!(table.block_for_token(256), Some(101));
        assert_eq!(table.block_for_token(512), Some(102));
        assert_eq!(table.block_for_token(700), None);
    }

    #[test]
    fn test_block_attention_update() {
        let loc = GpuLocation {
            device_id: 0,
            offset: 0,
            size: 1024,
        };
        let mut block = KvBlock::new_gpu(1, 0, 256, loc, 1024);
        assert_eq!(block.attention_score, 1.0);

        block.update_attention(0.0, 0.9);
        assert!((block.attention_score - 0.9).abs() < 1e-10);

        block.update_attention(0.0, 0.9);
        assert!((block.attention_score - 0.81).abs() < 1e-10);
    }
}
