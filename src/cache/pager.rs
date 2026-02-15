//! Tier manager (pager): orchestrates block movement between tiers.
//!
//! The pager is the central coordinator for the tiered KV cache. It:
//! - Tracks all blocks across all tiers
//! - Triggers eviction when a tier exceeds its high watermark
//! - Coordinates promotion (prefetch) of cold blocks to warmer tiers
//! - Maintains per-tier usage accounting

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::cache::block::{BlockId, BlockTable, KvBlock, Tier};
use crate::cache::compressor::Compressor;
use crate::cache::evictor::Evictor;
use crate::config::Config;

/// Per-tier usage statistics.
#[derive(Debug, Clone, Default)]
pub struct TierStats {
    /// Number of blocks in this tier.
    pub block_count: usize,
    /// Total bytes used in this tier.
    pub bytes_used: usize,
    /// Capacity budget in bytes.
    pub capacity: usize,
}

impl TierStats {
    /// Usage as a fraction of capacity (0.0 - 1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.bytes_used as f64 / self.capacity as f64
    }

    /// Whether this tier has exceeded its high watermark.
    pub fn above_high_watermark(&self, watermark: f64) -> bool {
        self.usage_fraction() > watermark
    }

    /// Whether this tier is below its low watermark.
    pub fn below_low_watermark(&self, watermark: f64) -> bool {
        self.usage_fraction() < watermark
    }
}

/// The central tier manager.
pub struct Pager {
    /// All blocks indexed by ID.
    blocks: HashMap<BlockId, KvBlock>,

    /// Block tables indexed by sequence ID.
    sequences: HashMap<u64, BlockTable>,

    /// Per-tier statistics.
    tier_stats: HashMap<Tier, TierStats>,

    /// Eviction policy.
    evictor: Evictor,

    /// Compression engine.
    compressor: Compressor,

    /// Configuration.
    config: Arc<Config>,
}

impl Pager {
    /// Create a new pager with the given configuration.
    pub fn new(config: Arc<Config>) -> Self {
        let evictor = Evictor::new(config.eviction.clone());
        let compressor = Compressor::new(config.compression.clone());

        let mut tier_stats = HashMap::new();
        tier_stats.insert(
            Tier::Gpu,
            TierStats {
                capacity: config.tiers.gpu_vram_budget,
                ..Default::default()
            },
        );
        tier_stats.insert(
            Tier::Ram,
            TierStats {
                capacity: config.tiers.host_ram_budget,
                ..Default::default()
            },
        );
        tier_stats.insert(
            Tier::LocalDisk,
            TierStats {
                capacity: config.tiers.local_ssd_budget,
                ..Default::default()
            },
        );
        if config.tiers.nfs_path.is_some() {
            tier_stats.insert(
                Tier::Nfs,
                TierStats {
                    capacity: config.tiers.nfs_budget,
                    ..Default::default()
                },
            );
        }

        Self {
            blocks: HashMap::new(),
            sequences: HashMap::new(),
            tier_stats,
            evictor,
            compressor,
            config,
        }
    }

    /// Register a new block in the pager.
    pub fn insert_block(&mut self, block: KvBlock) {
        let tier = block.tier;
        let size = block.data_size;
        let id = block.id;

        if let Some(stats) = self.tier_stats.get_mut(&tier) {
            stats.block_count += 1;
            stats.bytes_used += size;
        }

        self.blocks.insert(id, block);
    }

    /// Get a reference to a block by ID.
    pub fn get_block(&self, id: BlockId) -> Option<&KvBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable reference to a block by ID.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut KvBlock> {
        self.blocks.get_mut(&id)
    }

    /// Get or create a block table for a sequence.
    pub fn get_or_create_sequence(&mut self, sequence_id: u64) -> &mut BlockTable {
        self.sequences
            .entry(sequence_id)
            .or_insert_with(|| BlockTable::new(sequence_id, self.config.model.block_size))
    }

    /// Get the block table for a sequence.
    pub fn get_sequence(&self, sequence_id: u64) -> Option<&BlockTable> {
        self.sequences.get(&sequence_id)
    }

    /// Remove a sequence and all its blocks.
    pub fn remove_sequence(&mut self, sequence_id: u64) -> Vec<BlockId> {
        let mut removed = Vec::new();
        if let Some(table) = self.sequences.remove(&sequence_id) {
            for block_id in &table.blocks {
                if let Some(block) = self.blocks.remove(block_id) {
                    if let Some(stats) = self.tier_stats.get_mut(&block.tier) {
                        stats.block_count = stats.block_count.saturating_sub(1);
                        stats.bytes_used = stats.bytes_used.saturating_sub(block.data_size);
                    }
                    removed.push(block.id);
                }
            }
        }
        removed
    }

    /// Check if any tier exceeds its high watermark and needs eviction.
    pub fn needs_eviction(&self) -> Option<Tier> {
        for tier in &[Tier::Gpu, Tier::Ram, Tier::LocalDisk] {
            if let Some(stats) = self.tier_stats.get(tier) {
                if stats.above_high_watermark(self.config.tiers.high_watermark) {
                    return Some(*tier);
                }
            }
        }
        None
    }

    /// Run one round of eviction for the given tier.
    ///
    /// Selects victim blocks, compresses them, and moves them to the next slower tier.
    /// Returns the number of blocks evicted.
    pub async fn evict(&mut self, tier: Tier) -> anyhow::Result<usize> {
        let target_tier = match tier.demote() {
            Some(t) => t,
            None => {
                warn!("Cannot evict from coldest tier ({tier})");
                return Ok(0);
            }
        };

        // Determine how many blocks to evict to reach the low watermark.
        let stats = self
            .tier_stats
            .get(&tier)
            .cloned()
            .unwrap_or_default();

        let target_bytes = (self.config.tiers.low_watermark * stats.capacity as f64) as usize;
        let excess = stats.bytes_used.saturating_sub(target_bytes);
        if excess == 0 {
            return Ok(0);
        }

        // Estimate how many blocks to evict.
        let avg_block_size = if stats.block_count > 0 {
            stats.bytes_used / stats.block_count
        } else {
            return Ok(0);
        };
        let blocks_to_evict = (excess / avg_block_size).max(1);

        // Determine protected blocks (hot window).
        let protected: Vec<BlockId> = Vec::new(); // TODO: integrate with prefetcher

        let victims = self
            .evictor
            .select_victims(self.blocks.values(), tier, blocks_to_evict, &protected);

        let mut evicted = 0;
        for victim in victims {
            if let Some(block) = self.blocks.get_mut(&victim.block_id) {
                // Compress the block for the target tier.
                let compressed = self
                    .compressor
                    .compress_for_tier(block, target_tier)?;

                // Update accounting on source tier.
                if let Some(src_stats) = self.tier_stats.get_mut(&tier) {
                    src_stats.block_count = src_stats.block_count.saturating_sub(1);
                    src_stats.bytes_used = src_stats.bytes_used.saturating_sub(block.data_size);
                }

                // Move block data.
                block.ram_data = Some(compressed);
                block.tier = target_tier;
                block.data_size = block.ram_data.as_ref().map(|d| d.len()).unwrap_or(0);

                // Update accounting on target tier.
                if let Some(dst_stats) = self.tier_stats.get_mut(&target_tier) {
                    dst_stats.block_count += 1;
                    dst_stats.bytes_used += block.data_size;
                }

                evicted += 1;
                debug!(
                    block_id = victim.block_id,
                    from = %tier,
                    to = %target_tier,
                    "Evicted block"
                );
            }
        }

        if evicted > 0 {
            info!(
                evicted,
                tier = %tier,
                target = %target_tier,
                "Eviction round complete"
            );
        }

        Ok(evicted)
    }

    /// Get tier statistics for monitoring.
    pub fn tier_stats(&self) -> &HashMap<Tier, TierStats> {
        &self.tier_stats
    }

    /// Total number of blocks across all tiers.
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of active sequences.
    pub fn total_sequences(&self) -> usize {
        self.sequences.len()
    }
}

/// Thread-safe wrapper around the pager.
pub type SharedPager = Arc<RwLock<Pager>>;

/// Create a new thread-safe pager.
pub fn new_shared_pager(config: Arc<Config>) -> SharedPager {
    Arc::new(RwLock::new(Pager::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::block::{CacheFormat, KvBlock};

    fn test_config() -> Arc<Config> {
        let mut config = Config::default();
        config.tiers.gpu_vram_budget = 10000;
        config.tiers.host_ram_budget = 50000;
        config.tiers.high_watermark = 0.80;
        config.tiers.low_watermark = 0.50;
        Arc::new(config)
    }

    #[test]
    fn test_pager_insert_and_stats() {
        let config = test_config();
        let mut pager = Pager::new(config);

        let block = KvBlock::new_ram(1, 0, 256, vec![0u8; 5000], CacheFormat::Q8);
        pager.insert_block(block);

        let stats = pager.tier_stats().get(&Tier::Ram).unwrap();
        assert_eq!(stats.block_count, 1);
        assert_eq!(stats.bytes_used, 5000);
    }

    #[test]
    fn test_pager_remove_sequence() {
        let config = test_config();
        let mut pager = Pager::new(config);

        let block = KvBlock::new_ram(42, 0, 256, vec![0u8; 1000], CacheFormat::Q8);
        let block_id = block.id;
        pager.insert_block(block);

        let table = pager.get_or_create_sequence(42);
        table.push(block_id, 256);

        let removed = pager.remove_sequence(42);
        assert_eq!(removed.len(), 1);
        assert!(pager.get_block(block_id).is_none());
    }
}
