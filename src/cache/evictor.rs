//! Eviction policy: decides which blocks to move to slower tiers.
//!
//! Uses a weighted scoring function combining:
//! - Inverse cumulative attention score (low attention → evictable)
//! - Time since last access (old → evictable)  
//! - Tier preference (prefer evicting from GPU first to free VRAM)

use std::collections::BinaryHeap;
use std::time::Instant;

use crate::cache::block::{BlockId, KvBlock, Tier};
use crate::config::EvictionConfig;

/// An eviction candidate with its computed priority score.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub block_id: BlockId,
    pub score: f64,
    pub current_tier: Tier,
}

// Higher score = higher eviction priority (should be evicted first).
impl PartialEq for EvictionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for EvictionCandidate {}

impl PartialOrd for EvictionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvictionCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// The eviction policy engine.
pub struct Evictor {
    config: EvictionConfig,
}

impl Evictor {
    pub fn new(config: EvictionConfig) -> Self {
        Self { config }
    }

    /// Compute eviction priority for a single block.
    ///
    /// ```text
    /// eviction_priority(block) =
    ///     α × (1 / attention_score) +
    ///     β × (time_since_last_access) +
    ///     γ × (1 if tier == GPU else 0)
    /// ```
    pub fn compute_priority(&self, block: &KvBlock, now: Instant) -> f64 {
        let attention_component = if block.attention_score > 1e-10 {
            1.0 / block.attention_score
        } else {
            1e10 // effectively infinite priority for zero-attention blocks
        };

        let age_secs = now.duration_since(block.last_access).as_secs_f64();

        let tier_component = if block.tier == Tier::Gpu { 1.0 } else { 0.0 };

        self.config.alpha * attention_component
            + self.config.beta * age_secs
            + self.config.gamma * tier_component
    }

    /// Select up to `count` blocks to evict from the given tier.
    ///
    /// Returns block IDs ordered by eviction priority (highest first).
    /// Blocks in the protected set (e.g. recent hot window) are excluded.
    pub fn select_victims<'a>(
        &self,
        blocks: impl Iterator<Item = &'a KvBlock>,
        tier: Tier,
        count: usize,
        protected_block_ids: &[BlockId],
    ) -> Vec<EvictionCandidate> {
        let now = Instant::now();
        let mut heap = BinaryHeap::new();

        for block in blocks {
            if block.tier != tier {
                continue;
            }
            if protected_block_ids.contains(&block.id) {
                continue;
            }

            let score = self.compute_priority(block, now);
            heap.push(EvictionCandidate {
                block_id: block.id,
                score,
                current_tier: block.tier,
            });
        }

        let mut victims = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(candidate) = heap.pop() {
                victims.push(candidate);
            } else {
                break;
            }
        }

        victims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::block::{GpuLocation, KvBlock};

    fn make_block(id: u64, attention: f64, tier: Tier) -> KvBlock {
        let mut block = KvBlock::new_ram(1, id as usize * 256, 256, vec![0u8; 1024], crate::cache::block::CacheFormat::Q8);
        block.id = id;
        block.tier = tier;
        block.attention_score = attention;
        block
    }

    #[test]
    fn test_eviction_prefers_low_attention() {
        let evictor = Evictor::new(EvictionConfig::default());

        let blocks = vec![
            make_block(0, 10.0, Tier::Ram),  // high attention
            make_block(1, 0.01, Tier::Ram),  // low attention → should be evicted first
            make_block(2, 5.0, Tier::Ram),
        ];

        let victims = evictor.select_victims(blocks.iter(), Tier::Ram, 1, &[]);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].block_id, 1);
    }

    #[test]
    fn test_protected_blocks_excluded() {
        let evictor = Evictor::new(EvictionConfig::default());

        let blocks = vec![
            make_block(0, 0.001, Tier::Ram),
            make_block(1, 0.001, Tier::Ram),
        ];

        let victims = evictor.select_victims(blocks.iter(), Tier::Ram, 2, &[0]);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].block_id, 1);
    }
}
