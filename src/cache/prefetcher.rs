//! Prefetching: predicts which cold/warm blocks will be needed soon
//! and promotes them before the attention computation needs them.
//!
//! Strategies:
//! 1. Sliding window: keep the last N tokens hot
//! 2. Attention-pattern prediction (future)
//! 3. Prompt-aware heuristics (future)

use std::collections::HashSet;

use crate::cache::block::{BlockId, BlockTable, Tier};
use crate::config::PrefetchConfig;

/// A prefetch request: promote a block from its current tier toward GPU.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    pub block_id: BlockId,
    pub current_tier: Tier,
    pub target_tier: Tier,
    pub priority: f64,
}

/// The prefetcher decides which blocks should be proactively promoted.
pub struct Prefetcher {
    config: PrefetchConfig,
}

impl Prefetcher {
    pub fn new(config: PrefetchConfig) -> Self {
        Self { config }
    }

    /// Compute which blocks should be prefetched based on the sliding window strategy.
    ///
    /// Given the current token position, ensure that blocks covering the last
    /// `hot_window_tokens` tokens are in GPU VRAM, plus `prefetch_ahead` blocks
    /// beyond the hot window that should be in RAM.
    pub fn compute_prefetch_requests(
        &self,
        table: &BlockTable,
        current_token_pos: usize,
        block_tiers: &dyn Fn(BlockId) -> Option<Tier>,
    ) -> Vec<PrefetchRequest> {
        let mut requests = Vec::new();

        if table.is_empty() {
            return requests;
        }

        let block_size = table.block_size;

        // Hot window: blocks covering [current_pos - hot_window .. current_pos]
        let hot_start = current_token_pos.saturating_sub(self.config.hot_window_tokens);
        let hot_blocks = table.blocks_in_range(hot_start, current_token_pos + 1);

        // These blocks must be on GPU.
        for (i, &block_id) in hot_blocks.iter().enumerate() {
            if let Some(tier) = block_tiers(block_id) {
                if tier != Tier::Gpu {
                    requests.push(PrefetchRequest {
                        block_id,
                        current_tier: tier,
                        target_tier: Tier::Gpu,
                        priority: 100.0 - i as f64, // closer to current = higher priority
                    });
                }
            }
        }

        // Prefetch-ahead: blocks just outside the hot window should be in RAM.
        let prefetch_start_block = if current_token_pos > 0 {
            current_token_pos.saturating_sub(self.config.hot_window_tokens + self.config.prefetch_ahead * block_size)
        } else {
            0
        };
        let prefetch_end = hot_start;

        if prefetch_end > prefetch_start_block {
            let ahead_blocks = table.blocks_in_range(prefetch_start_block, prefetch_end);

            for &block_id in &ahead_blocks {
                if let Some(tier) = block_tiers(block_id) {
                    if tier == Tier::LocalDisk || tier == Tier::Nfs {
                        requests.push(PrefetchRequest {
                            block_id,
                            current_tier: tier,
                            target_tier: Tier::Ram,
                            priority: 50.0,
                        });
                    }
                }
            }
        }

        // Sort by priority descending.
        requests.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        requests
    }

    /// Return the set of block IDs that should be protected from eviction
    /// (i.e., the hot window blocks).
    pub fn protected_blocks(
        &self,
        table: &BlockTable,
        current_token_pos: usize,
    ) -> HashSet<BlockId> {
        let hot_start = current_token_pos.saturating_sub(self.config.hot_window_tokens);
        let blocks = table.blocks_in_range(hot_start, current_token_pos + 1);
        blocks.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_prefetch() {
        let config = PrefetchConfig {
            hot_window_tokens: 512,
            prefetch_ahead: 2,
            attention_based: false,
        };
        let prefetcher = Prefetcher::new(config);

        let mut table = BlockTable::new(1, 256);
        for i in 0..10 {
            table.push(i as u64, 256);
        }

        // Current position is at token 2048 (block 8).
        // Hot window covers tokens 1536..2048 → blocks 6, 7.
        let requests = prefetcher.compute_prefetch_requests(
            &table,
            2048,
            &|block_id| {
                // Blocks 6 and 7 are on SSD, rest on GPU.
                if block_id == 6 || block_id == 7 {
                    Some(Tier::LocalDisk)
                } else {
                    Some(Tier::Gpu)
                }
            },
        );

        // Should request blocks 6 and 7 to be promoted to GPU.
        let promoted: Vec<_> = requests
            .iter()
            .filter(|r| r.target_tier == Tier::Gpu)
            .collect();
        assert!(!promoted.is_empty());
    }

    #[test]
    fn test_protected_blocks() {
        let config = PrefetchConfig {
            hot_window_tokens: 512,
            prefetch_ahead: 2,
            attention_based: false,
        };
        let prefetcher = Prefetcher::new(config);

        let mut table = BlockTable::new(1, 256);
        for i in 0..10 {
            table.push(i as u64, 256);
        }

        let protected = prefetcher.protected_blocks(&table, 2048);
        // Hot window: 1536..2049 → blocks at indices 6 and 7 → IDs 6, 7
        assert!(protected.contains(&6));
        assert!(protected.contains(&7));
        assert!(!protected.contains(&5));
    }
}
