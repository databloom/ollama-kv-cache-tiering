//! Integration tests for the tiered KV cache.

use std::sync::Arc;

use kv_cache_tier::cache::block::{BlockTable, CacheFormat, KvBlock, Tier};
use kv_cache_tier::cache::pager::Pager;
use kv_cache_tier::config::Config;

#[test]
fn test_block_lifecycle() {
    let config = Arc::new(Config::default());
    let mut pager = Pager::new(config);

    // Insert blocks simulating a 1024-token sequence.
    let seq_id = 1;
    let block_size = 256;

    for i in 0..4 {
        let block = KvBlock::new_ram(
            seq_id,
            i * block_size,
            block_size,
            vec![0u8; 4096],
            CacheFormat::Q8,
        );
        let block_id = block.id;
        pager.insert_block(block);

        let table = pager.get_or_create_sequence(seq_id);
        table.push(block_id, block_size);
    }

    // Verify sequence state.
    let table = pager.get_sequence(seq_id).unwrap();
    assert_eq!(table.len(), 4);
    assert_eq!(table.total_tokens, 1024);

    // Verify tier stats.
    let stats = pager.tier_stats().get(&Tier::Ram).unwrap();
    assert_eq!(stats.block_count, 4);
    assert_eq!(stats.bytes_used, 4 * 4096);

    // Remove the sequence.
    let removed = pager.remove_sequence(seq_id);
    assert_eq!(removed.len(), 4);
    assert_eq!(pager.total_blocks(), 0);
}

#[test]
fn test_eviction_threshold() {
    let mut cfg = Config::default();
    cfg.tiers.host_ram_budget = 10000;
    cfg.tiers.high_watermark = 0.80;
    let config = Arc::new(cfg);

    let mut pager = Pager::new(config);

    // Insert blocks totaling 9000 bytes (90% of 10000 budget).
    for i in 0..9 {
        let block = KvBlock::new_ram(1, i * 256, 256, vec![0u8; 1000], CacheFormat::Q8);
        pager.insert_block(block);
    }

    // Should need eviction since 90% > 80% watermark.
    assert!(pager.needs_eviction().is_some());
    assert_eq!(pager.needs_eviction().unwrap(), Tier::Ram);
}

#[test]
fn test_no_eviction_below_watermark() {
    let mut cfg = Config::default();
    cfg.tiers.host_ram_budget = 10000;
    cfg.tiers.high_watermark = 0.80;
    let config = Arc::new(cfg);

    let mut pager = Pager::new(config);

    // Insert blocks totaling 5000 bytes (50% of 10000 budget).
    for i in 0..5 {
        let block = KvBlock::new_ram(1, i * 256, 256, vec![0u8; 1000], CacheFormat::Q8);
        pager.insert_block(block);
    }

    // Should not need eviction since 50% < 80% watermark.
    assert!(pager.needs_eviction().is_none());
}
