//! Integration tests for the eviction policy.

use std::time::Instant;

use kv_cache_tier::cache::block::{CacheFormat, KvBlock, Tier};
use kv_cache_tier::cache::evictor::Evictor;
use kv_cache_tier::config::EvictionConfig;

fn make_test_block(id: u64, attention: f64, tier: Tier) -> KvBlock {
    let mut block = KvBlock::new_ram(
        1,
        id as usize * 256,
        256,
        vec![0u8; 1024],
        CacheFormat::Q8,
    );
    block.id = id;
    block.tier = tier;
    block.attention_score = attention;
    block
}

#[test]
fn test_eviction_order_by_attention() {
    let config = EvictionConfig {
        alpha: 1.0, // only attention matters
        beta: 0.0,
        gamma: 0.0,
        ..Default::default()
    };

    let evictor = Evictor::new(config);

    let blocks = vec![
        make_test_block(0, 100.0, Tier::Ram), // high attention
        make_test_block(1, 0.1, Tier::Ram),   // low attention → evict first
        make_test_block(2, 50.0, Tier::Ram),
        make_test_block(3, 0.001, Tier::Ram), // very low → evict first
    ];

    let victims = evictor.select_victims(blocks.iter(), Tier::Ram, 2, &[]);
    assert_eq!(victims.len(), 2);

    // Block 3 (lowest attention) should be evicted first, then block 1.
    assert_eq!(victims[0].block_id, 3);
    assert_eq!(victims[1].block_id, 1);
}

#[test]
fn test_eviction_respects_tier_filter() {
    let config = EvictionConfig::default();
    let evictor = Evictor::new(config);

    let blocks = vec![
        make_test_block(0, 0.1, Tier::Gpu),
        make_test_block(1, 0.1, Tier::Ram),
        make_test_block(2, 0.1, Tier::LocalDisk),
    ];

    // Only select from RAM tier.
    let victims = evictor.select_victims(blocks.iter(), Tier::Ram, 10, &[]);
    assert_eq!(victims.len(), 1);
    assert_eq!(victims[0].block_id, 1);
}

#[test]
fn test_eviction_empty_returns_nothing() {
    let config = EvictionConfig::default();
    let evictor = Evictor::new(config);

    let blocks: Vec<KvBlock> = vec![];
    let victims = evictor.select_victims(blocks.iter(), Tier::Ram, 5, &[]);
    assert!(victims.is_empty());
}
