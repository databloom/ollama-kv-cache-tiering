//! Benchmarks for the KV cache subsystem.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kv_cache_tier::cache::block::{BlockTable, CacheFormat, KvBlock, Tier};
use kv_cache_tier::cache::compressor::Compressor;
use kv_cache_tier::cache::evictor::Evictor;
use kv_cache_tier::config::{CompressionConfig, EvictionConfig};

fn bench_eviction_scoring(c: &mut Criterion) {
    let evictor = Evictor::new(EvictionConfig::default());

    // Create 10,000 blocks.
    let blocks: Vec<KvBlock> = (0..10_000)
        .map(|i| {
            let mut b = KvBlock::new_ram(
                1,
                i * 256,
                256,
                vec![0u8; 128],
                CacheFormat::Q8,
            );
            b.id = i as u64;
            b.attention_score = (i as f64) / 10_000.0;
            b
        })
        .collect();

    c.bench_function("eviction_select_100_from_10k", |b| {
        b.iter(|| {
            let victims = evictor.select_victims(
                black_box(blocks.iter()),
                Tier::Ram,
                100,
                &[],
            );
            black_box(victims);
        })
    });
}

fn bench_compression(c: &mut Criterion) {
    let compressor = Compressor::new(CompressionConfig::default());

    // 256KB block (typical KV block size).
    let data = vec![42u8; 256 * 1024];

    c.bench_function("zstd_compress_256kb", |b| {
        b.iter(|| {
            let compressed = compressor.compress_for_tier(
                &KvBlock::new_ram(1, 0, 256, data.clone(), CacheFormat::Q8),
                Tier::LocalDisk,
            );
            black_box(compressed);
        })
    });
}

fn bench_block_table_lookup(c: &mut Criterion) {
    let mut table = BlockTable::new(1, 256);
    for i in 0..10_000 {
        table.push(i, 256);
    }

    c.bench_function("block_table_lookup_10k", |b| {
        b.iter(|| {
            for pos in (0..2_560_000).step_by(1000) {
                black_box(table.block_for_token(pos));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_eviction_scoring,
    bench_compression,
    bench_block_table_lookup,
);
criterion_main!(benches);
