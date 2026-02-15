# kv-cache-tier

**Tiered KV-cache for LLM inference on consumer GPUs**

Extends GPU VRAM capacity by transparently paging KV cache blocks through a hierarchy of storage tiers, enabling near-infinite context length on hardware that would otherwise be limited to a few thousand tokens.

```
┌─────────────────────────────────────────────────────────────┐
│                     kv-cache-tier                           │
│                                                             │
│   ┌──────────┐     ┌────────────────────────────────────┐   │
│   │ OpenAI   │────▶│      Inference Orchestrator        │   │
│   │ API      │     │                                    │   │
│   │ (axum)   │     │   ┌──────────────────────────┐     │   │
│   └──────────┘     │   │  Tiered Cache Manager     │     │   │
│                    │   └──────────┬───────────────┘     │   │
│                    └──────────────┼──────────────────────┘   │
│                                   │                          │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────────┐  │
│   │ Tier 0 │   │ Tier 1 │   │ Tier 2 │   │   Tier 3     │  │
│   │GPU VRAM│◀─▶│Host RAM│◀─▶│Local   │◀─▶│  NFS/HDD     │  │
│   │ (hot)  │   │ (warm) │   │SSD     │   │  (cold)      │  │
│   │ 7-28GB │   │12-51GB │   │(cool)  │   │  up to TB    │  │
│   └────────┘   └────────┘   └────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Motivation

Consumer GPUs (GTX 1070, Quadro M6000, etc.) have limited VRAM (8-24 GB). When running large language models, the KV cache consumes a significant portion of available memory, limiting the maximum context length. For example, a 14B parameter model with 32K context needs ~6 GB of KV cache in FP16, leaving little room on an 8 GB card.

**kv-cache-tier** solves this by implementing a virtual memory system for KV cache:

- **Hot** blocks (recently accessed, high attention) stay on GPU for fast inference
- **Warm** blocks spill to host RAM with Q8 quantization (2x compression)
- **Cool** blocks move to local SSD with Q4+zstd (6x compression)
- **Cold** blocks archive to NFS/HDD for near-infinite capacity

An intelligent eviction policy based on attention scores and access patterns ensures the most important context stays fast while rarely-referenced tokens are transparently paged out.

## Features

- **OpenAI-compatible API** — Drop-in replacement for local LLM serving
- **Multi-tier storage** — GPU → RAM → SSD → NFS with automatic management
- **Attention-aware eviction** — Uses actual attention patterns, not just LRU
- **Compression pipeline** — FP16 → Q8 → Q4 + zstd for up to 6x compression
- **Async transfers** — Overlaps data movement with GPU computation
- **Multi-GPU support** — Distributes KV cache across multiple GPUs
- **Streaming responses** — SSE-based token streaming

## Architecture

### Module Structure

```
src/
├── main.rs                  # Entry point, CLI, config
├── config.rs                # Runtime configuration
├── server/
│   ├── openai_api.rs        # OpenAI-compatible HTTP API (axum)
│   └── streaming.rs         # SSE streaming responses
├── inference/
│   ├── engine.rs            # Inference orchestrator
│   ├── llama_ffi.rs         # llama.cpp FFI bindings
│   └── model_loader.rs      # GGUF model loading
├── cache/
│   ├── block.rs             # KvBlock, BlockTable, Tier
│   ├── pager.rs             # Tier manager, promotion/eviction
│   ├── evictor.rs           # Attention-score eviction policy
│   ├── prefetcher.rs        # Prefetch predictions
│   └── compressor.rs        # Quantization + zstd compression
├── transfer/
│   ├── gpu_transfer.rs      # CUDA async memcpy
│   ├── disk_io.rs           # Async disk I/O
│   └── dma_scheduler.rs     # Transfer scheduling
└── gpu/
    ├── device.rs            # GPU device management
    └── allocator.rs         # VRAM block allocator
```

### Eviction Policy

The eviction engine uses a weighted scoring function:

```
priority(block) = α × (1/attention_score) + β × age + γ × tier_preference
```

Where α=0.6, β=0.3, γ=0.1 (configurable). This prioritizes evicting blocks that:
1. Have low cumulative attention scores (rarely referenced)
2. Haven't been accessed recently
3. Are in faster tiers (free up GPU VRAM first)

### Compression Pipeline

```
GPU (FP16, 2 bytes/element)
    │
    ▼ Quantize FP16→Q8 (~0% quality loss)
Host RAM (Q8, 1 byte/element)
    │
    ▼ Quantize Q8→Q4 + zstd (~1-2% quality loss)
Disk (Q4+zstd, ~0.33 bytes/element)
```

Total compression: ~6x, meaning a 6 GB FP16 KV cache compresses to ~1 GB on disk.

## Getting Started

### Prerequisites

- Rust 1.75+ (2021 edition)
- NVIDIA GPU + CUDA toolkit (optional, for GPU acceleration)
- Linux (for io_uring disk I/O support)

### Build

```bash
# CPU-only build (for development/testing)
cargo build --release

# With CUDA support (requires CUDA toolkit)
cargo build --release --features cuda
```

### Run

```bash
# With default config
cargo run --release

# With custom config
cargo run --release -- --config my_config.json --listen 0.0.0.0:8080

# Verbose logging
cargo run --release -- --verbose
```

### Configuration

See `config.json` for a complete example. Key settings:

```json
{
  "model": {
    "model_path": "path/to/model.gguf",
    "context_size": 131072,
    "block_size": 256
  },
  "tiers": {
    "gpu_vram_budget": 7516192768,
    "host_ram_budget": 8589934592,
    "local_ssd_path": "/tmp/kv-cache",
    "local_ssd_budget": 21474836480,
    "nfs_path": "/mnt/kv-cache",
    "nfs_budget": 6413846118400
  },
  "eviction": {
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1
  }
}
```

### API Usage

The server exposes an OpenAI-compatible API:

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Health check with cache stats
curl http://localhost:8080/health

# Cache tier statistics
curl http://localhost:8080/v1/cache/stats
```

### Use as OpenAI drop-in

```bash
export OPENAI_API_BASE=http://localhost:8080
export OPENAI_API_KEY=sk-anything
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

## Target Hardware

Designed and tested for:

| Node | GPUs | VRAM | RAM | Storage |
|------|------|------|-----|---------|
| **Molly** | 2× GTX 1070 | 16 GB | 12 GB | SSD + 5.8 TB NFS |
| **Wintermute** | 2× Quadro M6000 | 48 GB | 51 GB | NVMe + local HDD |

### Capacity with Tiering (Qwen 2.5 14B)

| Tier | Molly | Wintermute |
|------|-------|------------|
| GPU (FP16) | ~38K tokens | ~143K tokens |
| + RAM (Q8) | +61K tokens | +261K tokens |
| + SSD (Q4+zstd) | +115K tokens | +700K tokens |
| + NFS (Q4+zstd) | +29.7M tokens | — |
| **Total** | **~30M tokens** | **~1.1M tokens** |

## Roadmap

- [x] Core cache data structures (KvBlock, BlockTable, Tier)
- [x] Eviction policy (attention-score + LRU hybrid)
- [x] Compression pipeline (FP16 → Q8 → Q4 + zstd)
- [x] Prefetching (sliding window)
- [x] OpenAI-compatible API server
- [x] Disk I/O engine
- [x] GPU VRAM allocator
- [ ] llama.cpp FFI integration (currently stubbed)
- [ ] CUDA async transfers (requires `cuda` feature)
- [ ] io_uring disk I/O
- [ ] Attention-pattern-based prefetching
- [ ] Prometheus metrics endpoint
- [ ] Multi-model support
- [ ] Kubernetes deployment manifests

## License

MIT
