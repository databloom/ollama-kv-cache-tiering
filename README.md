# ollama-kv-cache-tiering

**Extends Ollama's effective context window beyond GPU VRAM.**

Two complementary components:

1. **Disk-backed KV cache tiering** (Go) — Saves evicted KV data to SSD/NFS during
   context shifts; restores it on cache hits to avoid recomputation.

2. **Paged ring attention** (CUDA) — Streams KV chunks from host memory through a
   double-buffered pipeline so the model can attend to far more positions than fit
   in GPU VRAM. Uses online softmax to combine chunks without materializing the
   full attention matrix.

```
                    ┌─────────────────────────────────────┐
                    │            Ollama                     │
                    │                                       │
  Client ──▶       │  ┌─────────────────────────────────┐  │
  /api/chat        │  │  ollamarunner                   │  │
                    │  │                                 │  │
                    │  │  InputCache                     │  │
                    │  │    │                            │  │
                    │  │    ▼                            │  │
                    │  │  TieredCausal ─────▶ DiskStore  │  │  ◀── Go layer
                    │  │    │                 (SSD/NFS)  │  │
                    │  │    ▼                            │  │
                    │  │  Causal (GPU)                   │  │
                    │  │    │                            │  │
                    │  │    ▼                            │  │
                    │  │  PagedAttn kernel ◀─── KVPager  │  │  ◀── CUDA layer
                    │  │    │     ▲    ▲        │       │  │
                    │  │    │     │    │        ▼       │  │
                    │  │    │  [ping] [pong]  host RAM  │  │
                    │  │    │  GPU bufs       / disk    │  │
                    │  │    ▼                            │  │
                    │  │  output                         │  │
                    │  └─────────────────────────────────┘  │
                    └─────────────────────────────────────┘
```

## Component 1: Disk-backed KV cache tiering

### Without tiering (stock Ollama)

1. Prompt arrives → tokenize → fill KV cache → generate tokens
2. Context window full → `ShiftCacheSlot` → **delete** oldest half → recompute if needed
3. New request → if prefix doesn't match in-memory cache → full recompute

### With tiering (this patch)

1. Prompt arrives → tokenize → fill KV cache → generate tokens
2. Context window full → `ShiftCacheSlot` → **snapshot K/V bytes to SSD** → then delete from GPU
3. SSD fills up → oldest blocks automatically migrate to NFS (background)
4. New request → check in-memory prefix → **extend match from disk** → restore K/V tensors
5. Only recompute tokens not found on disk or in memory

**What this means**: Context shifts go from ~500ms recompute to ~2ms disk read.
System prompts persist across sessions. Long conversations survive eviction.

## Component 2: Paged ring attention (CUDA kernel)

This is what actually **expands the attention window** beyond GPU VRAM.

### The problem

Standard attention requires all K/V tensors in GPU memory simultaneously:
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```
With 8 GB VRAM, you're limited to ~8K tokens for a 14B model.

### The solution

Process K/V in chunks using **online softmax** — the same algorithm
FlashAttention uses for tiling, but extended across a **host↔GPU pipeline**:

```
For each KV chunk (streamed from host RAM / disk):
    1. Async-copy next chunk to GPU buffer (ping-pong double-buffering)
    2. Compute partial attention scores: scores = Q · K_chunk^T / √d
    3. Update running state using online softmax:
         m_new = max(m_old, max(scores))
         correction = exp(m_old - m_new)
         O = O * correction + exp(scores - m_new) · V_chunk
         l = l * correction + Σ exp(scores - m_new)
    4. Swap ping/pong buffers
Final: output = O / l
```

Only **one chunk** of K and V lives on GPU at any time. The rest stays in
pinned host RAM or on disk.

### Performance (Quadro M6000, PCIe 3.0)

Simulating qwen2.5-coder:14b (40 Q heads, 8 KV heads, D=128):

| Context | Chunks | ms/token/layer | Est. full model (48L) | Effective tok/s |
|---------|--------|----------------|-----------------------|-----------------|
| 4K      | 2      | 6.2            | 298 ms                | 3.4             |
| 8K      | 4      | 12.4           | 595 ms                | 1.7             |
| 16K     | 8      | 24.8           | 1,190 ms              | 0.84            |
| 32K     | 16     | 46.8           | 2,246 ms              | 0.45            |
| 65K     | 32     | 91.7           | 4,402 ms              | 0.23            |

With hybrid mode (4K hot on GPU + cold paged from host), 16K context
would be ~10ms/layer cold + fast attention for hot → ~700ms total.

### Kernel properties

- **CC ≥ 5.2** — Works on Maxwell (M6000), Pascal (GTX 1070), and newer
- **f16 K/V, f32 accumulation** — Mixed precision, < 0.05% relative error
- **GQA support** — Handles grouped query attention (e.g., 40 Q / 8 KV heads)
- **Templated head_dim** — Optimized for D=64, 80, 96, 128, 256
- **11/11 correctness tests passing** against f32 reference implementation

## Architecture

```
ollama-kv-cache-tiering/
├── diskstore/              # Go: two-tier disk storage (SSD → NFS)
│   ├── store.go            #   Put/Get/Has/RemoveSeq with LRU eviction
│   └── store_test.go       #   Unit tests
├── kvcache/                # Go: TieredCausal wrapper for Ollama
│   └── tiered.go           #   Intercepts Remove() to snapshot, RestoreRange() to reload
├── ggml-paged/             # CUDA: paged ring attention kernel
│   ├── paged_attn.h        #   Public C API
│   ├── paged_attn.cu       #   Kernel + double-buffered orchestration
│   ├── kv_pager.h          #   Host-side page manager API
│   ├── kv_pager.c          #   Page manager (pinned RAM + disk tiers)
│   ├── ggml_paged_bridge.h #   GGML ↔ kernel bridge
│   ├── ggml_paged_bridge.cu#   Bridge implementation (context pool + dispatch)
│   ├── CMakeLists.txt       #   Build system (targets sm_52, sm_61)
│   └── test_paged_attn.cu  #   Correctness tests
├── patches/
│   ├── ollama-tiered-kvcache.patch   # Go-layer tiering patch
│   └── ggml-paged-attention.patch    # GGML integration guide
├── cmd/patch-ollama/       # Helper: prints integration guide
└── Makefile
```

## Installation

### Prerequisites

- Go 1.24+ (for Ollama v0.16.x)
- CUDA toolkit 11.5+
- cmake 3.18+, gcc
- Ollama source (v0.16.x)

### Build the CUDA kernel

```bash
cd ggml-paged
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="52;61"
make -j$(nproc)

# Run correctness tests
./test_paged_attn
```

### Apply the Go-layer tiering patch

```bash
git clone https://github.com/ollama/ollama.git
cd ollama && git checkout v0.16.1

# Copy diskstore
cp -r ../ollama-kv-cache-tiering/diskstore .

# Apply patch
git apply ../ollama-kv-cache-tiering/patches/ollama-tiered-kvcache.patch

# Add dependency
go get github.com/klauspost/compress@v1.17.11

# Build
go generate ./...
go build .
```

### Integrate the CUDA paged attention

See `patches/ggml-paged-attention.patch` for the step-by-step GGML integration
guide. The key changes:

1. Copy `ggml-paged/` into `ml/backend/ggml/ggml/src/ggml-paged/`
2. Add `GGML_OP_FLASH_ATTN_EXT_PAGED` op to GGML
3. Wire CUDA dispatch to our kernel
4. Modify `kvcache.Causal` to support host-resident KV allocation

### Run with tiering enabled

```bash
OLLAMA_KV_TIERING=1 \
OLLAMA_KV_TIER_LOCAL=/tmp/kv-cache \
OLLAMA_KV_TIER_REMOTE=/mnt/nfs/kv-cache \
OLLAMA_KV_TIER_LOCAL_GB=20 \
OLLAMA_KV_TIER_REMOTE_GB=5000 \
OLLAMA_KV_TIER_COMPRESS=1 \
./ollama serve
```

### Run with paged attention (expanded window)

```bash
OLLAMA_PAGED_ATTN=1 \
OLLAMA_PAGED_CHUNK_SIZE=2048 \
OLLAMA_PAGED_HOST_GB=8 \
OLLAMA_NUM_CTX=65536 \
./ollama serve
```

## Configuration

### Tiering (Go layer)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_KV_TIERING` | `0` | Set to `1` to enable tiered KV cache |
| `OLLAMA_KV_TIER_LOCAL` | `/tmp/ollama-kv-cache` | Path for local SSD storage |
| `OLLAMA_KV_TIER_REMOTE` | *(empty)* | Path for NFS/HDD storage (optional) |
| `OLLAMA_KV_TIER_LOCAL_GB` | `20` | Local tier budget in GB |
| `OLLAMA_KV_TIER_REMOTE_GB` | `0` | Remote tier budget in GB |
| `OLLAMA_KV_TIER_COMPRESS` | `0` | Set to `1` for zstd compression |

### Paged attention (CUDA layer)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_PAGED_ATTN` | `0` | Set to `1` to enable paged attention |
| `OLLAMA_PAGED_CHUNK_SIZE` | `2048` | Positions per GPU page (power of 2) |
| `OLLAMA_PAGED_HOST_GB` | `8` | Host RAM budget for KV pages |
| `OLLAMA_NUM_CTX` | model default | Context window size (can now be >> VRAM) |

## Target hardware

| Node | GPUs | VRAM | CC | PCIe | Host RAM |
|------|------|------|----|------|----------|
| **Molly** | 2× GTX 1070 | 16 GB | 6.1 | 3.0 x16 | 15 GB |
| **Wintermute** | 2× Quadro M6000 | 48 GB | 5.2 | 3.0 x16 | 62 GB |

### Memory math (Qwen 2.5 Coder 14B, FP16)

| Metric | Value |
|--------|-------|
| KV per token per layer | 2 × 8 × 128 × 2 = 4,096 bytes |
| KV per token (48 layers) | 192 KB |
| 8K context in GPU | ~1.5 GB |
| 64K context in host RAM | ~12 GB |
| 64K context on disk (zstd ~2.5×) | ~4.8 GB |

## Testing

```bash
# Go: diskstore unit tests
go test ./diskstore/ -v

# CUDA: paged attention correctness
cd ggml-paged/build && ./test_paged_attn

# CUDA: performance benchmark
./bench_paged  # (if built)
```

## Limitations

- **Paged attention is slow at very long context.** At 65K tokens the PCIe
  bandwidth cost is ~4.4 sec/token for a 48-layer model. Hybrid mode
  (hot window + cold paging) mitigates this significantly.
- **GGML integration is not yet automated.** The CUDA kernel works standalone
  but wiring it into GGML's op graph requires manual patching (see patch guide).
- **WrapperCache (encoder-decoder models) not yet supported.**
- **Tensor byte access assumes contiguous memory.**

## Roadmap

- [x] Disk store with two-tier eviction (local SSD → remote NFS)
- [x] Zstd compression
- [x] TieredCausal snapshot/restore logic
- [x] CUDA paged attention kernel with online softmax
- [x] Double-buffered host→GPU pipeline
- [x] GQA (grouped query attention) support
- [x] Correctness test suite (11/11 passing)
- [x] Performance benchmark
- [ ] Hybrid hot/cold attention (recent on GPU + historical paged)
- [ ] Automated GGML patch application
- [ ] Prometheus metrics
- [ ] Background async snapshot
- [ ] Quantized KV compression (FP16 → Q8_0 before disk write)

## License

MIT
