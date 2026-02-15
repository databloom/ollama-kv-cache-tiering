# ollama-kv-cache-tiering

**Transparent disk-backed KV cache tiering for Ollama.**

Patches Ollama to save evicted KV cache tensor data to disk (SSD → NFS) instead
of discarding it. On subsequent requests, matching data is restored from disk,
skipping recomputation. This turns context window shifts from a full recompute
into a fast disk read.

```
                              ┌──────────────────────────┐
                              │        Ollama             │
                              │                          │
  Client ──▶ /api/chat ──▶   │  ┌────────────────────┐  │
                              │  │  ollamarunner      │  │
                              │  │                    │  │
                              │  │  InputCache        │  │
                              │  │    │               │  │
                              │  │    ▼               │  │
                              │  │  TieredCausal      │  │  ◀── this project
                              │  │    │       │       │  │
                              │  │    ▼       ▼       │  │
                              │  │  Causal   DiskStore│  │
                              │  │  (GPU)    (SSD/NFS)│  │
                              │  └────────────────────┘  │
                              └──────────────────────────┘
```

## How it works

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

### What this means in practice

- **Context shifts are fast**: restoring 2K tokens from NVMe takes ~2ms vs ~500ms recompute
- **System prompts are persistent**: cached on disk, never recomputed across sessions
- **Long conversations survive shifts**: evicted context preserved on disk for the session
- **Multi-TB cold storage via NFS**: Molly's 5.8 TB NFS mount from Wintermute as cold tier

## Architecture

The patch touches two layers of Ollama:

### 1. `kvcache/tiered.go` — TieredCausal

Wraps Ollama's `kvcache.Causal` (the standard KV cache). Intercepts:

- **`Remove()`** — Before the parent frees cells, reads the raw K/V tensor bytes
  via `ml.Tensor.Bytes()` and writes them to the disk store. Each tensor row
  (one token position × one layer) is stored as a separate block.

- **`RestoreRange()`** — Reads blocks from disk, writes the bytes back into the
  cache tensors via `copy()` into `ml.Tensor.Bytes()`, and marks the cells as
  occupied. This extends the prefix match beyond what's in GPU memory.

### 2. `runner/ollamarunner/cache.go` — Modified InputCache

- **`NewInputCache()`** — Checks `OLLAMA_KV_TIERING=1` env var. If set, wraps the
  model's cache with `TieredCausal` using a `diskstore.Store`.

- **`LoadCacheSlot()`** — After finding the in-memory prefix match, checks if the
  disk store has contiguous data extending the match. If so, calls `RestoreRange()`
  to load it, increasing the effective prefix length.

### 3. `diskstore/` — Tiered disk storage (standalone package)

Two-tier storage engine with no Ollama dependencies:

- **Local tier** (SSD/NVMe): fast reads/writes, configurable budget
- **Remote tier** (NFS/HDD): large capacity, blocks migrate here when local fills up
- **Optional zstd compression**: reduces I/O and storage footprint
- **Persistent index**: survives restarts, stored as JSON alongside data
- **Sharded directory structure**: avoids filesystem bottlenecks with many files

## Installation

### Prerequisites

- Go 1.23+
- Ollama source (v0.16.x)
- Standard Ollama build dependencies (gcc, cmake for CGO)

### Quick start

```bash
# Clone both repos
git clone https://github.com/ollama/ollama.git
git clone https://github.com/databloom/ollama-kv-cache-tiering.git

# Apply the patch
cd ollama
cp -r ../ollama-kv-cache-tiering/diskstore .
git apply ../ollama-kv-cache-tiering/patches/ollama-tiered-kvcache.patch

# Build
go generate ./...
go build .

# Run with tiering
OLLAMA_KV_TIERING=1 \
OLLAMA_KV_TIER_LOCAL=/tmp/kv-cache \
OLLAMA_KV_TIER_REMOTE=/mnt/kv-cache \
OLLAMA_KV_TIER_LOCAL_GB=20 \
OLLAMA_KV_TIER_REMOTE_GB=5000 \
OLLAMA_KV_TIER_COMPRESS=1 \
./ollama serve
```

Or use the Makefile:

```bash
# Apply patch to ../ollama
make patch OLLAMA_DIR=../ollama

# Build
make build-ollama OLLAMA_DIR=../ollama
```

## Configuration

All configuration is via environment variables (matching Ollama's pattern):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_KV_TIERING` | `0` | Set to `1` to enable tiered KV cache |
| `OLLAMA_KV_TIER_LOCAL` | `/tmp/ollama-kv-cache` | Path for local SSD storage |
| `OLLAMA_KV_TIER_REMOTE` | *(empty)* | Path for NFS/HDD storage (optional) |
| `OLLAMA_KV_TIER_LOCAL_GB` | `20` | Local tier budget in GB |
| `OLLAMA_KV_TIER_REMOTE_GB` | `0` | Remote tier budget in GB |
| `OLLAMA_KV_TIER_COMPRESS` | `0` | Set to `1` for zstd compression |

## Target hardware

| Node | GPUs | VRAM | Local storage | Remote storage |
|------|------|------|---------------|----------------|
| **Molly** | 2× GTX 1070 | 16 GB | SSD | 5.8 TB NFS from Wintermute |
| **Wintermute** | 2× Quadro M6000 | 48 GB | NVMe | 12 TB local HDD |

### Per-token KV cache sizes (Qwen 2.5 Coder 14B, FP16)

| Component | Size |
|-----------|------|
| KV per token per layer | 2 × 8 × 128 × 2 = 4,096 bytes |
| KV per token (48 layers) | 196,608 bytes (~192 KB) |
| 8K context window | ~1.5 GB |
| 32K context (with tiering) | ~6 GB on disk |

With zstd compression on KV data (typical 2-3× on floating point), 5.8 TB of NFS
stores roughly **60-90 million tokens** of evicted context.

## Testing

```bash
# Run diskstore tests (no Ollama dependency required)
make test

# Or directly
go test ./diskstore/ -v
```

## How the patch works (detailed)

### Tensor data access

Ollama's `ml.Tensor` interface provides:
- `Bytes() []byte` — returns the raw backing memory of the tensor
- `FromBytes([]byte)` — writes raw bytes into the tensor

The KV cache tensors in `kvcache.Causal` are:
```
keys[layer]   → shape: [headDim, numKVHeads, cacheSize]
values[layer] → shape: [headDim, numKVHeads, cacheSize]  (or permuted)
```

Each token position occupies one "row" of `Stride(2)` bytes. To snapshot
position `i`, we read `bytes[stride*i : stride*(i+1)]`. To restore, we
write that slice back.

### Eviction flow

```
ShiftCacheSlot(slot, numKeep)
  │
  ├── ShiftDiscard → compute how many positions to evict
  │
  ├── TieredCausal.Remove(seq, numKeep, numKeep+discard)
  │     │
  │     ├── snapshotRange(seq, numKeep, numKeep+discard)
  │     │     │
  │     │     └── for each layer, for each cell in range:
  │     │           read key.Bytes()[offset:offset+rowSize]
  │     │           read value.Bytes()[offset:offset+rowSize]
  │     │           store.Put(blockKey, data)
  │     │
  │     └── Causal.Remove(seq, numKeep, numKeep+discard)  ← cells freed
  │
  └── shift remaining inputs
```

### Restore flow

```
LoadCacheSlot(prompt, cachePrompt)
  │
  ├── findLongestCacheSlot → numPast (in-memory match)
  │
  ├── TieredCausal.RestoreRange(ctx, seq, numPast, targetEnd)
  │     │
  │     └── for pos in numPast..targetEnd:
  │           if store.Has(key for pos, layer 0):
  │             find free cell
  │             for each layer:
  │               data = store.Get(key)
  │               copy(tensor.Bytes()[cellOffset:], data)
  │             mark cell occupied
  │             numPast++
  │           else:
  │             break  (prefix must be contiguous)
  │
  └── return slot, prompt[numPast:], nil
```

## Limitations

- **Not infinite context in a single forward pass.** The attention mechanism still
  operates within `num_ctx` tokens. Tiering extends *prefix caching* to disk, making
  context shifts cheaper — not expanding the attention window itself.
- **WrapperCache (encoder-decoder models) not yet supported.** Only pure causal
  (decoder-only) models like Llama, Qwen, Gemma are supported.
- **Tensor byte access assumes contiguous memory.** This is true for GGML CPU and
  CUDA backends but may not hold for future backends.

## Roadmap

- [x] Disk store with two-tier eviction (local SSD → remote NFS)
- [x] Zstd compression
- [x] Persistent index across restarts
- [x] TieredCausal snapshot/restore logic (in patch)
- [x] Environment variable configuration
- [ ] Benchmark suite (snapshot latency, restore latency, cache hit rate)
- [ ] WrapperCache support for encoder-decoder models
- [ ] Background async snapshot (currently synchronous in eviction path)
- [ ] Quantized KV compression (FP16 → Q8_0 before disk write)
- [ ] Prometheus metrics for cache hit/miss rates

## License

MIT
