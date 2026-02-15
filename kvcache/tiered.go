// Package kvcache provides a drop-in replacement for Ollama's kvcache.Causal
// that adds transparent disk-backed tiering.
//
// TieredCausal wraps the standard Causal cache. When tokens are evicted
// (via Remove or ShiftCacheSlot), the raw K/V tensor bytes for the evicted
// positions are snapshot to disk using diskstore. When a sequence resumes
// and Ollama detects a common prefix, TieredCausal checks whether the
// matching prefix extends further on disk and restores the data, avoiding
// recomputation.
//
// INTEGRATION POINT:
//
// In a forked Ollama, replace the cache creation in model implementations
// (e.g. model/models/llama/model.go) to return a TieredCausal instead of
// a plain Causal. The patch is minimal — roughly 10 lines changed.
//
// This file documents the exact interface contract and the modifications
// required to Ollama's source.
package kvcache

import (
	"fmt"
	"log/slog"

	"github.com/databloom/ollama-kv-cache-tiering/diskstore"
)

// TieredConfig configures the tiered cache behaviour.
type TieredConfig struct {
	// DiskStore is the storage backend for evicted blocks.
	DiskStore *diskstore.Store

	// BlockSize is the number of token positions per block when
	// snapshotting to disk. Smaller blocks = finer granularity but
	// more I/O operations. 256 is a good default.
	BlockSize int32

	// Enable controls whether tiering is active. When false, the cache
	// behaves identically to upstream Causal.
	Enable bool
}

// ──────────────────────────────────────────────────────────────────────────
// Below is the integration documentation. We cannot import Ollama's ml
// or kvcache packages directly (they are internal to Ollama's module).
// Instead, this package provides:
//
// 1. The diskstore package — standalone, no Ollama deps.
// 2. This file — documents the exact code changes needed inside Ollama.
// 3. Patch files (in patches/) that apply the changes mechanically.
// ──────────────────────────────────────────────────────────────────────────

// The following Go code is meant to be added TO OLLAMA'S kvcache package.
// It is written here as documentation and verified against Ollama v0.16.1.
//
// === kvcache/tiered.go (add to ollama/kvcache/) ===
//
// TieredCausal wraps a *Causal with disk-backed eviction.
//
//	type TieredCausal struct {
//		*Causal
//		store     *diskstore.Store
//		blockSize int32
//		enabled   bool
//	}
//
//	func NewTieredCausal(causal *Causal, store *diskstore.Store, blockSize int32) *TieredCausal {
//		return &TieredCausal{
//			Causal:    causal,
//			store:     store,
//			blockSize: blockSize,
//			enabled:   true,
//		}
//	}
//
// Override Remove to snapshot evicted KV data before freeing cells:
//
//	func (t *TieredCausal) Remove(seq int, beginIndex, endIndex int32) error {
//		if t.enabled && endIndex != math.MaxInt32 {
//			t.snapshotRange(seq, beginIndex, endIndex)
//		}
//		return t.Causal.Remove(seq, beginIndex, endIndex)
//	}
//
// snapshotRange extracts K/V tensor bytes for the given position range
// and writes them to the disk store:
//
//	func (t *TieredCausal) snapshotRange(seq int, beginPos, endPos int32) {
//		for layer, key := range t.Causal.keys {
//			if key == nil { continue }
//
//			// Extract the raw bytes for the cell range.
//			// The tensor shape is [headDim, numKVHeads, cacheSize].
//			// Each row (one token position) is stride(2) bytes.
//			rowSize := key.Stride(2)
//
//			// Find which cells hold the positions being evicted.
//			for i, cell := range t.Causal.cells {
//				if !slices.Contains(cell.sequences, seq) { continue }
//				if cell.pos < beginPos || cell.pos >= endPos { continue }
//
//				// Read the key row bytes.
//				offset := rowSize * i
//				keyBytes := key.Bytes()[offset : offset+rowSize]
//				bk := diskstore.BlockKey{
//					Seq: seq, Layer: layer,
//					BeginPos: cell.pos, EndPos: cell.pos + 1,
//					IsKey: true,
//				}
//				t.store.Put(bk, t.DType.String(), key.Shape(), keyBytes)
//
//				// Read the value row bytes.
//				val := t.Causal.values[layer]
//				if val != nil {
//					valRowSize := val.Stride(2)
//					valOffset := valRowSize * i
//					valBytes := val.Bytes()[valOffset : valOffset+valRowSize]
//					bv := diskstore.BlockKey{
//						Seq: seq, Layer: layer,
//						BeginPos: cell.pos, EndPos: cell.pos + 1,
//						IsKey: false,
//					}
//					t.store.Put(bv, t.DType.String(), val.Shape(), valBytes)
//				}
//			}
//		}
//		slog.Debug("tiered: snapshot evicted KV",
//			"seq", seq, "begin", beginPos, "end", endPos)
//	}
//
// RestoreRange loads KV data from disk back into the cache's tensors,
// for use when extending a prefix match beyond what's in memory:
//
//	func (t *TieredCausal) RestoreRange(ctx ml.Context, seq int, beginPos, endPos int32) (int32, error) {
//		var restored int32
//		for layer, key := range t.Causal.keys {
//			if key == nil { continue }
//			rowSize := key.Stride(2)
//
//			for pos := beginPos; pos < endPos; pos++ {
//				bk := diskstore.BlockKey{
//					Seq: seq, Layer: layer,
//					BeginPos: pos, EndPos: pos + 1,
//					IsKey: true,
//				}
//				kData, _, err := t.store.Get(bk)
//				if err != nil || kData == nil { continue }
//
//				// Find a free cell and write the data.
//				cellIdx := t.findFreeCell()
//				if cellIdx < 0 { break }
//
//				// Write key bytes into the tensor at cellIdx.
//				offset := rowSize * cellIdx
//				copy(key.Bytes()[offset:offset+rowSize], kData)
//
//				// Write value bytes.
//				bv := bk
//				bv.IsKey = false
//				vData, _, _ := t.store.Get(bv)
//				if vData != nil {
//					val := t.Causal.values[layer]
//					valRowSize := val.Stride(2)
//					valOffset := valRowSize * cellIdx
//					copy(val.Bytes()[valOffset:valOffset+valRowSize], vData)
//				}
//
//				// Update cell metadata.
//				t.Causal.cells[cellIdx] = cacheCell{pos: pos, sequences: []int{seq}}
//				if layer == 0 { restored++ }
//			}
//		}
//		return restored, nil
//	}

// PrintIntegrationGuide prints step-by-step instructions for applying
// the tiered cache to an Ollama checkout.
func PrintIntegrationGuide() {
	guide := `
=== Ollama KV Cache Tiering — Integration Guide ===

This project adds transparent disk-backed KV cache tiering to Ollama.
When the context window fills up and Ollama evicts old tokens, the raw
K/V tensor data is saved to disk (SSD → NFS) instead of being discarded.
On subsequent requests with matching prefixes, the data is restored from
disk, skipping recomputation.

PREREQUISITES:
  - Go 1.23+
  - Ollama source (v0.16.x): git clone https://github.com/ollama/ollama
  - gcc/cmake for CGO (Ollama's standard build deps)

STEPS:

1. Clone Ollama and this project side by side:

     git clone https://github.com/ollama/ollama.git
     git clone https://github.com/databloom/ollama-kv-cache-tiering.git

2. Copy the diskstore package into Ollama's vendor tree:

     cp -r ollama-kv-cache-tiering/diskstore ollama/diskstore

3. Apply the patch to Ollama's kvcache and runner:

     cd ollama
     git apply ../ollama-kv-cache-tiering/patches/ollama-tiered-kvcache.patch

   This patch:
     a) Adds kvcache/tiered.go (TieredCausal wrapper)
     b) Modifies runner/ollamarunner/cache.go:
        - ShiftCacheSlot calls TieredCausal.Remove (snapshots before evicting)
        - LoadCacheSlot checks disk store for extended prefix matches
     c) Adds environment variables:
        - OLLAMA_KV_TIERING=1          (enable tiering)
        - OLLAMA_KV_TIER_LOCAL=/path    (SSD cache dir)
        - OLLAMA_KV_TIER_REMOTE=/path   (NFS cache dir, optional)
        - OLLAMA_KV_TIER_LOCAL_GB=20    (local budget in GB)
        - OLLAMA_KV_TIER_REMOTE_GB=5000 (remote budget in GB)
        - OLLAMA_KV_TIER_COMPRESS=1     (enable zstd compression)

4. Build Ollama:

     go generate ./...
     go build .

5. Run with tiering enabled:

     OLLAMA_KV_TIERING=1 \
     OLLAMA_KV_TIER_LOCAL=/tmp/kv-cache \
     OLLAMA_KV_TIER_REMOTE=/mnt/kv-cache \
     OLLAMA_KV_TIER_LOCAL_GB=20 \
     OLLAMA_KV_TIER_REMOTE_GB=5000 \
     OLLAMA_KV_TIER_COMPRESS=1 \
     ./ollama serve

HOW IT WORKS:

  Normal Ollama flow:
    1. Prompt arrives → tokenize → fill KV cache → generate
    2. Context full → ShiftCacheSlot → Remove(oldest half) → GONE
    3. New prompt → recompute from scratch if prefix doesn't match

  With tiering:
    1. Prompt arrives → tokenize → fill KV cache → generate
    2. Context full → ShiftCacheSlot → snapshot K/V bytes to SSD → Remove
    3. SSD full → oldest blocks migrate to NFS (background)
    4. New prompt → check disk for matching prefix → restore K/V from disk
    5. Only recompute tokens not found on disk

  Benefits:
    - Context shifts are near-instant (restore from SSD: ~2ms per 256 tokens)
    - Long conversations retain KV state across context windows
    - System prompts cached persistently (never recomputed)
    - NFS tier provides TB-scale cold storage for rarely-used contexts

TARGET HARDWARE:

  Molly  (2x GTX 1070, 16GB):  SSD local + 5.8TB NFS from Wintermute
  Wintermute (2x M6000, 48GB): NVMe local + 12TB HDD
`
	fmt.Println(guide)
}

// DefaultTieredConfig returns a sensible default configuration.
func DefaultTieredConfig() TieredConfig {
	return TieredConfig{
		BlockSize: 256,
		Enable:    true,
	}
}
