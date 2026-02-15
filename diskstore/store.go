// Package diskstore implements tiered storage for evicted KV cache blocks.
//
// Blocks are written to a fast local tier (SSD) first and can be promoted
// to a slow remote tier (NFS/HDD) when the local tier fills up.
// Data is optionally compressed with zstd before writing.
package diskstore

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

// BlockKey uniquely identifies an evicted KV block.
type BlockKey struct {
	Seq       int   `json:"seq"`        // Sequence (slot) ID
	Layer     int   `json:"layer"`      // Transformer layer index
	BeginPos  int32 `json:"begin_pos"`  // First token position in block
	EndPos    int32 `json:"end_pos"`    // One-past-last token position
	IsKey     bool  `json:"is_key"`     // true = key tensor, false = value tensor
}

// String returns a human-readable key for logging.
func (k BlockKey) String() string {
	kv := "v"
	if k.IsKey {
		kv = "k"
	}
	return fmt.Sprintf("seq%d_L%d_%s_p%d-%d", k.Seq, k.Layer, kv, k.BeginPos, k.EndPos)
}

// BlockMeta holds metadata about a stored block, persisted alongside the data.
type BlockMeta struct {
	Key        BlockKey  `json:"key"`
	DTypeStr   string    `json:"dtype"`        // e.g. "f16", "q8_0"
	Shape      []int     `json:"shape"`        // original tensor shape
	SizeBytes  int       `json:"size_bytes"`   // uncompressed size
	Compressed bool      `json:"compressed"`
	Tier       string    `json:"tier"`         // "local" or "remote"
	StoredAt   time.Time `json:"stored_at"`
	AccessedAt time.Time `json:"accessed_at"`
}

// Store is the tiered disk-backed storage engine.
type Store struct {
	mu sync.RWMutex

	// local is the fast tier (SSD/NVMe).
	localPath string
	// remote is the slow tier (NFS/HDD), optional.
	remotePath string

	// In-memory index of all stored blocks.
	index map[string]*BlockMeta // keyed by BlockKey.String()

	// Budget limits.
	localBudget int64
	remoteBudget int64
	localUsed   int64
	remoteUsed  int64

	// Compression.
	compress    bool
	encoder     *zstd.Encoder
	decoder     *zstd.Decoder
}

// Config for creating a new Store.
type Config struct {
	LocalPath    string // Path to local SSD storage directory.
	RemotePath   string // Path to NFS/HDD storage directory (empty to disable).
	LocalBudget  int64  // Max bytes on local tier.
	RemoteBudget int64  // Max bytes on remote tier.
	Compress     bool   // Apply zstd compression.
}

// New creates a new tiered disk store.
func New(cfg Config) (*Store, error) {
	if err := os.MkdirAll(cfg.LocalPath, 0755); err != nil {
		return nil, fmt.Errorf("diskstore: create local dir: %w", err)
	}
	if cfg.RemotePath != "" {
		if err := os.MkdirAll(cfg.RemotePath, 0755); err != nil {
			return nil, fmt.Errorf("diskstore: create remote dir: %w", err)
		}
	}

	var enc *zstd.Encoder
	var dec *zstd.Decoder
	if cfg.Compress {
		var err error
		enc, err = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
		if err != nil {
			return nil, fmt.Errorf("diskstore: create zstd encoder: %w", err)
		}
		dec, err = zstd.NewReader(nil)
		if err != nil {
			return nil, fmt.Errorf("diskstore: create zstd decoder: %w", err)
		}
	}

	s := &Store{
		localPath:    cfg.LocalPath,
		remotePath:   cfg.RemotePath,
		index:        make(map[string]*BlockMeta),
		localBudget:  cfg.LocalBudget,
		remoteBudget: cfg.RemoteBudget,
		compress:     cfg.Compress,
		encoder:      enc,
		decoder:      dec,
	}

	// Load existing index if present.
	s.loadIndex()

	return s, nil
}

// Put stores a KV tensor block to the local tier.
func (s *Store) Put(key BlockKey, dtype string, shape []int, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	payload := data
	compressed := false
	if s.compress && s.encoder != nil {
		payload = s.encoder.EncodeAll(data, nil)
		compressed = true
	}

	// Check local budget; if full, evict oldest local blocks to remote.
	for s.localUsed+int64(len(payload)) > s.localBudget {
		if !s.evictLocalToRemote() {
			break // no remote tier or remote is full
		}
	}

	path := s.blockPath(key, "local")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	if err := os.WriteFile(path, payload, 0644); err != nil {
		return err
	}

	meta := &BlockMeta{
		Key:        key,
		DTypeStr:   dtype,
		Shape:      shape,
		SizeBytes:  len(data),
		Compressed: compressed,
		Tier:       "local",
		StoredAt:   time.Now(),
		AccessedAt: time.Now(),
	}
	s.index[key.String()] = meta
	s.localUsed += int64(len(payload))

	return nil
}

// Get retrieves a KV tensor block. Returns the raw (decompressed) bytes and metadata.
// Returns nil, nil if not found.
func (s *Store) Get(key BlockKey) ([]byte, *BlockMeta, error) {
	s.mu.RLock()
	meta, ok := s.index[key.String()]
	s.mu.RUnlock()

	if !ok {
		return nil, nil, nil
	}

	path := s.blockPath(key, meta.Tier)
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("diskstore: read block %s: %w", key, err)
	}

	data := payload
	if meta.Compressed && s.decoder != nil {
		data, err = s.decoder.DecodeAll(payload, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("diskstore: decompress block %s: %w", key, err)
		}
	}

	s.mu.Lock()
	meta.AccessedAt = time.Now()
	s.mu.Unlock()

	return data, meta, nil
}

// Has checks whether a block exists in the store.
func (s *Store) Has(key BlockKey) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.index[key.String()]
	return ok
}

// GetRange returns all stored blocks for a given sequence, layer, and key/value type
// that overlap with the position range [beginPos, endPos).
func (s *Store) GetRange(seq, layer int, isKey bool, beginPos, endPos int32) []BlockMeta {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []BlockMeta
	for _, meta := range s.index {
		if meta.Key.Seq == seq &&
			meta.Key.Layer == layer &&
			meta.Key.IsKey == isKey &&
			meta.Key.BeginPos < endPos &&
			meta.Key.EndPos > beginPos {
			results = append(results, *meta)
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Key.BeginPos < results[j].Key.BeginPos
	})
	return results
}

// RemoveSeq removes all blocks for a given sequence.
func (s *Store) RemoveSeq(seq int) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	var removed int
	for k, meta := range s.index {
		if meta.Key.Seq == seq {
			path := s.blockPath(meta.Key, meta.Tier)
			os.Remove(path)
			if meta.Tier == "local" {
				s.localUsed -= int64(meta.SizeBytes)
			} else {
				s.remoteUsed -= int64(meta.SizeBytes)
			}
			delete(s.index, k)
			removed++
		}
	}
	return removed
}

// Stats returns storage statistics.
type Stats struct {
	LocalBlocks  int   `json:"local_blocks"`
	RemoteBlocks int   `json:"remote_blocks"`
	LocalUsed    int64 `json:"local_used"`
	RemoteUsed   int64 `json:"remote_used"`
	LocalBudget  int64 `json:"local_budget"`
	RemoteBudget int64 `json:"remote_budget"`
}

func (s *Store) Stats() Stats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var local, remote int
	for _, meta := range s.index {
		if meta.Tier == "local" {
			local++
		} else {
			remote++
		}
	}

	return Stats{
		LocalBlocks:  local,
		RemoteBlocks: remote,
		LocalUsed:    s.localUsed,
		RemoteUsed:   s.remoteUsed,
		LocalBudget:  s.localBudget,
		RemoteBudget: s.remoteBudget,
	}
}

// Close flushes the index and releases resources.
func (s *Store) Close() error {
	s.saveIndex()
	if s.encoder != nil {
		s.encoder.Close()
	}
	if s.decoder != nil {
		s.decoder.Close()
	}
	return nil
}

// ── internal ────────────────────────────────────────────────────────────────

func (s *Store) blockPath(key BlockKey, tier string) string {
	base := s.localPath
	if tier == "remote" {
		base = s.remotePath
	}
	shard := key.Seq % 256
	return filepath.Join(base, fmt.Sprintf("%02x", shard), key.String()+".kvblk")
}

// evictLocalToRemote moves the oldest local block to remote tier.
// Must be called with s.mu held.
func (s *Store) evictLocalToRemote() bool {
	if s.remotePath == "" {
		return false
	}

	// Find oldest local block.
	var oldest *BlockMeta
	for _, meta := range s.index {
		if meta.Tier == "local" {
			if oldest == nil || meta.AccessedAt.Before(oldest.AccessedAt) {
				oldest = meta
			}
		}
	}
	if oldest == nil {
		return false
	}

	// Check remote budget.
	if s.remoteUsed+int64(oldest.SizeBytes) > s.remoteBudget {
		return false
	}

	srcPath := s.blockPath(oldest.Key, "local")
	dstPath := s.blockPath(oldest.Key, "remote")

	if err := os.MkdirAll(filepath.Dir(dstPath), 0755); err != nil {
		return false
	}

	data, err := os.ReadFile(srcPath)
	if err != nil {
		return false
	}
	if err := os.WriteFile(dstPath, data, 0644); err != nil {
		return false
	}
	os.Remove(srcPath)

	s.localUsed -= int64(len(data))
	s.remoteUsed += int64(len(data))
	oldest.Tier = "remote"

	return true
}

func (s *Store) indexPath() string {
	return filepath.Join(s.localPath, "index.json")
}

func (s *Store) saveIndex() {
	data, err := json.MarshalIndent(s.index, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(s.indexPath(), data, 0644)
}

func (s *Store) loadIndex() {
	data, err := os.ReadFile(s.indexPath())
	if err != nil {
		return
	}
	json.Unmarshal(data, &s.index)

	// Recalculate usage.
	for _, meta := range s.index {
		if meta.Tier == "local" {
			s.localUsed += int64(meta.SizeBytes)
		} else {
			s.remoteUsed += int64(meta.SizeBytes)
		}
	}
}

// Uint32Bytes is a helper for encoding position as bytes.
func Uint32Bytes(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}
