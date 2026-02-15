package diskstore

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPutAndGet(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024, // 1 MB
		Compress:    false,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	key := BlockKey{Seq: 0, Layer: 3, BeginPos: 100, EndPos: 101, IsKey: true}
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i % 256)
	}

	if err := store.Put(key, "f16", []int{128, 8, 1}, data); err != nil {
		t.Fatalf("Put: %v", err)
	}

	got, meta, err := store.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if meta == nil {
		t.Fatal("Get returned nil meta")
	}
	if len(got) != len(data) {
		t.Fatalf("Get: got %d bytes, want %d", len(got), len(data))
	}
	for i := range got {
		if got[i] != data[i] {
			t.Fatalf("Get: byte %d: got %d, want %d", i, got[i], data[i])
		}
	}
}

func TestPutAndGetCompressed(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024,
		Compress:    true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	key := BlockKey{Seq: 1, Layer: 0, BeginPos: 0, EndPos: 1, IsKey: false}
	// Highly compressible data.
	data := make([]byte, 8192)
	for i := range data {
		data[i] = 42
	}

	if err := store.Put(key, "f16", []int{128, 8, 1}, data); err != nil {
		t.Fatalf("Put: %v", err)
	}

	got, meta, err := store.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !meta.Compressed {
		t.Error("expected compressed=true")
	}
	if len(got) != len(data) {
		t.Fatalf("Get: got %d bytes, want %d", len(got), len(data))
	}

	// Verify on-disk size is smaller than original.
	path := store.blockPath(key, "local")
	fi, _ := os.Stat(path)
	if fi.Size() >= int64(len(data)) {
		t.Errorf("compressed file (%d) should be smaller than original (%d)", fi.Size(), len(data))
	}
}

func TestEvictLocalToRemote(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:    filepath.Join(dir, "local"),
		RemotePath:   filepath.Join(dir, "remote"),
		LocalBudget:  5000,      // very small local budget
		RemoteBudget: 1024 * 1024,
		Compress:     false,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	// Fill local past budget → should trigger eviction to remote.
	for i := 0; i < 5; i++ {
		key := BlockKey{Seq: 0, Layer: 0, BeginPos: int32(i), EndPos: int32(i + 1), IsKey: true}
		data := make([]byte, 2000) // 5 × 2000 = 10000 > 5000 budget
		if err := store.Put(key, "f16", []int{128, 1}, data); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	stats := store.Stats()
	if stats.RemoteBlocks == 0 {
		t.Error("expected some blocks on remote tier after exceeding local budget")
	}

	// Verify we can still read evicted blocks.
	for i := 0; i < 5; i++ {
		key := BlockKey{Seq: 0, Layer: 0, BeginPos: int32(i), EndPos: int32(i + 1), IsKey: true}
		got, _, err := store.Get(key)
		if err != nil {
			t.Fatalf("Get %d: %v", i, err)
		}
		if got == nil {
			t.Fatalf("Get %d: returned nil", i)
		}
	}
}

func TestRemoveSeq(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	// Add blocks for two sequences.
	for seq := 0; seq < 2; seq++ {
		for i := 0; i < 3; i++ {
			key := BlockKey{Seq: seq, Layer: 0, BeginPos: int32(i), EndPos: int32(i + 1), IsKey: true}
			store.Put(key, "f16", []int{128}, make([]byte, 100))
		}
	}

	removed := store.RemoveSeq(0)
	if removed != 3 {
		t.Errorf("RemoveSeq: removed %d, want 3", removed)
	}

	// Seq 0 should be gone.
	key := BlockKey{Seq: 0, Layer: 0, BeginPos: 0, EndPos: 1, IsKey: true}
	if store.Has(key) {
		t.Error("seq 0 block still present after RemoveSeq")
	}

	// Seq 1 should still be there.
	key = BlockKey{Seq: 1, Layer: 0, BeginPos: 0, EndPos: 1, IsKey: true}
	if !store.Has(key) {
		t.Error("seq 1 block missing after removing seq 0")
	}
}

func TestGetRange(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	// Store positions 0-9 for layer 0, key.
	for i := int32(0); i < 10; i++ {
		key := BlockKey{Seq: 0, Layer: 0, BeginPos: i, EndPos: i + 1, IsKey: true}
		store.Put(key, "f16", []int{128}, make([]byte, 64))
	}

	// Query range [3, 7).
	results := store.GetRange(0, 0, true, 3, 7)
	if len(results) != 4 {
		t.Errorf("GetRange: got %d results, want 4", len(results))
	}
	if results[0].Key.BeginPos != 3 {
		t.Errorf("GetRange: first result pos=%d, want 3", results[0].Key.BeginPos)
	}
}

func TestHas(t *testing.T) {
	dir := t.TempDir()
	store, err := New(Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	key := BlockKey{Seq: 0, Layer: 0, BeginPos: 0, EndPos: 1, IsKey: true}
	if store.Has(key) {
		t.Error("Has: should be false before Put")
	}

	store.Put(key, "f16", []int{128}, make([]byte, 64))
	if !store.Has(key) {
		t.Error("Has: should be true after Put")
	}
}

func TestIndexPersistence(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		LocalPath:   filepath.Join(dir, "local"),
		LocalBudget: 1024 * 1024,
	}

	store, _ := New(cfg)
	key := BlockKey{Seq: 0, Layer: 0, BeginPos: 42, EndPos: 43, IsKey: true}
	store.Put(key, "f16", []int{128}, make([]byte, 256))
	store.Close()

	// Reopen — should recover the index.
	store2, _ := New(cfg)
	defer store2.Close()
	if !store2.Has(key) {
		t.Error("index not persisted across close/reopen")
	}
}
