//! Async disk I/O for KV cache blocks.
//!
//! Handles reading/writing blocks to local SSD and NFS storage.
//! Uses tokio's async file I/O (with io_uring on supported kernels).

use std::path::{Path, PathBuf};

use thiserror::Error;
use tokio::fs;
use tracing::{debug, warn};

use crate::cache::block::{BlockId, Tier};

#[derive(Error, Debug)]
pub enum DiskIoError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Block file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Storage path not configured for tier {0:?}")]
    PathNotConfigured(Tier),
}

/// Disk I/O engine for reading and writing KV cache blocks.
pub struct DiskIoEngine {
    /// Base path for local SSD storage.
    local_ssd_path: PathBuf,

    /// Base path for NFS storage (optional).
    nfs_path: Option<PathBuf>,

    /// Transfer statistics.
    stats: DiskIoStats,
}

#[derive(Debug, Default)]
pub struct DiskIoStats {
    pub total_writes: u64,
    pub total_reads: u64,
    pub total_bytes_written: u64,
    pub total_bytes_read: u64,
}

impl DiskIoEngine {
    /// Create a new disk I/O engine.
    pub async fn new(
        local_ssd_path: PathBuf,
        nfs_path: Option<PathBuf>,
    ) -> Result<Self, DiskIoError> {
        // Ensure directories exist.
        fs::create_dir_all(&local_ssd_path).await?;
        if let Some(ref nfs) = nfs_path {
            fs::create_dir_all(nfs).await?;
        }

        Ok(Self {
            local_ssd_path,
            nfs_path,
            stats: DiskIoStats::default(),
        })
    }

    /// Generate the file path for a block in a given tier.
    fn block_path(&self, block_id: BlockId, tier: Tier) -> Result<PathBuf, DiskIoError> {
        let base = match tier {
            Tier::LocalDisk => &self.local_ssd_path,
            Tier::Nfs => self.nfs_path.as_ref().ok_or(DiskIoError::PathNotConfigured(tier))?,
            _ => return Err(DiskIoError::PathNotConfigured(tier)),
        };

        // Use a two-level directory structure to avoid too many files in one directory.
        // block_id 12345 → 12/12345.kvblock
        let shard = block_id / 1000;
        Ok(base.join(format!("{shard}")).join(format!("{block_id}.kvblock")))
    }

    /// Write a block's data to disk.
    pub async fn write_block(
        &mut self,
        block_id: BlockId,
        data: &[u8],
        tier: Tier,
    ) -> Result<PathBuf, DiskIoError> {
        let path = self.block_path(block_id, tier)?;

        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(&path, data).await?;

        debug!(
            block_id,
            path = %path.display(),
            size = data.len(),
            tier = ?tier,
            "Wrote block to disk"
        );

        self.stats.total_writes += 1;
        self.stats.total_bytes_written += data.len() as u64;

        Ok(path)
    }

    /// Read a block's data from disk.
    pub async fn read_block(
        &mut self,
        block_id: BlockId,
        tier: Tier,
    ) -> Result<Vec<u8>, DiskIoError> {
        let path = self.block_path(block_id, tier)?;

        if !path.exists() {
            return Err(DiskIoError::FileNotFound(path));
        }

        let data = fs::read(&path).await?;

        debug!(
            block_id,
            path = %path.display(),
            size = data.len(),
            tier = ?tier,
            "Read block from disk"
        );

        self.stats.total_reads += 1;
        self.stats.total_bytes_read += data.len() as u64;

        Ok(data)
    }

    /// Delete a block's file from disk.
    pub async fn delete_block(
        &self,
        block_id: BlockId,
        tier: Tier,
    ) -> Result<(), DiskIoError> {
        let path = self.block_path(block_id, tier)?;

        if path.exists() {
            fs::remove_file(&path).await?;
            debug!(block_id, path = %path.display(), "Deleted block file");
        }

        Ok(())
    }

    /// Copy a block from one tier to another on disk (e.g., SSD → NFS).
    pub async fn copy_block(
        &mut self,
        block_id: BlockId,
        from_tier: Tier,
        to_tier: Tier,
    ) -> Result<PathBuf, DiskIoError> {
        let data = self.read_block(block_id, from_tier).await?;
        self.write_block(block_id, &data, to_tier).await
    }

    /// Get disk I/O statistics.
    pub fn stats(&self) -> &DiskIoStats {
        &self.stats
    }

    /// Get disk usage for a tier's storage path.
    pub async fn disk_usage(&self, tier: Tier) -> Result<u64, DiskIoError> {
        let base = match tier {
            Tier::LocalDisk => &self.local_ssd_path,
            Tier::Nfs => self.nfs_path.as_ref().ok_or(DiskIoError::PathNotConfigured(tier))?,
            _ => return Err(DiskIoError::PathNotConfigured(tier)),
        };

        let mut total = 0u64;
        let mut entries = fs::read_dir(base).await?;
        while let Some(entry) = entries.next_entry().await? {
            let meta = entry.metadata().await?;
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                // Recurse one level (our shard dirs).
                let mut sub_entries = fs::read_dir(entry.path()).await?;
                while let Some(sub) = sub_entries.next_entry().await? {
                    let sub_meta = sub.metadata().await?;
                    if sub_meta.is_file() {
                        total += sub_meta.len();
                    }
                }
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_and_read_block() {
        let tmp = TempDir::new().unwrap();
        let ssd_path = tmp.path().join("ssd");
        let mut engine = DiskIoEngine::new(ssd_path, None).await.unwrap();

        let data = vec![42u8; 4096];
        let path = engine.write_block(0, &data, Tier::LocalDisk).await.unwrap();
        assert!(path.exists());

        let read_data = engine.read_block(0, Tier::LocalDisk).await.unwrap();
        assert_eq!(read_data, data);
    }

    #[tokio::test]
    async fn test_delete_block() {
        let tmp = TempDir::new().unwrap();
        let ssd_path = tmp.path().join("ssd");
        let mut engine = DiskIoEngine::new(ssd_path, None).await.unwrap();

        let data = vec![1u8; 1024];
        engine.write_block(5, &data, Tier::LocalDisk).await.unwrap();
        engine.delete_block(5, Tier::LocalDisk).await.unwrap();

        let result = engine.read_block(5, Tier::LocalDisk).await;
        assert!(result.is_err());
    }
}
