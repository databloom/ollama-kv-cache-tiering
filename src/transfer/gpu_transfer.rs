//! GPU ↔ Host RAM transfer engine.
//!
//! Uses CUDA async memory copies to overlap data movement with computation.
//! When the `cuda` feature is disabled, provides stub implementations for
//! CPU-only testing.

use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info};

use crate::cache::block::{BlockId, GpuLocation, Tier};

#[derive(Error, Debug)]
pub enum GpuTransferError {
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Block not found on GPU: {0}")]
    BlockNotOnGpu(BlockId),

    #[error("GPU device {0} not available")]
    DeviceNotAvailable(usize),

    #[error("Transfer buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
}

/// A pending GPU transfer operation.
#[derive(Debug)]
pub struct GpuTransferOp {
    pub block_id: BlockId,
    pub direction: TransferDirection,
    pub size_bytes: usize,
    pub device_id: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum TransferDirection {
    /// Device to Host (GPU → RAM).
    DeviceToHost,
    /// Host to Device (RAM → GPU).
    HostToDevice,
}

/// GPU transfer engine.
///
/// Manages async memory copies between GPU VRAM and host RAM.
/// When compiled without CUDA, uses stub implementations that
/// simply copy data in host memory (for testing).
pub struct GpuTransferEngine {
    /// Number of available GPU devices.
    device_count: usize,

    /// Staging buffers per device for async transfers.
    staging_buffers: Vec<Vec<u8>>,

    /// Transfer statistics.
    stats: TransferStats,
}

#[derive(Debug, Default)]
pub struct TransferStats {
    pub total_d2h_bytes: u64,
    pub total_h2d_bytes: u64,
    pub total_d2h_transfers: u64,
    pub total_h2d_transfers: u64,
}

impl GpuTransferEngine {
    /// Create a new transfer engine.
    ///
    /// `device_count`: number of GPU devices.
    /// `staging_buffer_size`: size of per-device staging buffer in bytes.
    pub fn new(device_count: usize, staging_buffer_size: usize) -> Self {
        let staging_buffers = (0..device_count)
            .map(|_| vec![0u8; staging_buffer_size])
            .collect();

        Self {
            device_count,
            staging_buffers,
            stats: TransferStats::default(),
        }
    }

    /// Copy block data from GPU to host RAM (Device-to-Host).
    ///
    /// In a real CUDA implementation, this would use `cudaMemcpyAsync`
    /// with a dedicated copy stream to overlap with compute.
    pub async fn copy_to_host(
        &mut self,
        gpu_location: &GpuLocation,
        _block_id: BlockId,
    ) -> Result<Vec<u8>, GpuTransferError> {
        if gpu_location.device_id >= self.device_count {
            return Err(GpuTransferError::DeviceNotAvailable(gpu_location.device_id));
        }

        // Stub: in a real implementation this would be:
        // 1. cudarc::driver::CudaDevice::dtoh_sync_copy() or async variant
        // 2. Using a pinned host buffer for better throughput
        debug!(
            device = gpu_location.device_id,
            offset = gpu_location.offset,
            size = gpu_location.size,
            "D2H transfer"
        );

        // For now, return a zero-filled buffer of the right size.
        let data = vec![0u8; gpu_location.size];

        self.stats.total_d2h_bytes += gpu_location.size as u64;
        self.stats.total_d2h_transfers += 1;

        Ok(data)
    }

    /// Copy block data from host RAM to GPU (Host-to-Device).
    pub async fn copy_to_device(
        &mut self,
        data: &[u8],
        gpu_location: &GpuLocation,
        _block_id: BlockId,
    ) -> Result<(), GpuTransferError> {
        if gpu_location.device_id >= self.device_count {
            return Err(GpuTransferError::DeviceNotAvailable(gpu_location.device_id));
        }

        if data.len() > gpu_location.size {
            return Err(GpuTransferError::BufferTooSmall {
                needed: data.len(),
                available: gpu_location.size,
            });
        }

        // Stub: in a real implementation this would be:
        // cudarc::driver::CudaDevice::htod_sync_copy() or async variant
        debug!(
            device = gpu_location.device_id,
            offset = gpu_location.offset,
            size = data.len(),
            "H2D transfer"
        );

        self.stats.total_h2d_bytes += data.len() as u64;
        self.stats.total_h2d_transfers += 1;

        Ok(())
    }

    /// Get transfer statistics.
    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }

    /// Number of available GPU devices.
    pub fn device_count(&self) -> usize {
        self.device_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_d2h_transfer() {
        let mut engine = GpuTransferEngine::new(2, 1024 * 1024);

        let loc = GpuLocation {
            device_id: 0,
            offset: 0,
            size: 4096,
        };

        let data = engine.copy_to_host(&loc, 0).await.unwrap();
        assert_eq!(data.len(), 4096);
        assert_eq!(engine.stats().total_d2h_transfers, 1);
    }

    #[tokio::test]
    async fn test_h2d_transfer() {
        let mut engine = GpuTransferEngine::new(2, 1024 * 1024);
        let data = vec![42u8; 2048];

        let loc = GpuLocation {
            device_id: 1,
            offset: 0,
            size: 4096,
        };

        engine.copy_to_device(&data, &loc, 0).await.unwrap();
        assert_eq!(engine.stats().total_h2d_transfers, 1);
        assert_eq!(engine.stats().total_h2d_bytes, 2048);
    }

    #[tokio::test]
    async fn test_invalid_device() {
        let mut engine = GpuTransferEngine::new(1, 1024);

        let loc = GpuLocation {
            device_id: 5,
            offset: 0,
            size: 1024,
        };

        let result = engine.copy_to_host(&loc, 0).await;
        assert!(result.is_err());
    }
}
