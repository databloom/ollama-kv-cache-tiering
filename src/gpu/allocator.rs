//! GPU VRAM block allocator for KV cache.
//!
//! Manages a pre-allocated VRAM region as a pool of fixed-size blocks.
//! Uses a simple free-list allocator with O(1) alloc/free.

use std::collections::VecDeque;

use thiserror::Error;
use tracing::debug;

use crate::cache::block::GpuLocation;

#[derive(Error, Debug)]
pub enum AllocatorError {
    #[error("Out of GPU VRAM: no free blocks on device {device_id}")]
    OutOfMemory { device_id: usize },

    #[error("Block not found at offset {offset} on device {device_id}")]
    BlockNotFound { device_id: usize, offset: usize },

    #[error("Device {0} not initialized")]
    DeviceNotInitialized(usize),
}

/// Per-device VRAM allocator.
#[derive(Debug)]
struct DeviceAllocator {
    /// Device ID.
    device_id: usize,

    /// Block size in bytes.
    block_size: usize,

    /// Total number of blocks.
    total_blocks: usize,

    /// Free block offsets.
    free_list: VecDeque<usize>,

    /// Number of allocated blocks.
    allocated: usize,
}

impl DeviceAllocator {
    fn new(device_id: usize, total_vram: usize, block_size: usize) -> Self {
        let total_blocks = total_vram / block_size;
        let free_list: VecDeque<usize> = (0..total_blocks)
            .map(|i| i * block_size)
            .collect();

        Self {
            device_id,
            block_size,
            total_blocks,
            free_list,
            allocated: 0,
        }
    }

    fn allocate(&mut self) -> Result<GpuLocation, AllocatorError> {
        match self.free_list.pop_front() {
            Some(offset) => {
                self.allocated += 1;
                Ok(GpuLocation {
                    device_id: self.device_id,
                    offset,
                    size: self.block_size,
                })
            }
            None => Err(AllocatorError::OutOfMemory {
                device_id: self.device_id,
            }),
        }
    }

    fn free(&mut self, offset: usize) -> Result<(), AllocatorError> {
        if offset % self.block_size != 0 || offset / self.block_size >= self.total_blocks {
            return Err(AllocatorError::BlockNotFound {
                device_id: self.device_id,
                offset,
            });
        }
        self.free_list.push_back(offset);
        self.allocated = self.allocated.saturating_sub(1);
        Ok(())
    }

    fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.allocated as f64 / self.total_blocks as f64
    }
}

/// Multi-device VRAM allocator.
///
/// Manages block allocation across multiple GPUs.
pub struct VramAllocator {
    /// Per-device allocators.
    devices: Vec<DeviceAllocator>,

    /// Block size in bytes.
    block_size: usize,
}

impl VramAllocator {
    /// Create a new allocator for the given devices.
    ///
    /// `device_vram`: list of (device_id, vram_budget_bytes) pairs.
    /// `block_size`: size of each block in bytes.
    pub fn new(device_vram: &[(usize, usize)], block_size: usize) -> Self {
        let devices = device_vram
            .iter()
            .map(|&(id, vram)| DeviceAllocator::new(id, vram, block_size))
            .collect();

        Self {
            devices,
            block_size,
        }
    }

    /// Allocate a block on the specified device.
    pub fn allocate(&mut self, device_id: usize) -> Result<GpuLocation, AllocatorError> {
        let dev = self
            .devices
            .iter_mut()
            .find(|d| d.device_id == device_id)
            .ok_or(AllocatorError::DeviceNotInitialized(device_id))?;

        let loc = dev.allocate()?;
        debug!(
            device = device_id,
            offset = loc.offset,
            "Allocated GPU block"
        );
        Ok(loc)
    }

    /// Allocate a block on whichever device has the most free space.
    pub fn allocate_best(&mut self) -> Result<GpuLocation, AllocatorError> {
        let best_device = self
            .devices
            .iter()
            .filter(|d| !d.free_list.is_empty())
            .max_by_key(|d| d.free_list.len())
            .map(|d| d.device_id)
            .ok_or(AllocatorError::OutOfMemory { device_id: 0 })?;

        self.allocate(best_device)
    }

    /// Free a block.
    pub fn free(&mut self, location: &GpuLocation) -> Result<(), AllocatorError> {
        let dev = self
            .devices
            .iter_mut()
            .find(|d| d.device_id == location.device_id)
            .ok_or(AllocatorError::DeviceNotInitialized(location.device_id))?;

        dev.free(location.offset)?;
        debug!(
            device = location.device_id,
            offset = location.offset,
            "Freed GPU block"
        );
        Ok(())
    }

    /// Get utilization for each device as (device_id, fraction).
    pub fn utilization(&self) -> Vec<(usize, f64)> {
        self.devices
            .iter()
            .map(|d| (d.device_id, d.utilization()))
            .collect()
    }

    /// Total free blocks across all devices.
    pub fn total_free(&self) -> usize {
        self.devices.iter().map(|d| d.free_list.len()).sum()
    }

    /// Total allocated blocks across all devices.
    pub fn total_allocated(&self) -> usize {
        self.devices.iter().map(|d| d.allocated).sum()
    }

    /// Block size in bytes.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut alloc = VramAllocator::new(&[(0, 4096), (1, 4096)], 1024);

        // Allocate 4 blocks on device 0 (fills it).
        let locs: Vec<_> = (0..4).map(|_| alloc.allocate(0).unwrap()).collect();
        assert_eq!(alloc.total_allocated(), 4);

        // Device 0 should be full.
        assert!(alloc.allocate(0).is_err());

        // Free one.
        alloc.free(&locs[0]).unwrap();
        assert_eq!(alloc.total_allocated(), 3);

        // Can allocate again.
        alloc.allocate(0).unwrap();
    }

    #[test]
    fn test_allocate_best() {
        let mut alloc = VramAllocator::new(&[(0, 2048), (1, 4096)], 1024);

        // Best should pick device 1 (more free space).
        let loc = alloc.allocate_best().unwrap();
        assert_eq!(loc.device_id, 1);
    }

    #[test]
    fn test_utilization() {
        let mut alloc = VramAllocator::new(&[(0, 4096)], 1024);

        let util = alloc.utilization();
        assert_eq!(util[0].1, 0.0);

        alloc.allocate(0).unwrap();
        alloc.allocate(0).unwrap();

        let util = alloc.utilization();
        assert!((util[0].1 - 0.5).abs() < 1e-10);
    }
}
