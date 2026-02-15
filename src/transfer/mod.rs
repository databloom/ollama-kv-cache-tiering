//! Data transfer between tiers.
//!
//! - [`gpu_transfer`]: CUDA async memcpy for GPU ↔ RAM transfers
//! - [`disk_io`]: Async disk I/O for RAM ↔ Disk and Disk ↔ NFS transfers
//! - [`dma_scheduler`]: Coordinates overlapping transfers with computation

pub mod disk_io;
pub mod dma_scheduler;
pub mod gpu_transfer;
