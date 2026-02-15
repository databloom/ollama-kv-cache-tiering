//! GPU device management and VRAM allocation.
//!
//! - [`device`]: GPU device discovery and info
//! - [`allocator`]: Block-based VRAM allocator for KV cache

pub mod allocator;
pub mod device;
