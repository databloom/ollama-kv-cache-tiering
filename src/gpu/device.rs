//! GPU device discovery and information.
//!
//! Detects available GPUs and their VRAM capacity.
//! When compiled without the `cuda` feature, provides stub info.

use serde::{Deserialize, Serialize};
use tracing::info;

/// Information about a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device index.
    pub id: usize,

    /// Device name (e.g., "NVIDIA GeForce GTX 1070").
    pub name: String,

    /// Total VRAM in bytes.
    pub total_vram: usize,

    /// Free VRAM in bytes (at detection time).
    pub free_vram: usize,

    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),

    /// PCIe bandwidth in bytes/sec (theoretical max).
    pub pcie_bandwidth: u64,
}

/// Detect all available GPU devices.
///
/// With the `cuda` feature enabled, uses CUDA runtime to enumerate devices.
/// Without it, returns an empty list (CPU-only mode).
pub fn detect_devices() -> Vec<GpuDeviceInfo> {
    #[cfg(feature = "cuda")]
    {
        detect_devices_cuda()
    }

    #[cfg(not(feature = "cuda"))]
    {
        info!("CUDA not enabled, running in CPU-only mode");
        Vec::new()
    }
}

#[cfg(feature = "cuda")]
fn detect_devices_cuda() -> Vec<GpuDeviceInfo> {
    // Real implementation would use cudarc to enumerate devices.
    // This is a compile-time gated stub that would be filled in
    // when cudarc is available.
    todo!("Implement CUDA device detection with cudarc")
}

/// Create stub GPU device info for testing.
///
/// Simulates the hardware available in the target cluster:
/// - Molly: 2x GTX 1070 (8 GB each)
/// - Wintermute: 2x Quadro M6000 (24 GB each)
pub fn stub_devices_molly() -> Vec<GpuDeviceInfo> {
    vec![
        GpuDeviceInfo {
            id: 0,
            name: "NVIDIA GeForce GTX 1070".to_string(),
            total_vram: 8 * 1024 * 1024 * 1024,       // 8 GB
            free_vram: 7 * 1024 * 1024 * 1024,         // ~7 GB free
            compute_capability: (6, 1),
            pcie_bandwidth: 12_000_000_000,             // ~12 GB/s PCIe 3.0 x16
        },
        GpuDeviceInfo {
            id: 1,
            name: "NVIDIA GeForce GTX 1070".to_string(),
            total_vram: 8 * 1024 * 1024 * 1024,
            free_vram: 7 * 1024 * 1024 * 1024,
            compute_capability: (6, 1),
            pcie_bandwidth: 12_000_000_000,
        },
    ]
}

pub fn stub_devices_wintermute() -> Vec<GpuDeviceInfo> {
    vec![
        GpuDeviceInfo {
            id: 0,
            name: "NVIDIA Quadro M6000".to_string(),
            total_vram: 24 * 1024 * 1024 * 1024,      // 24 GB
            free_vram: 22 * 1024 * 1024 * 1024,        // ~22 GB free
            compute_capability: (5, 2),
            pcie_bandwidth: 12_000_000_000,
        },
        GpuDeviceInfo {
            id: 1,
            name: "NVIDIA Quadro M6000".to_string(),
            total_vram: 24 * 1024 * 1024 * 1024,
            free_vram: 22 * 1024 * 1024 * 1024,
            compute_capability: (5, 2),
            pcie_bandwidth: 12_000_000_000,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_devices() {
        let molly = stub_devices_molly();
        assert_eq!(molly.len(), 2);
        assert_eq!(molly[0].total_vram, 8 * 1024 * 1024 * 1024);

        let wintermute = stub_devices_wintermute();
        assert_eq!(wintermute.len(), 2);
        assert_eq!(wintermute[0].total_vram, 24 * 1024 * 1024 * 1024);
    }
}
