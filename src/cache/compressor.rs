//! Compression pipeline for KV cache blocks.
//!
//! Handles quantization (FP16 → Q8 → Q4) and zstd compression
//! for tier transitions. Decompression reverses the pipeline.

use crate::cache::block::{CacheFormat, KvBlock, Tier};
use crate::config::CompressionConfig;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("Zstd compression failed: {0}")]
    ZstdError(#[from] std::io::Error),

    #[error("Quantization failed: source format {from:?} cannot be quantized to {to:?}")]
    InvalidQuantization { from: CacheFormat, to: CacheFormat },

    #[error("Block has no data to compress")]
    NoData,
}

/// The compression engine handles format transitions between tiers.
pub struct Compressor {
    config: CompressionConfig,
}

impl Compressor {
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a block's data for storage in the target tier.
    ///
    /// Applies the appropriate quantization and compression based on
    /// the source format and target tier.
    pub fn compress_for_tier(
        &self,
        block: &KvBlock,
        target_tier: Tier,
    ) -> Result<Vec<u8>, CompressionError> {
        let data = block
            .ram_data
            .as_ref()
            .ok_or(CompressionError::NoData)?;

        match (block.tier, target_tier) {
            // GPU → RAM: optionally quantize FP16 → Q8
            (Tier::Gpu, Tier::Ram) => {
                if self.config.gpu_to_ram_quantize {
                    self.quantize_fp16_to_q8(data)
                } else {
                    Ok(data.clone())
                }
            }
            // RAM → Disk: optionally quantize Q8 → Q4, then zstd
            (Tier::Ram, Tier::LocalDisk) | (Tier::Ram, Tier::Nfs) => {
                let quantized = if self.config.ram_to_disk_quantize {
                    self.quantize_q8_to_q4(data)?
                } else {
                    data.clone()
                };
                if self.config.disk_zstd_compression {
                    self.zstd_compress(&quantized)
                } else {
                    Ok(quantized)
                }
            }
            // Disk → NFS: already compressed, just copy
            (Tier::LocalDisk, Tier::Nfs) => Ok(data.clone()),
            // Same tier or unsupported transition
            _ => Ok(data.clone()),
        }
    }

    /// Decompress data from a given format back toward FP16.
    pub fn decompress(
        &self,
        data: &[u8],
        format: CacheFormat,
    ) -> Result<Vec<u8>, CompressionError> {
        match format {
            CacheFormat::Q4Zstd => {
                let decompressed = self.zstd_decompress(data)?;
                let dequantized = self.dequantize_q4_to_q8(&decompressed)?;
                self.dequantize_q8_to_fp16(&dequantized)
            }
            CacheFormat::Q4 => {
                let dequantized = self.dequantize_q4_to_q8(data)?;
                self.dequantize_q8_to_fp16(&dequantized)
            }
            CacheFormat::Q8 => self.dequantize_q8_to_fp16(data),
            CacheFormat::Fp16 => Ok(data.to_vec()),
        }
    }

    /// Simulate FP16 → Q8 quantization.
    ///
    /// Real implementation would use GGML quantization routines.
    /// This placeholder halves the data size.
    fn quantize_fp16_to_q8(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Placeholder: in a real implementation, each FP16 value (2 bytes) is
        // mapped to a Q8 value (1 byte) using a block-wise scale factor.
        // For now, we take every other byte to simulate 2x compression.
        let output: Vec<u8> = data.iter().step_by(2).copied().collect();
        Ok(output)
    }

    /// Simulate Q8 → Q4 quantization.
    fn quantize_q8_to_q4(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Placeholder: pack two Q8 values into one Q4 byte.
        let mut output = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks(2) {
            let hi = chunk[0] >> 4;
            let lo = chunk.get(1).map(|b| b >> 4).unwrap_or(0);
            output.push((hi << 4) | lo);
        }
        Ok(output)
    }

    /// Simulate Q8 → FP16 dequantization.
    fn dequantize_q8_to_fp16(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Placeholder: expand each Q8 byte back to 2 FP16 bytes.
        let mut output = Vec::with_capacity(data.len() * 2);
        for &byte in data {
            output.push(byte);
            output.push(0); // zero-fill high byte
        }
        Ok(output)
    }

    /// Simulate Q4 → Q8 dequantization.
    fn dequantize_q4_to_q8(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Placeholder: unpack two Q4 nibbles into two Q8 bytes.
        let mut output = Vec::with_capacity(data.len() * 2);
        for &byte in data {
            output.push((byte >> 4) << 4);
            output.push((byte & 0x0F) << 4);
        }
        Ok(output)
    }

    /// Compress data with zstd.
    fn zstd_compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let compressed = zstd::encode_all(data as &[u8], self.config.zstd_level)?;
        Ok(compressed)
    }

    /// Decompress zstd data.
    fn zstd_decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let decompressed = zstd::decode_all(data as &[u8])?;
        Ok(decompressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let compressor = Compressor::new(CompressionConfig::default());
        let data = vec![42u8; 4096];

        let compressed = compressor.zstd_compress(&data).unwrap();
        assert!(compressed.len() < data.len()); // should compress well

        let decompressed = compressor.zstd_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_quantize_reduces_size() {
        let compressor = Compressor::new(CompressionConfig::default());
        let data = vec![128u8; 1024]; // simulated FP16 data

        let q8 = compressor.quantize_fp16_to_q8(&data).unwrap();
        assert_eq!(q8.len(), 512); // 2x compression

        let q4 = compressor.quantize_q8_to_q4(&q8).unwrap();
        assert_eq!(q4.len(), 256); // 2x more compression
    }
}
