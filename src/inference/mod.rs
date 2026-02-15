//! LLM inference engine.
//!
//! - [`engine`]: High-level inference orchestrator
//! - [`llama_ffi`]: FFI bindings to llama.cpp
//! - [`model_loader`]: GGUF model loading and configuration

pub mod engine;
pub mod llama_ffi;
pub mod model_loader;
