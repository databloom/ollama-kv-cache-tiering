//! HTTP server providing an OpenAI-compatible API.
//!
//! - [`openai_api`]: Request/response types and route handlers
//! - [`streaming`]: SSE streaming for token-by-token responses

pub mod openai_api;
pub mod streaming;
