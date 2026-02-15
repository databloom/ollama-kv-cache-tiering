//! Inference orchestrator: coordinates model execution with tiered KV cache.
//!
//! The engine is the top-level component that:
//! 1. Receives tokenized prompts
//! 2. Manages KV cache block allocation and tier placement
//! 3. Drives the decode loop (one token at a time)
//! 4. Triggers eviction/prefetch between decode steps
//! 5. Returns generated tokens via a streaming channel

use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::cache::block::{BlockTable, CacheFormat, KvBlock, Tier};
use crate::cache::pager::SharedPager;
use crate::config::Config;
use crate::inference::llama_ffi::{LlamaContext, LlamaModel, TokenId};

/// A generation request.
#[derive(Debug)]
pub struct GenerationRequest {
    /// Unique request ID.
    pub request_id: String,

    /// Input token IDs (prompt).
    pub prompt_tokens: Vec<TokenId>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    pub top_p: f64,

    /// Stop sequences (as token IDs).
    pub stop_tokens: Vec<TokenId>,
}

/// A generated token event.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A new token was generated.
    Token {
        token_id: TokenId,
        text: String,
    },
    /// Generation is complete.
    Done {
        total_tokens: usize,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// An error occurred during generation.
    Error(String),
}

/// The inference engine.
pub struct InferenceEngine {
    /// Tiered cache pager.
    pager: SharedPager,

    /// Configuration.
    config: Arc<Config>,

    /// Next sequence ID.
    next_seq_id: u64,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(pager: SharedPager, config: Arc<Config>) -> Self {
        Self {
            pager,
            config,
            next_seq_id: 0,
        }
    }

    /// Run a generation request, streaming tokens to the returned receiver.
    ///
    /// This is the main entry point for inference. It:
    /// 1. Allocates a sequence in the block table
    /// 2. Processes the prompt (prefill)
    /// 3. Generates tokens one at a time (decode)
    /// 4. Manages KV cache tiers between decode steps
    pub async fn generate(
        &mut self,
        request: GenerationRequest,
    ) -> mpsc::Receiver<GenerationEvent> {
        let (tx, rx) = mpsc::channel(32);

        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let pager = self.pager.clone();
        let config = self.config.clone();
        let max_tokens = request.max_tokens;
        let prompt_len = request.prompt_tokens.len();

        tokio::spawn(async move {
            info!(
                request_id = request.request_id,
                prompt_tokens = prompt_len,
                max_tokens,
                "Starting generation"
            );

            // Create block table for this sequence.
            {
                let mut pager = pager.write().await;
                pager.get_or_create_sequence(seq_id);
            }

            // Simulate token generation.
            let mut generated = 0;
            for i in 0..max_tokens {
                // Check if we need to evict blocks from any tier.
                {
                    let mut pager = pager.write().await;
                    if let Some(tier) = pager.needs_eviction() {
                        match pager.evict(tier).await {
                            Ok(n) => debug!(evicted = n, tier = %tier, "Eviction complete"),
                            Err(e) => warn!("Eviction failed: {e}"),
                        }
                    }
                }

                // Stub: generate a token.
                // In a real implementation, this would:
                // 1. Ensure required KV blocks are on GPU
                // 2. Call llama_decode for the new token
                // 3. Call llama_sample to get the next token
                // 4. Allocate new KV blocks as the context grows
                let token_id = (i % 100) as TokenId;
                let text = format!("token_{i}");

                generated += 1;

                if tx
                    .send(GenerationEvent::Token {
                        token_id,
                        text,
                    })
                    .await
                    .is_err()
                {
                    // Receiver dropped, stop generating.
                    break;
                }

                // Check for stop tokens.
                if request.stop_tokens.contains(&token_id) {
                    break;
                }
            }

            let _ = tx
                .send(GenerationEvent::Done {
                    total_tokens: prompt_len + generated,
                    prompt_tokens: prompt_len,
                    completion_tokens: generated,
                })
                .await;

            // Clean up sequence.
            {
                let mut pager = pager.write().await;
                pager.remove_sequence(seq_id);
            }

            info!(
                request_id = request.request_id,
                generated,
                "Generation complete"
            );
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::pager::new_shared_pager;

    #[tokio::test]
    async fn test_generation_produces_tokens() {
        let config = Arc::new(Config::default());
        let pager = new_shared_pager(config.clone());
        let mut engine = InferenceEngine::new(pager, config);

        let request = GenerationRequest {
            request_id: "test-1".to_string(),
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 5,
            temperature: 0.0,
            top_p: 1.0,
            stop_tokens: vec![],
        };

        let mut rx = engine.generate(request).await;

        let mut token_count = 0;
        let mut got_done = false;
        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token { .. } => token_count += 1,
                GenerationEvent::Done { completion_tokens, .. } => {
                    assert_eq!(completion_tokens, 5);
                    got_done = true;
                }
                GenerationEvent::Error(e) => panic!("Unexpected error: {e}"),
            }
        }

        assert_eq!(token_count, 5);
        assert!(got_done);
    }
}
