//! Integration tests for the full inference pipeline.

use std::sync::Arc;
use std::time::Instant;

use kv_cache_tier::cache::pager::new_shared_pager;
use kv_cache_tier::config::Config;
use kv_cache_tier::inference::engine::{GenerationEvent, GenerationRequest, InferenceEngine};

#[tokio::test]
async fn test_full_generation_pipeline() {
    let config = Arc::new(Config::default());
    let pager = new_shared_pager(config.clone());
    let mut engine = InferenceEngine::new(pager, config);

    let request = GenerationRequest {
        request_id: "integration-test-1".to_string(),
        prompt_tokens: vec![1, 2, 3, 4, 5],
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
        stop_tokens: vec![],
    };

    let mut rx = engine.generate(request).await;

    let mut tokens = Vec::new();
    let mut done = false;

    while let Some(event) = rx.recv().await {
        match event {
            GenerationEvent::Token { token_id, .. } => {
                tokens.push(token_id);
            }
            GenerationEvent::Done {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            } => {
                assert_eq!(prompt_tokens, 5);
                assert_eq!(completion_tokens, 10);
                assert_eq!(total_tokens, 15);
                done = true;
            }
            GenerationEvent::Error(e) => panic!("Unexpected error: {e}"),
        }
    }

    assert_eq!(tokens.len(), 10);
    assert!(done);
}

#[tokio::test]
async fn test_multiple_sequences() {
    let config = Arc::new(Config::default());
    let pager = new_shared_pager(config.clone());
    let mut engine = InferenceEngine::new(pager, config);

    // Run two sequences concurrently.
    let req1 = GenerationRequest {
        request_id: "seq-1".to_string(),
        prompt_tokens: vec![1, 2],
        max_tokens: 3,
        temperature: 0.0,
        top_p: 1.0,
        stop_tokens: vec![],
    };
    let req2 = GenerationRequest {
        request_id: "seq-2".to_string(),
        prompt_tokens: vec![10, 20, 30],
        max_tokens: 5,
        temperature: 0.0,
        top_p: 1.0,
        stop_tokens: vec![],
    };

    let mut rx1 = engine.generate(req1).await;
    let mut rx2 = engine.generate(req2).await;

    let mut count1 = 0;
    let mut count2 = 0;

    loop {
        tokio::select! {
            Some(event) = rx1.recv() => {
                if matches!(event, GenerationEvent::Token { .. }) {
                    count1 += 1;
                }
                if matches!(event, GenerationEvent::Done { .. }) {
                    if count2 >= 5 { break; }
                }
            }
            Some(event) = rx2.recv() => {
                if matches!(event, GenerationEvent::Token { .. }) {
                    count2 += 1;
                }
                if matches!(event, GenerationEvent::Done { .. }) {
                    if count1 >= 3 { break; }
                }
            }
            else => break,
        }
    }

    assert_eq!(count1, 3);
    assert_eq!(count2, 5);
}

#[tokio::test]
async fn test_stop_token() {
    let config = Arc::new(Config::default());
    let pager = new_shared_pager(config.clone());
    let mut engine = InferenceEngine::new(pager, config);

    // Token generation produces token_id = i % 100.
    // Setting stop_token = 3 should stop after 4 tokens (0, 1, 2, 3).
    let request = GenerationRequest {
        request_id: "stop-test".to_string(),
        prompt_tokens: vec![1],
        max_tokens: 100,
        temperature: 0.0,
        top_p: 1.0,
        stop_tokens: vec![3],
    };

    let mut rx = engine.generate(request).await;

    let mut count = 0;
    while let Some(event) = rx.recv().await {
        if matches!(event, GenerationEvent::Token { .. }) {
            count += 1;
        }
    }

    assert_eq!(count, 4); // tokens 0, 1, 2, 3 (stops after generating stop token)
}
