//! OpenAI-compatible HTTP API.
//!
//! Implements the subset of the OpenAI API needed for LLM inference:
//! - POST /v1/chat/completions
//! - POST /v1/completions
//! - GET /v1/models
//! - GET /health

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tracing::info;
use uuid::Uuid;

use crate::cache::pager::SharedPager;
use crate::config::Config;
use crate::inference::engine::{GenerationEvent, GenerationRequest, InferenceEngine};
use crate::server::streaming::generation_to_sse_stream;

/// Application state shared across handlers.
pub struct AppState {
    pub engine: RwLock<InferenceEngine>,
    pub config: Arc<Config>,
    pub pager: SharedPager,
    pub start_time: Instant,
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/v1/cache/stats", get(cache_stats))
        .with_state(state)
}

// ─── Request/Response Types ────────────────────────────────────────────────

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.9
}

/// Chat completion response (non-streaming).
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Completion request (non-chat).
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model listing response.
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
    pub cache: CacheStatsResponse,
}

/// Cache statistics response.
#[derive(Debug, Serialize)]
pub struct CacheStatsResponse {
    pub total_blocks: usize,
    pub total_sequences: usize,
    pub tiers: Vec<TierStatsResponse>,
}

#[derive(Debug, Serialize)]
pub struct TierStatsResponse {
    pub name: String,
    pub block_count: usize,
    pub bytes_used: usize,
    pub capacity: usize,
    pub utilization: f64,
}

// ─── Route Handlers ────────────────────────────────────────────────────────

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let request_id = Uuid::new_v4().to_string();

    info!(
        request_id = request_id,
        model = req.model,
        messages = req.messages.len(),
        stream = req.stream,
        "Chat completion request"
    );

    // Concatenate messages into a prompt string for tokenization.
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Stub tokenization: ~1 token per 4 chars.
    let prompt_tokens: Vec<i32> = (0..(prompt.len() / 4).max(1) as i32).collect();
    let prompt_token_count = prompt_tokens.len();

    let gen_request = GenerationRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_tokens: vec![],
    };

    if req.stream {
        // Streaming response via SSE.
        let mut engine = state.engine.write().await;
        let rx = engine.generate(gen_request).await;
        let stream = generation_to_sse_stream(rx, request_id.clone(), req.model.clone());
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming: collect all tokens.
        let mut engine = state.engine.write().await;
        let mut rx = engine.generate(gen_request).await;

        let mut text = String::new();
        let mut completion_tokens = 0;

        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token { text: t, .. } => {
                    text.push_str(&t);
                    completion_tokens += 1;
                }
                GenerationEvent::Done { .. } => break,
                GenerationEvent::Error(e) => {
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{request_id}"),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_token_count,
                completion_tokens,
                total_tokens: prompt_token_count + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let request_id = Uuid::new_v4().to_string();

    let prompt_tokens: Vec<i32> = (0..(req.prompt.len() / 4).max(1) as i32).collect();
    let prompt_token_count = prompt_tokens.len();

    let gen_request = GenerationRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: 1.0,
        stop_tokens: vec![],
    };

    let mut engine = state.engine.write().await;
    let mut rx = engine.generate(gen_request).await;

    let mut text = String::new();
    let mut completion_tokens = 0;

    while let Some(event) = rx.recv().await {
        match event {
            GenerationEvent::Token { text: t, .. } => {
                text.push_str(&t);
                completion_tokens += 1;
            }
            GenerationEvent::Done { .. } => break,
            GenerationEvent::Error(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{request_id}"),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    }))
}

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.model.model_path.display().to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        }],
    })
}

async fn health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let pager = state.pager.read().await;
    let tier_stats: Vec<TierStatsResponse> = pager
        .tier_stats()
        .iter()
        .map(|(tier, stats)| TierStatsResponse {
            name: tier.to_string(),
            block_count: stats.block_count,
            bytes_used: stats.bytes_used,
            capacity: stats.capacity,
            utilization: stats.usage_fraction(),
        })
        .collect();

    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        cache: CacheStatsResponse {
            total_blocks: pager.total_blocks(),
            total_sequences: pager.total_sequences(),
            tiers: tier_stats,
        },
    })
}

async fn cache_stats(
    State(state): State<Arc<AppState>>,
) -> Json<CacheStatsResponse> {
    let pager = state.pager.read().await;
    let tier_stats: Vec<TierStatsResponse> = pager
        .tier_stats()
        .iter()
        .map(|(tier, stats)| TierStatsResponse {
            name: tier.to_string(),
            block_count: stats.block_count,
            bytes_used: stats.bytes_used,
            capacity: stats.capacity,
            utilization: stats.usage_fraction(),
        })
        .collect();

    Json(CacheStatsResponse {
        total_blocks: pager.total_blocks(),
        total_sequences: pager.total_sequences(),
        tiers: tier_stats,
    })
}
