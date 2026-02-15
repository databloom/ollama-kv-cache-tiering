//! SSE (Server-Sent Events) streaming for token-by-token responses.
//!
//! Converts a channel of GenerationEvents into an SSE stream compatible
//! with the OpenAI streaming format.

use axum::response::sse::Event;
use futures::stream::Stream;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::inference::engine::GenerationEvent;

/// Streaming chat completion chunk (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Convert a generation event receiver into an SSE stream.
pub fn generation_to_sse_stream(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let stream = ReceiverStream::new(rx);
    let mut first = true;

    stream
        .map(move |event| {
            let chunk = match event {
                GenerationEvent::Token { text, .. } => {
                    let mut delta = ChunkDelta {
                        role: None,
                        content: Some(text),
                    };

                    // First chunk includes the role.
                    if first {
                        delta.role = Some("assistant".to_string());
                        first = false;
                    }

                    ChatCompletionChunk {
                        id: format!("chatcmpl-{request_id}"),
                        object: "chat.completion.chunk".to_string(),
                        created: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta,
                            finish_reason: None,
                        }],
                    }
                }
                GenerationEvent::Done { .. } => ChatCompletionChunk {
                    id: format!("chatcmpl-{request_id}"),
                    object: "chat.completion.chunk".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                },
                GenerationEvent::Error(e) => ChatCompletionChunk {
                    id: format!("chatcmpl-{request_id}"),
                    object: "chat.completion.chunk".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(format!("[Error: {e}]")),
                        },
                        finish_reason: Some("error".to_string()),
                    }],
                },
            };

            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(data))
        })
        // Append the [DONE] sentinel after all events.
        .chain(tokio_stream::once(Ok(
            Event::default().data("[DONE]"),
        )))
}
