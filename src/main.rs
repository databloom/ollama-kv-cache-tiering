//! kv-cache-tier: Tiered KV-cache for LLM inference.
//!
//! Extends GPU VRAM capacity by transparently paging KV cache blocks
//! through a hierarchy of storage tiers:
//!   GPU VRAM (hot) → Host RAM (warm) → Local SSD (cool) → NFS/HDD (cold)
//!
//! Exposes an OpenAI-compatible HTTP API for drop-in integration.

pub mod cache;
pub mod config;
pub mod gpu;
pub mod inference;
pub mod server;
pub mod transfer;

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{error, info};

use cache::pager::new_shared_pager;
use config::{Cli, Config};
use inference::engine::InferenceEngine;
use server::openai_api::{build_router, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI arguments.
    let cli = Cli::parse();

    // Initialize tracing/logging.
    let filter = if cli.verbose {
        "kv_cache_tier=debug,tower_http=debug"
    } else {
        "kv_cache_tier=info,tower_http=info"
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with_target(true)
        .init();

    info!("kv-cache-tier v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration.
    let config = Config::load(&cli.config)?;
    let config = Arc::new(config);

    info!(
        model = %config.model.model_path.display(),
        context_size = config.model.context_size,
        block_size = config.model.block_size,
        "Configuration loaded"
    );

    // Print tier capacities.
    info!(
        gpu_vram = config.tiers.gpu_vram_budget,
        host_ram = config.tiers.host_ram_budget,
        local_ssd = config.tiers.local_ssd_budget,
        nfs = config.tiers.nfs_budget,
        "Tier capacities"
    );

    // Compute KV cache sizing.
    let block_bytes = config.kv_block_bytes();
    let tokens_gpu = config.tokens_for_budget(config.tiers.gpu_vram_budget);
    let tokens_ram = config.tokens_for_budget(config.tiers.host_ram_budget);
    let tokens_ssd = config.tokens_for_budget(config.tiers.local_ssd_budget);

    info!(
        block_bytes,
        tokens_gpu,
        tokens_ram,
        tokens_ssd,
        total_tokens = tokens_gpu + tokens_ram + tokens_ssd,
        "KV cache capacity (FP16 equivalent)"
    );

    // Initialize the tiered cache pager.
    let pager = new_shared_pager(config.clone());

    // Initialize the inference engine.
    let engine = InferenceEngine::new(pager.clone(), config.clone());

    // Build application state.
    let state = Arc::new(AppState {
        engine: RwLock::new(engine),
        config: config.clone(),
        pager,
        start_time: Instant::now(),
    });

    // Build the HTTP router.
    let app = build_router(state);

    // Start the server.
    let listen_addr = cli.listen;
    info!(addr = listen_addr, "Starting server");

    let listener = TcpListener::bind(&listen_addr).await?;
    info!("Listening on {listen_addr}");

    axum::serve(listener, app).await?;

    Ok(())
}
