//! DMA scheduler: coordinates overlapping data transfers with computation.
//!
//! Manages a queue of transfer operations and executes them asynchronously,
//! allowing GPU compute to proceed while data moves between tiers.

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::cache::block::{BlockId, Tier};

/// A scheduled transfer operation.
#[derive(Debug, Clone)]
pub struct TransferOp {
    /// Block being transferred.
    pub block_id: BlockId,

    /// Source tier.
    pub from: Tier,

    /// Destination tier.
    pub to: Tier,

    /// Priority (higher = more urgent).
    pub priority: u32,

    /// Whether this is a prefetch (can be cancelled if not needed).
    pub is_prefetch: bool,
}

/// Status of a transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Result of a completed transfer.
#[derive(Debug)]
pub struct TransferResult {
    pub block_id: BlockId,
    pub status: TransferStatus,
    pub bytes_transferred: usize,
    pub duration_us: u64,
}

/// The DMA scheduler manages a priority queue of transfer operations.
pub struct DmaScheduler {
    /// Pending operations ordered by priority.
    queue: VecDeque<TransferOp>,

    /// Maximum concurrent transfers.
    max_concurrent: usize,

    /// Currently in-flight transfer count.
    in_flight: usize,

    /// Statistics.
    stats: DmaStats,
}

#[derive(Debug, Default)]
pub struct DmaStats {
    pub total_scheduled: u64,
    pub total_completed: u64,
    pub total_cancelled: u64,
    pub total_failed: u64,
}

impl DmaScheduler {
    /// Create a new DMA scheduler.
    ///
    /// `max_concurrent`: maximum number of simultaneous transfers.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_concurrent,
            in_flight: 0,
            stats: DmaStats::default(),
        }
    }

    /// Schedule a transfer operation.
    pub fn schedule(&mut self, op: TransferOp) {
        // Insert in priority order (higher priority first).
        let pos = self
            .queue
            .iter()
            .position(|existing| existing.priority < op.priority)
            .unwrap_or(self.queue.len());

        debug!(
            block_id = op.block_id,
            from = %op.from,
            to = %op.to,
            priority = op.priority,
            "Scheduled transfer"
        );

        self.queue.insert(pos, op);
        self.stats.total_scheduled += 1;
    }

    /// Dequeue the next transfer if there's capacity.
    pub fn next(&mut self) -> Option<TransferOp> {
        if self.in_flight >= self.max_concurrent {
            return None;
        }

        if let Some(op) = self.queue.pop_front() {
            self.in_flight += 1;
            Some(op)
        } else {
            None
        }
    }

    /// Mark a transfer as completed.
    pub fn complete(&mut self, block_id: BlockId, success: bool) {
        self.in_flight = self.in_flight.saturating_sub(1);
        if success {
            self.stats.total_completed += 1;
        } else {
            self.stats.total_failed += 1;
        }
    }

    /// Cancel all pending prefetch operations (e.g., when a sequence is freed).
    pub fn cancel_prefetches(&mut self) -> usize {
        let before = self.queue.len();
        self.queue.retain(|op| !op.is_prefetch);
        let cancelled = before - self.queue.len();
        self.stats.total_cancelled += cancelled as u64;
        cancelled
    }

    /// Cancel all pending transfers for a specific block.
    pub fn cancel_block(&mut self, block_id: BlockId) -> bool {
        let before = self.queue.len();
        self.queue.retain(|op| op.block_id != block_id);
        let removed = before != self.queue.len();
        if removed {
            self.stats.total_cancelled += 1;
        }
        removed
    }

    /// Number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Number of in-flight transfers.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> &DmaStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = DmaScheduler::new(4);

        scheduler.schedule(TransferOp {
            block_id: 1,
            from: Tier::Ram,
            to: Tier::Gpu,
            priority: 10,
            is_prefetch: false,
        });
        scheduler.schedule(TransferOp {
            block_id: 2,
            from: Tier::LocalDisk,
            to: Tier::Ram,
            priority: 50,
            is_prefetch: true,
        });
        scheduler.schedule(TransferOp {
            block_id: 3,
            from: Tier::Ram,
            to: Tier::Gpu,
            priority: 100,
            is_prefetch: false,
        });

        // Highest priority first.
        let op = scheduler.next().unwrap();
        assert_eq!(op.block_id, 3);
        assert_eq!(op.priority, 100);

        let op = scheduler.next().unwrap();
        assert_eq!(op.block_id, 2);
    }

    #[test]
    fn test_max_concurrent() {
        let mut scheduler = DmaScheduler::new(1);

        scheduler.schedule(TransferOp {
            block_id: 1,
            from: Tier::Ram,
            to: Tier::Gpu,
            priority: 10,
            is_prefetch: false,
        });
        scheduler.schedule(TransferOp {
            block_id: 2,
            from: Tier::Ram,
            to: Tier::Gpu,
            priority: 10,
            is_prefetch: false,
        });

        // Can dequeue one.
        assert!(scheduler.next().is_some());
        // Cannot dequeue another (at max concurrent).
        assert!(scheduler.next().is_none());

        // Complete the first.
        scheduler.complete(1, true);
        // Now can dequeue.
        assert!(scheduler.next().is_some());
    }

    #[test]
    fn test_cancel_prefetches() {
        let mut scheduler = DmaScheduler::new(4);

        scheduler.schedule(TransferOp {
            block_id: 1,
            from: Tier::Ram,
            to: Tier::Gpu,
            priority: 10,
            is_prefetch: false,
        });
        scheduler.schedule(TransferOp {
            block_id: 2,
            from: Tier::LocalDisk,
            to: Tier::Ram,
            priority: 10,
            is_prefetch: true,
        });

        let cancelled = scheduler.cancel_prefetches();
        assert_eq!(cancelled, 1);
        assert_eq!(scheduler.pending_count(), 1);
    }
}
