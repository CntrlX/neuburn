//! Loss functions and utilities for SNN training.

mod loss;
mod lateral;

pub use lateral::lateral_inhibition;
pub use loss::{ce_rate_loss, ce_temporal_loss, mse_count_loss};
