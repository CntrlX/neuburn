//! SNN loss functions: count, rate, and temporal.

use burn::tensor::backend::Backend;
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::Tensor;

/// MSE loss between total spike counts and target counts.
/// spike_counts: [batch, num_classes], target_counts: [batch, num_classes].
pub fn mse_count_loss<B: Backend>(
    spike_counts: Tensor<B, 2>,
    target_counts: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = spike_counts - target_counts;
    diff.clone().mul(diff).mean()
}

/// Cross-entropy on mean firing rates (rate-coded readout).
/// rates: [batch, num_classes], target_probs: [batch, num_classes] (e.g. one-hot as float).
pub fn ce_rate_loss<B: Backend>(
    rates: Tensor<B, 2>,
    target_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_with_logits(rates, target_probs).mean()
}

/// Cross-entropy on logits vs target class probabilities (e.g. for temporal or rate readout).
/// logits: [batch, num_classes], target_probs: [batch, num_classes].
pub fn ce_temporal_loss<B: Backend>(
    logits: Tensor<B, 2>,
    target_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_with_logits(logits, target_probs).mean()
}
