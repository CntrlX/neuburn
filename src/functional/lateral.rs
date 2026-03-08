//! Lateral inhibition: subtract the maximum spike in the layer (per batch item).
//! Encourages sparse, winner-take-all-like activity.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Apply lateral inhibition to spike tensor [batch, units]:
/// subtract the max along the feature dimension so the strongest response is zero.
/// Result is not necessarily in [0, 1]; clamp or use as-is for downstream.
pub fn lateral_inhibition<B: Backend>(spikes: Tensor<B, 2>) -> Tensor<B, 2> {
    let max_val = spikes.clone().max_dim(1);
    spikes - max_val
}
