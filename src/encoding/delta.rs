//! Delta (change-based) encoding: spikes when change exceeds threshold.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Delta encoding: compare data to previous; output spike representation of changes.
/// For the first step, prev can be zeros. Returns [batch, features] float: +1 for pos, -1 for neg, 0 else.
pub fn delta_encode<B: Backend>(
    data: Tensor<B, 2>,
    prev: Option<Tensor<B, 2>>,
    threshold: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let prev = prev.unwrap_or_else(|| Tensor::zeros(data.dims(), device));
    let diff = data.clone() - prev;
    let pos = diff.clone().greater_equal_elem(threshold).float();
    let neg = diff.lower_equal_elem(-threshold).float();
    pos - neg
}
