//! Straight-through estimator: backward gradient is 1 (identity); forward still Heaviside.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::Surrogate;

/// Straight-through estimator: surrogate value equals input (gradient 1 in backward).
#[derive(Debug, Clone, Default)]
pub struct StraightThrough;

impl Surrogate for StraightThrough {
    fn surrogate_forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x
    }
}
