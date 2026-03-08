//! Surrogate gradient trait: smooth surrogate used in backward pass while forward uses Heaviside.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Surrogate for spike nonlinearity: forward uses Heaviside, backward uses this smooth function.
/// Implementors provide the smooth value σ(x) so that straight-through is:
/// `spike = heaviside(x).detach() + (surrogate_forward(x) - surrogate_forward(x).detach())`.
pub trait Surrogate: Send + Sync {
    /// Compute the smooth surrogate value for the given input (e.g. membrane - threshold).
    /// Used in the straight-through expression so gradients flow through this.
    fn surrogate_forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}
