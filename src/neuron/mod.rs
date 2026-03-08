//! Spiking neuron modules (LIF, Synaptic, Alpha, RLeaky, Lapicque).

mod alpha;
mod lapicque;
mod leaky;
mod parallel;
mod rleaky;
mod rsynaptic;
mod synaptic;

pub use alpha::Alpha;
pub use lapicque::Lapicque;
pub use leaky::{Leaky, LeakyConfig};
pub use parallel::LeakyParallel;
pub use rleaky::RLeaky;
pub use rsynaptic::RSynaptic;
pub use synaptic::Synaptic;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Shared spike computation: Heaviside forward, surrogate backward (straight-through).
pub(crate) fn spike_straight_through<B: Backend, S: crate::surrogate::Surrogate>(
    mem: Tensor<B, 2>,
    threshold: f32,
    surrogate: &S,
) -> Tensor<B, 2> {
    let x = mem.sub_scalar(threshold);
    let spike_hard = x.clone().greater_equal_elem(0.0).float();
    let spike_surrogate = surrogate.surrogate_forward(x);
    spike_hard.detach() + (spike_surrogate.clone() - spike_surrogate.detach())
}

/// Spike with tensor threshold (e.g. learnable per-neuron or broadcast).
pub(crate) fn spike_straight_through_with_threshold<B: Backend, S: crate::surrogate::Surrogate>(
    mem: Tensor<B, 2>,
    threshold: Tensor<B, 2>,
    surrogate: &S,
) -> Tensor<B, 2> {
    let x = mem - threshold;
    let spike_hard = x.clone().greater_equal_elem(0.0).float();
    let spike_surrogate = surrogate.surrogate_forward(x);
    spike_hard.detach() + (spike_surrogate.clone() - spike_surrogate.detach())
}
