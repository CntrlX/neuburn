//! Poisson rate coding: sample spikes from rates over num_steps.

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Poisson rate encoding: data in [0,1] -> spike train over num_steps.
/// Returns tensor of shape [batch, num_steps, features].
pub fn rate_encode<B: Backend>(
    data: Tensor<B, 2>,
    num_steps: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let shape = data.dims();
    let batch = shape[0];
    let features = shape[1];
    let mut spikes = Vec::with_capacity(num_steps);
    for _ in 0..num_steps {
        let r: Tensor<B, 2> =
            Tensor::random([batch, features], Distribution::Uniform(0.0, 1.0), device);
        let spike = r.lower_equal(data.clone());
        spikes.push(spike.float());
    }
    Tensor::stack(spikes, 1)
}
