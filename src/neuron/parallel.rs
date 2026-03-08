//! LeakyParallel: run LIF over the time dimension (batch, time, features).
//! Convenience for vectorized-in-time or sequential stepping.

use super::Leaky;
use super::LeakyConfig;
use crate::state::NeuronState;
use crate::surrogate::Surrogate;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Runs a Leaky (LIF) layer over the time dimension.
/// Input shape: [batch, time, features]. Output shape: [batch, time, features] (spikes).
#[derive(Module, Debug)]
pub struct LeakyParallel<B: Backend> {
    pub lif: Leaky<B>,
}

impl<B: Backend> LeakyParallel<B> {
    pub fn new(config: &LeakyConfig, device: &B::Device) -> Self {
        Self {
            lif: Leaky::new(config, device),
        }
    }

    /// Forward over time: input [batch, time, features], returns spikes [batch, time, features].
    pub fn forward<S: Surrogate>(
        &self,
        input: Tensor<B, 3>,
        surrogate: &S,
    ) -> Tensor<B, 3> {
        let device = input.device();
        let [batch, num_steps, features] = input.dims();
        let mut state = NeuronState::zeros_mem_only(&device, batch, features);
        let mut outputs = Vec::with_capacity(num_steps);
        for t in 0..num_steps {
            let x_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..features])
                .squeeze_dims(&[1]);
            let (spike, new_state) = self.lif.step(x_t, &state, surrogate);
            state = new_state;
            outputs.push(spike);
        }
        Tensor::stack::<3>(outputs, 1)
    }
}
