//! Recurrent LIF: I = W_in @ x + W_rec @ spike_prev.

use super::spike_straight_through;
use crate::state::{apply_reset, NeuronState, ResetMode};
use crate::surrogate::Surrogate;
use burn::module::{Ignored, Module, Param};
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct RLeaky<B: Backend> {
    pub linear_in: Linear<B>,
    pub linear_rec: Linear<B>,
    pub beta: Param<Tensor<B, 1>>,
    pub threshold: f32,
    pub reset_mode: Ignored<ResetMode>,
}

impl<B: Backend> RLeaky<B> {
    pub fn new(
        device: &B::Device,
        in_features: usize,
        out_features: usize,
        beta_init: f32,
        threshold: f32,
        reset_mode: ResetMode,
    ) -> Self {
        let linear_in = LinearConfig::new(in_features, out_features).init(device);
        let linear_rec = LinearConfig::new(out_features, out_features).init(device);
        let beta_raw = (beta_init / (1.0 - beta_init.max(0.01f32))).ln();
        let beta = Param::from_tensor(Tensor::from_floats([beta_raw], device));
        Self {
            linear_in,
            linear_rec,
            beta,
            threshold,
            reset_mode: Ignored(reset_mode),
        }
    }

    pub fn step<S: Surrogate>(
        &self,
        input: Tensor<B, 2>,
        spike_prev: Tensor<B, 2>,
        state: &NeuronState<B>,
        surrogate: &S,
    ) -> (Tensor<B, 2>, NeuronState<B>) {
        let i_in = self.linear_in.forward(input);
        let i_rec = self.linear_rec.forward(spike_prev);
        let current = i_in + i_rec;
        let b = sigmoid(self.beta.val().clone()).reshape([1, 1]).expand(state.mem.dims());
        let mem_new = state.mem.clone().mul(b) + current;
        let spike = spike_straight_through(mem_new.clone(), self.threshold, surrogate);
        let spike_hard = spike.clone().greater_equal_elem(0.5).float();
        let mem_reset = apply_reset(mem_new, &spike_hard, self.threshold, self.reset_mode.0);
        let new_state = NeuronState {
            mem: mem_reset,
            syn: state.syn.clone(),
        };
        (spike, new_state)
    }
}
