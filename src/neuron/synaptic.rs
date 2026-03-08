//! LIF with synaptic conductance traces.

use super::spike_straight_through;
use crate::state::{apply_reset, NeuronState, ResetMode};
use crate::surrogate::Surrogate;
use burn::module::{Ignored, Module, Param};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Synaptic<B: Backend> {
    pub beta: Param<Tensor<B, 1>>,
    pub alpha_syn: Param<Tensor<B, 1>>,
    pub threshold: f32,
    pub reset_mode: Ignored<ResetMode>,
}

impl<B: Backend> Synaptic<B> {
    pub fn new(
        device: &B::Device,
        beta_init: f32,
        alpha_syn_init: f32,
        threshold: f32,
        reset_mode: ResetMode,
    ) -> Self {
        let beta_raw = (beta_init / (1.0 - beta_init.max(0.01f32))).ln();
        let beta = Param::from_tensor(Tensor::from_floats([beta_raw], device));
        let alpha_raw = (alpha_syn_init / (1.0 - alpha_syn_init.max(0.01f32))).ln();
        let alpha_syn = Param::from_tensor(Tensor::from_floats([alpha_raw], device));
        Self {
            beta,
            alpha_syn,
            threshold,
            reset_mode: Ignored(reset_mode),
        }
    }

    pub fn step<S: Surrogate>(
        &self,
        input: Tensor<B, 2>,
        state: &NeuronState<B>,
        surrogate: &S,
    ) -> (Tensor<B, 2>, NeuronState<B>) {
        let syn = state.syn.as_ref().unwrap_or(&state.mem);
        let b = sigmoid(self.beta.val().clone()).reshape([1, 1]).expand(state.mem.dims());
        let a = sigmoid(self.alpha_syn.val().clone())
            .reshape([1, 1])
            .expand(state.mem.dims());
        let syn_new = syn.clone().mul(a) + input.clone();
        let mem_new = state.mem.clone().mul(b) + syn_new.clone();
        let spike = spike_straight_through(mem_new.clone(), self.threshold, surrogate);
        let spike_hard = spike.clone().greater_equal_elem(0.5).float();
        let mem_reset = apply_reset(mem_new, &spike_hard, self.threshold, self.reset_mode.0);
        let new_state = NeuronState {
            mem: mem_reset,
            syn: Some(syn_new),
        };
        (spike, new_state)
    }
}
