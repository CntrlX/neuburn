//! Lapicque (RC circuit) neuron model.

use super::spike_straight_through;
use crate::state::{apply_reset, NeuronState, ResetMode};
use crate::surrogate::Surrogate;
use burn::module::{Ignored, Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Lapicque<B: Backend> {
    pub tau: Param<Tensor<B, 1>>,
    pub threshold: f32,
    pub reset_mode: Ignored<ResetMode>,
}

impl<B: Backend> Lapicque<B> {
    pub fn new(device: &B::Device, tau_init: f32, threshold: f32, reset_mode: ResetMode) -> Self {
        let tau = Param::from_tensor(Tensor::from_floats([tau_init], device));
        Self {
            tau,
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
        let t = self.tau.val().reshape([1, 1]).expand(state.mem.dims());
        let dt = 1.0f32;
        let mem_new = state.mem.clone() + (input - state.mem.clone()).div(t).mul_scalar(dt);
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
