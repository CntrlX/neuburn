//! Leaky Integrate-and-Fire (LIF) neuron: U' = β·U + I, Heaviside spike, configurable reset.

use super::spike_straight_through_with_threshold;
use crate::state::{apply_reset, NeuronState, ResetMode};
use crate::surrogate::Surrogate;
use burn::module::{Ignored, Module, Param};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for Leaky (LIF) neuron.
#[derive(Debug, Clone)]
pub struct LeakyConfig {
    /// Initial membrane decay β in (0, 1).
    pub beta_init: f32,
    /// Spike threshold (used when learn_threshold is false).
    pub threshold: f32,
    /// If true, threshold is a learnable parameter (positive via softplus).
    pub learn_threshold: bool,
    /// Reset mode after spike.
    pub reset_mode: ResetMode,
}

impl Default for LeakyConfig {
    fn default() -> Self {
        Self {
            beta_init: 0.9,
            threshold: 1.0,
            learn_threshold: false,
            reset_mode: ResetMode::Subtract,
        }
    }
}

/// Leaky Integrate-and-Fire layer. Learnable β; optional learnable threshold.
#[derive(Module, Debug)]
pub struct Leaky<B: Backend> {
    /// Raw beta (sigmoid(beta) = decay in (0,1)). Shape [1].
    pub beta: Param<Tensor<B, 1>>,
    /// Fixed threshold (used when learn_threshold was false).
    pub threshold: f32,
    /// Learnable threshold raw (softplus gives V_th); only present when learn_threshold.
    pub threshold_param: Option<Param<Tensor<B, 1>>>,
    /// Reset mode after spike (not part of trainable record).
    pub reset_mode: Ignored<ResetMode>,
}

impl<B: Backend> Leaky<B> {
    pub fn new(config: &LeakyConfig, device: &B::Device) -> Self {
        let beta_raw = if config.beta_init <= 0.0 || config.beta_init >= 1.0 {
            0.0f32
        } else {
            (config.beta_init / (1.0 - config.beta_init)).ln()
        };
        let beta = Param::from_tensor(Tensor::from_floats([beta_raw], device));
        let threshold_param = if config.learn_threshold {
            let raw = (config.threshold.max(0.001f32)).ln();
            Some(Param::from_tensor(Tensor::from_floats([raw], device)))
        } else {
            None
        };
        Self {
            beta,
            threshold: config.threshold,
            threshold_param,
            reset_mode: Ignored(config.reset_mode),
        }
    }

    fn threshold_tensor(&self, device: &B::Device, dims: [usize; 2]) -> Tensor<B, 2> {
        if let Some(ref p) = self.threshold_param {
            let v = p.val();
            let pos = v.clone().exp().add_scalar(1.0).log();
            pos.reshape([1, 1]).expand(dims)
        } else {
            Tensor::<B, 1>::from_floats([self.threshold], device)
                .reshape([1, 1])
                .expand(dims)
        }
    }

    /// One step: input [batch, units], state.mem [batch, units]. Returns (spikes, new_state).
    pub fn step<S: Surrogate>(
        &self,
        input: Tensor<B, 2>,
        state: &NeuronState<B>,
        surrogate: &S,
    ) -> (Tensor<B, 2>, NeuronState<B>) {
        let dims = state.mem.dims();
        let device = state.mem.device();
        let b = sigmoid(self.beta.val().clone());
        let beta = b.reshape([1, 1]).expand(dims);
        let mem_new = state.mem.clone().mul(beta) + input;
        let th = self.threshold_tensor(&device, dims);
        let spike = spike_straight_through_with_threshold(mem_new.clone(), th.clone(), surrogate);
        let spike_hard = spike.clone().greater_equal_elem(0.5).float();
        let th_scalar = if let Some(ref p) = self.threshold_param {
            let v = p.val();
            v.clone().exp().add_scalar(1.0).log().into_data().as_slice::<f32>().unwrap()[0]
        } else {
            self.threshold
        };
        let mem_reset = apply_reset(mem_new, &spike_hard, th_scalar, self.reset_mode.0);
        let new_state = NeuronState {
            mem: mem_reset,
            syn: state.syn.clone(),
        };
        (spike, new_state)
    }
}
