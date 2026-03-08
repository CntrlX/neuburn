//! Spiking KAN layer: trace + KAN + shortcut + mix + norm + LIF.

use crate::event_driven::EventDrivenKANSynapse;
use crate::neuron::Leaky;
use crate::neuron::LeakyConfig;
use crate::state::{LayerState, NeuronState};
use crate::surrogate::Surrogate;
use burn::module::Module;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::module::Param;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct EventDrivenSpikingKANLayer<B: Backend> {
    pub kan_synapse: EventDrivenKANSynapse<B>,
    pub shortcut: Option<Linear<B>>,
    pub kan_weight: Param<Tensor<B, 1>>,
    pub layer_norm: LayerNorm<B>,
    pub lif: Leaky<B>,
    trace_alpha: f32,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> EventDrivenSpikingKANLayer<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        trace_alpha: f32,
        delta: f32,
        kan_grid_size: usize,
        use_residual: bool,
        device: &B::Device,
    ) -> Self {
        let kan_synapse = EventDrivenKANSynapse::new(
            in_features,
            out_features,
            kan_grid_size,
            delta,
            true,
            device,
        );
        let shortcut = if use_residual && in_features != out_features {
            Some(LinearConfig::new(in_features, out_features).init(device))
        } else {
            None
        };
        let kan_weight = Param::from_tensor(Tensor::from_floats([0.1], device));
        let layer_norm = LayerNormConfig::new(out_features).init(device);
        let lif = Leaky::new(
            &LeakyConfig {
                beta_init: 0.95,
                threshold: 1.0,
                learn_threshold: false,
                reset_mode: crate::state::ResetMode::Subtract,
            },
            device,
        );
        Self {
            kan_synapse,
            shortcut,
            kan_weight,
            layer_norm,
            lif,
            trace_alpha,
            in_features,
            out_features,
        }
    }

    pub fn forward<S: Surrogate>(
        &self,
        x: Tensor<B, 2>,
        state: &LayerState<B>,
        surrogate: &S,
    ) -> (Tensor<B, 2>, LayerState<B>) {
        let trace_new = state
            .trace
            .clone()
            .mul_scalar(self.trace_alpha)
            + x.mul_scalar(1.0 - self.trace_alpha);
        let (kan_out, _, _) = self
            .kan_synapse
            .forward(trace_new.clone(), state.last_trace.clone(), state.last_output.clone());
        let synaptic_current = if let Some(ref sc) = self.shortcut {
            let short = sc.forward(trace_new.clone());
            let mix_val = sigmoid(self.kan_weight.val().clone())
                .into_data()
                .as_slice::<f32>()
                .unwrap()[0];
            kan_out.clone().mul_scalar(mix_val) + short.mul_scalar(1.0 - mix_val)
        } else {
            kan_out.clone()
        };
        let normed = self.layer_norm.forward(synaptic_current);
        let neuron_state = NeuronState {
            mem: state.mem.clone(),
            syn: None,
        };
        let (spike, new_neuron_state) = self.lif.step(normed, &neuron_state, surrogate);
        let new_state = LayerState {
            trace: trace_new.clone(),
            mem: new_neuron_state.mem,
            last_trace: Some(trace_new),
            last_output: Some(kan_out),
        };
        (spike, new_state)
    }
}
