//! Event-driven full SKAN: input_fc → norm → LIF → layer1 → layer2 → readout.

use super::EventDrivenSpikingKANLayer;
use crate::neuron::Leaky;
use crate::neuron::LeakyConfig;
use crate::state::{LayerState, NeuronState};
use crate::surrogate::Surrogate;
use burn::module::Module;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::tensor::backend::Backend;
use burn::module::Param;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct EventDrivenFullSKAN<B: Backend> {
    pub input_fc: Linear<B>,
    pub input_norm: LayerNorm<B>,
    pub input_lif: Leaky<B>,
    pub layer1: EventDrivenSpikingKANLayer<B>,
    pub layer2: EventDrivenSpikingKANLayer<B>,
    pub output_weight: Param<Tensor<B, 1>>,
    num_steps: usize,
    hidden1: usize,
    hidden2: usize,
    output_size: usize,
}

impl<B: Backend> EventDrivenFullSKAN<B> {
    pub fn new(
        input_size: usize,
        hidden1: usize,
        hidden2: usize,
        output_size: usize,
        num_steps: usize,
        beta: f32,
        trace_alpha: f32,
        delta: f32,
        kan_grid_size: usize,
        device: &B::Device,
    ) -> Self {
        let input_fc = LinearConfig::new(input_size, hidden1).init(device);
        let input_norm = LayerNormConfig::new(hidden1).init(device);
        let input_lif = Leaky::new(
            &LeakyConfig {
                beta_init: beta,
                threshold: 1.0,
                learn_threshold: false,
                reset_mode: crate::state::ResetMode::Subtract,
            },
            device,
        );
        let layer1 = EventDrivenSpikingKANLayer::new(
            hidden1,
            hidden2,
            trace_alpha,
            delta,
            kan_grid_size,
            true,
            device,
        );
        let layer2 = EventDrivenSpikingKANLayer::new(
            hidden2,
            output_size,
            trace_alpha,
            delta,
            kan_grid_size,
            false,
            device,
        );
        let output_weight = Param::from_tensor(Tensor::ones([output_size], device));
        Self {
            input_fc,
            input_norm,
            input_lif,
            layer1,
            layer2,
            output_weight,
            num_steps,
            hidden1,
            hidden2,
            output_size,
        }
    }

    pub fn forward<S: Surrogate>(&self, x: Tensor<B, 2>, surrogate: &S) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let device = x.device();
        let batch = x.dims()[0];
        let mut mem_input = NeuronState::zeros_mem_only(&device, batch, self.hidden1);
        let mut layer1_state = LayerState::zeros(
            &device,
            batch,
            self.hidden1,
            self.hidden2,
            true,
        );
        let mut layer2_state = LayerState::zeros(
            &device,
            batch,
            self.hidden2,
            self.output_size,
            true,
        );
        let mut mem_accum = Tensor::zeros([batch, self.output_size], &device);
        let mut spike_count = Tensor::zeros([batch, self.output_size], &device);

        for _ in 0..self.num_steps {
            let cur_input = self.input_norm.forward(self.input_fc.forward(x.clone()));
            let (spk_input, new_mem_input) =
                self.input_lif.step(cur_input, &mem_input, surrogate);
            mem_input = new_mem_input;

            let (spk1, new_l1) = self.layer1.forward(spk_input, &layer1_state, surrogate);
            layer1_state = new_l1;

            let (spk2, new_l2) = self.layer2.forward(spk1, &layer2_state, surrogate);
            layer2_state = new_l2;

            mem_accum = mem_accum + layer2_state.mem.clone();
            spike_count = spike_count + spk2.clone();
        }

        let out = (mem_accum / self.num_steps as f32)
            .mul(self.output_weight.val().reshape([1, self.output_size]).expand([batch, self.output_size]))
            + spike_count.clone();
        (out, spike_count)
    }
}
