//! SpikingConv2d: Conv2d followed by LIF. Input [B, C_in, H, W] -> spikes [B, C_out, H', W'].

use crate::neuron::{Leaky, LeakyConfig};
use crate::state::NeuronState;
use crate::surrogate::Surrogate;
use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Conv2d followed by a LIF layer. Applies LIF per spatial location over channels.
#[derive(Module, Debug)]
pub struct SpikingConv2d<B: Backend> {
    pub conv: Conv2d<B>,
    pub lif: Leaky<B>,
    out_channels: usize,
}

impl<B: Backend> SpikingConv2d<B> {
    pub fn new(
        conv_config: Conv2dConfig,
        lif_config: &LeakyConfig,
        device: &B::Device,
    ) -> Self {
        let conv = conv_config.init(device);
        let lif = Leaky::new(lif_config, device);
        let out_channels = conv_config.channels[1];
        Self {
            conv,
            lif,
            out_channels,
        }
    }

    /// One step: input [batch, c_in, h, w], state.mem [batch * h' * w', c_out].
    /// Returns (spikes [batch, c_out, h', w'], new_state).
    pub fn step<S: Surrogate>(
        &self,
        input: Tensor<B, 4>,
        state: &NeuronState<B>,
        surrogate: &S,
    ) -> (Tensor<B, 4>, NeuronState<B>) {
        let out = self.conv.forward(input);
        let [batch, c_out, h, w] = out.dims();
        let flat = out.reshape([batch * h * w, c_out]);
        let (spikes_2d, new_state) = self.lif.step(flat, state, surrogate);
        let spikes_4d = spikes_2d.reshape([batch, c_out, h, w]);
        (spikes_4d, new_state)
    }

    /// Create zero state for the given 4D output shape (e.g. after one forward to get dimensions).
    pub fn state_zeros(device: &B::Device, batch: usize, c_out: usize, h: usize, w: usize) -> NeuronState<B> {
        NeuronState::zeros_mem_only(device, batch * h * w, c_out)
    }
}
