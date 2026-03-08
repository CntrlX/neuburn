//! Neuron and layer state types (membrane potential, synaptic current, trace).
//! State is not part of Module/Record; it is passed in/out of step().
//! ResetMode controls how membrane potential is reset after a spike (snntorch-style).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// How to reset membrane potential after a spike (mirrors snntorch reset_mechanism).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResetMode {
    /// Subtract threshold from membrane at spike locations (default).
    #[default]
    Subtract,
    /// Set membrane to zero where the neuron spiked.
    Zero,
    /// No reset; membrane retains value (e.g. for rate-coded readouts).
    None,
}

/// Apply reset to membrane: `mem_new`, `spike_hard` (0/1), `threshold` (for Subtract).
pub fn apply_reset<B: Backend>(
    mem_new: Tensor<B, 2>,
    spike_hard: &Tensor<B, 2>,
    threshold: f32,
    mode: ResetMode,
) -> Tensor<B, 2> {
    match mode {
        ResetMode::Subtract => mem_new - spike_hard.clone().mul_scalar(threshold),
        ResetMode::Zero => mem_new.clone() * (spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0)),
        ResetMode::None => mem_new,
    }
}

/// Hidden state for a single neuron layer: membrane potential and optional synaptic/trace.
/// Shape of each tensor: `[batch_size, units]`.
#[derive(Debug, Clone)]
pub struct NeuronState<B: Backend> {
    /// Membrane potential (batch, units).
    pub mem: Tensor<B, 2>,
    /// Synaptic current or conductance trace (batch, units). Optional for simple LIF.
    pub syn: Option<Tensor<B, 2>>,
}

impl<B: Backend> NeuronState<B> {
    /// Create zero state for the given device and dimensions.
    /// Use this to initialize hidden state at the start of a sequence (snntorch-style init_hidden).
    pub fn zeros(device: &B::Device, batch_size: usize, units: usize) -> Self {
        let mem = Tensor::zeros([batch_size, units], device);
        let syn = Tensor::zeros([batch_size, units], device);
        Self {
            mem,
            syn: Some(syn),
        }
    }

    /// Create state with only membrane (no syn); syn will be None.
    /// Use for LIF/Leaky/RLeaky layers that don't use synaptic traces.
    pub fn zeros_mem_only(device: &B::Device, batch_size: usize, units: usize) -> Self {
        let mem = Tensor::zeros([batch_size, units], device);
        Self { mem, syn: None }
    }

    /// Create from existing tensors (e.g. for inference where caller manages state).
    pub fn new(mem: Tensor<B, 2>, syn: Option<Tensor<B, 2>>) -> Self {
        Self { mem, syn }
    }
}

/// Extended state for event-driven layers: trace + membrane + optional delta-gating buffers.
#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    /// Synaptic trace (batch, in_features).
    pub trace: Tensor<B, 2>,
    /// Membrane potential (batch, out_features).
    pub mem: Tensor<B, 2>,
    /// For delta-gated KAN: last trace input (optional).
    pub last_trace: Option<Tensor<B, 2>>,
    /// For delta-gated KAN: last KAN output (optional).
    pub last_output: Option<Tensor<B, 2>>,
}

impl<B: Backend> LayerState<B> {
    /// Create zero state for event-driven layer.
    pub fn zeros(
        device: &B::Device,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        with_kan_cache: bool,
    ) -> Self {
        let trace = Tensor::zeros([batch_size, in_features], device);
        let mem = Tensor::zeros([batch_size, out_features], device);
        let (last_trace, last_output) = if with_kan_cache {
            (
                Some(Tensor::zeros([batch_size, in_features], device)),
                Some(Tensor::zeros([batch_size, out_features], device)),
            )
        } else {
            (None, None)
        };
        Self {
            trace,
            mem,
            last_trace,
            last_output,
        }
    }
}
