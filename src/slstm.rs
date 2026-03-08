//! Spiking LSTM: LSTM whose output is passed through a spike (surrogate) nonlinearity.
//! Input [batch, seq_len, input_size] -> spiked output [batch, seq_len, hidden_size].

use crate::surrogate::Surrogate;
use burn::module::Module;
use burn::nn::Lstm;
use burn::nn::LstmConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Spiking LSTM: LSTM with spike nonlinearity on the hidden output at each time step.
#[derive(Module, Debug)]
pub struct SLSTM<B: Backend> {
    pub lstm: Lstm<B>,
    pub threshold: f32,
}

impl<B: Backend> SLSTM<B> {
    pub fn new(d_input: usize, d_hidden: usize, threshold: f32, device: &B::Device) -> Self {
        let config = LstmConfig::new(d_input, d_hidden, true);
        let lstm = config.init(device);
        Self { lstm, threshold }
    }

    /// Forward: run LSTM then apply spike to each time step's hidden output.
    /// Input [batch, seq_len, d_input]. Output [batch, seq_len, d_hidden] (spiked).
    pub fn forward<S: Surrogate>(
        &self,
        x: Tensor<B, 3>,
        surrogate: &S,
    ) -> Tensor<B, 3> {
        let (out, _state) = self.lstm.forward(x, None);
        let [batch, seq, hidden] = out.dims();
        let flat = out.reshape([batch * seq, hidden]);
        let x = flat.sub_scalar(self.threshold);
        let spike_hard = x.clone().greater_equal_elem(0.0).float();
        let spike_surrogate = surrogate.surrogate_forward(x);
        let spiked_flat = spike_hard.detach() + (spike_surrogate.clone() - spike_surrogate.detach());
        spiked_flat.reshape([batch, seq, hidden])
    }
}
