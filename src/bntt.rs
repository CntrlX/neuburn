//! BatchNorm through time (BNTT): separate BatchNorm parameters per time step.
//! Used in SNNs to normalize activations at each time step independently.

use burn::module::Module;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// BatchNorm through time for 1D (dense) features.
/// Input shape: [batch, channels, length]. One BatchNorm per time step.
#[derive(Module, Debug)]
pub struct BatchNormTT1d<B: Backend> {
    /// One BatchNorm layer per time step.
    pub layers: Vec<BatchNorm<B, 1>>,
}

impl<B: Backend> BatchNormTT1d<B> {
    pub fn new(num_steps: usize, num_features: usize, device: &B::Device) -> Self {
        let config = BatchNormConfig::new(num_features);
        let layers = (0..num_steps)
            .map(|_| config.init::<B, 1>(device))
            .collect();
        Self { layers }
    }

    /// Forward at the given time step. Input: [batch, channels, length].
    pub fn forward(&self, step_index: usize, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layers[step_index].forward(x)
    }
}

/// BatchNorm through time for 2D (spatial) features.
/// Input shape: [batch, channels, height, width]. One BatchNorm per time step.
#[derive(Module, Debug)]
pub struct BatchNormTT2d<B: Backend> {
    /// One BatchNorm layer per time step.
    pub layers: Vec<BatchNorm<B, 2>>,
}

impl<B: Backend> BatchNormTT2d<B> {
    pub fn new(num_steps: usize, num_features: usize, device: &B::Device) -> Self {
        let config = BatchNormConfig::new(num_features);
        let layers = (0..num_steps)
            .map(|_| config.init::<B, 2>(device))
            .collect();
        Self { layers }
    }

    /// Forward at the given time step. Input: [batch, channels, height, width].
    pub fn forward(&self, step_index: usize, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.layers[step_index].forward(x)
    }
}
