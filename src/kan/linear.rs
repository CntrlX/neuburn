//! B-spline based KAN linear layer: base path (SiLU + linear) + spline path.

use burn::module::Module;
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::module::Param;
use burn::tensor::{Distribution, Tensor};

/// KAN linear layer: y = silu(x) @ W_base + spline_component(x).
/// Spline component: x expanded to basis then matmul with spline weights (simplified).
#[derive(Module, Debug)]
pub struct KANLinear<B: Backend> {
    /// Base weight [out_features, in_features].
    pub base_weight: Param<Tensor<B, 2>>,
    /// Spline weight [out_features, in_features * n_coeff] for basis expansion.
    pub spline_weight: Param<Tensor<B, 2>>,
    in_features: usize,
    out_features: usize,
    n_coeff: usize,
}

impl<B: Backend> KANLinear<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        spline_order: usize,
        _grid_range: [f32; 2],
        scale_base: f32,
        scale_spline: f32,
        device: &B::Device,
    ) -> Self {
        let n_coeff = grid_size + spline_order;
        let base_weight = Param::from_tensor(Tensor::random(
            [out_features, in_features],
            Distribution::Uniform(-scale_base as f64, scale_base as f64),
            device,
        ));
        let sc = scale_spline as f64 / n_coeff as f64;
        let spline_weight = Param::from_tensor(Tensor::random(
            [out_features, in_features * n_coeff],
            Distribution::Uniform(-sc, sc),
            device,
        ));
        Self {
            base_weight,
            spline_weight,
            in_features,
            out_features,
            n_coeff,
        }
    }

    /// Simple basis: expand x (placeholder for full B-spline Cox-de Boor).
    fn basis(&self, x: Tensor<B, 2>, _device: &B::Device) -> Tensor<B, 2> {
        let dims = x.dims();
        let batch = dims[0];
        let x3 = x.unsqueeze_dim::<3>(2).expand([batch, self.in_features, self.n_coeff]);
        x3.reshape([batch, self.in_features * self.n_coeff])
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = x.device();
        let base_act = silu(x.clone());
        let base_out = base_act.matmul(self.base_weight.val().transpose());
        let basis = self.basis(x, &device);
        let spline_out = basis.matmul(self.spline_weight.val().transpose());
        base_out + spline_out
    }
}
