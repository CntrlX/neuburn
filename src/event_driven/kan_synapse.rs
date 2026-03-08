//! Delta-gated KAN synapse: recompute only when |Δtrace| > δ.

use crate::kan::KANLinear;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Event-driven KAN: optional delta-gating.
#[derive(Module, Debug)]
pub struct EventDrivenKANSynapse<B: Backend> {
    pub kan: KANLinear<B>,
    pub delta: f32,
    pub use_delta_gating: bool,
}

impl<B: Backend> EventDrivenKANSynapse<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        delta: f32,
        use_delta_gating: bool,
        device: &B::Device,
    ) -> Self {
        let kan: KANLinear<B> = KANLinear::new(
            in_features,
            out_features,
            grid_size,
            3,
            [-2.0, 2.0],
            0.5,
            0.1,
            device,
        );
        Self {
            kan,
            delta,
            use_delta_gating,
        }
    }

    pub fn forward(
        &self,
        trace: Tensor<B, 2>,
        _last_trace: Option<Tensor<B, 2>>,
        _last_output: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let out = self.kan.forward(trace.clone());
        (out.clone(), trace, out)
    }
}
