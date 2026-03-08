//! First-spike latency encoding: intensity -> spike time (earlier = stronger).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Latency encoding: data in [0,1] -> one spike per neuron at time proportional to (1 - value).
/// Returns tensor [batch, num_steps, features]: 1 at latency step, 0 elsewhere.
/// Step index = round((1 - data) * (num_steps - 1)), clamped to [0, num_steps-1].
pub fn latency_encode<B: Backend>(
    data: Tensor<B, 2>,
    num_steps: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let shape = data.dims();
    let _batch = shape[0];
    let _features = shape[1];
    let one = Tensor::ones(shape, device);
    let max_step = (num_steps - 1) as f32;
    let step_float = (one - data).mul_scalar(max_step);
    let mut spikes = Vec::with_capacity(num_steps);
    for t in 0..num_steps {
        let t_val = t as f32;
        let t_tensor = Tensor::full(shape, t_val, device);
        let t_plus = Tensor::full(shape, t_val + 1.0, device);
        let ge = step_float.clone().greater_equal(t_tensor).float();
        let le = step_float.clone().lower_equal(t_plus).float();
        let mask = ge.mul(le);
        spikes.push(mask);
    }
    Tensor::stack(spikes, 1)
}
