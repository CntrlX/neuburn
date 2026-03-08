//! MNIST SNN example: two-layer SNN with RLeaky neurons and FastSigmoid surrogate.
//! Uses a simple manual training loop (no full Learner).

use burn::backend::Autodiff;
use burn::data::dataset::Dataset;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Int, Tensor};
use burn_ndarray::{NdArray, NdArrayDevice};
use neuburn::encoding::rate_encode;
use neuburn::neuron::RLeaky;
use neuburn::state::{NeuronState, ResetMode};
use neuburn::surrogate::FastSigmoid;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

const NUM_STEPS: usize = 16;
const HIDDEN: usize = 128;
const EPOCHS: usize = 2;
const BATCH_SIZE: usize = 32;
const LR: f64 = 1e-3;

#[derive(burn::module::Module, Debug)]
pub struct TwoLayerSnn<B: burn::tensor::backend::Backend> {
    layer1: RLeaky<B>,
    layer2: RLeaky<B>,
}

impl<B: burn::tensor::backend::Backend> TwoLayerSnn<B> {
    pub fn new(device: &B::Device) -> Self {
        let layer1 = RLeaky::new(device, 784, HIDDEN, 0.9, 1.0, ResetMode::Subtract);
        let layer2 = RLeaky::new(device, HIDDEN, 10, 0.9, 1.0, ResetMode::Subtract);
        Self { layer1, layer2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>, surrogate: &FastSigmoid) -> Tensor<B, 2> {
        let device = x.device();
        let dims = x.dims();
        let batch = dims[0];

        let mut state1 = NeuronState::zeros_mem_only(&device, batch, HIDDEN);
        let mut state2 = NeuronState::zeros_mem_only(&device, batch, 10);
        let mut spike1_prev = Tensor::zeros([batch, HIDDEN], &device);
        let mut spike2_prev = Tensor::zeros([batch, 10], &device);
        let mut logits_accum = Tensor::zeros([batch, 10], &device);

        for t in 0..NUM_STEPS {
            let input_t = x.clone().slice([0..batch, t..t + 1, 0..784]).squeeze_dims(&[1]);
            let (s1, new_state1) = self
                .layer1
                .step(input_t, spike1_prev.clone(), &state1, surrogate);
            state1 = new_state1;
            spike1_prev = s1.clone();

            let (s2, new_state2) = self
                .layer2
                .step(s1, spike2_prev.clone(), &state2, surrogate);
            state2 = new_state2;
            spike2_prev = s2.clone();
            logits_accum = logits_accum + state2.mem.clone();
        }

        logits_accum.div_scalar(NUM_STEPS as f32)
    }
}

fn main() {
    let device = NdArrayDevice::Cpu;
    println!("Loading MNIST...");
    let train = burn::data::dataset::vision::MnistDataset::train();
    let n = train.len();
    println!("Train size: {}", n);

    let mut model = TwoLayerSnn::<MyAutodiffBackend>::new(&device);
    let mut optim = AdamConfig::new().init();
    let surrogate = FastSigmoid::default();

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        let mut batches = 0usize;
        for start in (0..n).step_by(BATCH_SIZE).take(100) {
            let end = (start + BATCH_SIZE).min(n);
            let batch_size = end - start;
            let mut images = Vec::with_capacity(batch_size * 28 * 28);
            let mut targets = Vec::with_capacity(batch_size);
            for i in start..end {
                let item = train.get(i).unwrap();
                for row in item.image.iter() {
                    for &p in row.iter() {
                        images.push(p / 255.0);
                    }
                }
                targets.push(item.label as i64);
            }
            let images =
                Tensor::<MyAutodiffBackend, 1>::from_floats(images.as_slice(), &device)
                    .reshape([batch_size, 28 * 28]);
            let spike_train = rate_encode(images, NUM_STEPS, &device);
            let targets_t =
                Tensor::<MyAutodiffBackend, 1, Int>::from_ints(targets.as_slice(), &device);

            let logits = model.forward(spike_train, &surrogate);
            let one_hot = one_hot_batch::<MyAutodiffBackend>(&targets_t, 10, &device);
            let loss = cross_entropy_with_logits(logits, one_hot).mean();
            total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LR, model, grads);
        }
        println!("Epoch {} loss: {}", epoch + 1, total_loss / batches as f32);
    }
    println!("Done.");
}

fn one_hot_batch<B: burn::tensor::backend::Backend>(
    indices: &Tensor<B, 1, Int>,
    num_classes: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = indices.dims()[0];
    let indices_data = indices.clone().into_data();
    let idx_slice = indices_data.as_slice::<i64>().unwrap();
    let mut floats = Vec::with_capacity(batch * num_classes);
    for b in 0..batch {
        let idx_u = idx_slice[b] as usize;
        for c in 0..num_classes {
            floats.push(if c == idx_u { 1.0f32 } else { 0.0f32 });
        }
    }
    Tensor::<B, 1>::from_floats(floats.as_slice(), device).reshape([batch, num_classes])
}
