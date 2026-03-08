# neuBurn

Spiking neural network (SNN) framework built on [Burn](https://github.com/burn-rs/burn) — a Rust-native alternative to snntorch.

## Features

- **Neuron models**: Leaky (LIF), Synaptic, Alpha, RLeaky, RSynaptic, Lapicque
- **Reset modes**: Subtract, Zero, None (snntorch-style)
- **Learnable** membrane decay (β) and optional learnable threshold
- **Surrogate gradients**: FastSigmoid, Sigmoid, StraightThrough, ATan
- **Encoding**: rate (Poisson), latency, delta
- **Losses**: MSE count, CE rate, CE temporal
- **Layers**: SpikingConv2d, LeakyParallel, BNTT (BatchNorm through time), SLSTM, KAN-based event-driven blocks

## Usage

```toml
[dependencies]
neuburn = "0.1"
burn = { version = "0.14", features = ["train"] }
```

```rust
use neuburn::prelude::*;
```

## License

MIT OR Apache-2.0
