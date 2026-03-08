//! Prelude: most commonly used types.

pub use crate::bntt::{BatchNormTT1d, BatchNormTT2d};
pub use crate::encoding::{delta_encode, latency_encode, rate_encode};
pub use crate::functional::{ce_rate_loss, ce_temporal_loss, lateral_inhibition, mse_count_loss};
pub use crate::kan::KANLinear;
pub use crate::layers::SpikingConv2d;
pub use crate::neuron::{Leaky, LeakyConfig, LeakyParallel, RLeaky, RSynaptic};
pub use crate::slstm::SLSTM;
pub use crate::state::{LayerState, NeuronState, ResetMode};
pub use crate::surrogate::FastSigmoid;
