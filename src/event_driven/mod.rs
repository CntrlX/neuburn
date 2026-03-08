//! Event-driven SKAN: delta-gated KAN + LIF layers.

mod full_skan;
mod kan_synapse;
mod spiking_kan_layer;

pub use full_skan::EventDrivenFullSKAN;
pub use kan_synapse::EventDrivenKANSynapse;
pub use spiking_kan_layer::EventDrivenSpikingKANLayer;
