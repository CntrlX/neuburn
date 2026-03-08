//! Spike encoding utilities: rate, latency, delta.

mod delta;
mod latency;
mod rate;

pub use delta::delta_encode;
pub use latency::latency_encode;
pub use rate::rate_encode;
