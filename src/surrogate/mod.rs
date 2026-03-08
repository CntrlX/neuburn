//! Surrogate gradients for backpropagation through spike (Heaviside) nonlinearity.

mod atan;
mod fast_sigmoid;
mod sigmoid;
mod straight_through;
mod trait_impl;

pub use atan::ATan;
pub use fast_sigmoid::FastSigmoid;
pub use sigmoid::SigmoidSurrogate;
pub use straight_through::StraightThrough;
pub use trait_impl::Surrogate;
