use burn_autodiff::ADBackendDecorator;
use burn_tch::{TchBackend, TchDevice};

use burn_examples::training;

fn main() {
    let device = TchDevice::Cpu;
    training::run::<ADBackendDecorator<TchBackend<f32>>>(device);
}
