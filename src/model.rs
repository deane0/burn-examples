use burn::{
    config::Config,
    module::{Module, Param},
    nn::{self, loss::CrossEntropyLoss},
    optim::AdamConfig,
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    data::MNISTBatch,
    mlp::{Mlp, MlpConfig},
};

#[derive(Config)]
pub struct MnistConfig {
    #[config(default = 6)]
    pub num_epochs: usize,
    #[config(default = 12)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub optimizer: AdamConfig,
    pub mlp: MlpConfig,
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
    num_classes: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.d_model));
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.d_model, num_classes));

        Self {
            mlp: Param::new(mlp),
            input: Param::new(input),
            output: Param::new(output),
            num_classes,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.input.forward(x);
        x = self.mlp.forward(x);
        x = self.output.forward(x);

        x
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(self.num_classes, None);
        let loss = loss.forward(&output, &targets);

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<B, MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<B, ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
