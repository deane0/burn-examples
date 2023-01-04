use std::sync::Arc;

use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    optim::{decay::WeightDecayConfig, Adam, AdamConfig},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::{
    data::MNISTBatcher,
    mlp::MlpConfig,
    model::{MnistConfig, Model},
};

static ARTIFACT_DIR: &str = "/tmp/burn-mnist";

pub fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer =
        AdamConfig::new(1e-4).with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config_mlp = MlpConfig::new();
    let config = MnistConfig::new(config_optimizer, config_mlp);
    B::seed(config.seed);

    // Data
    let batcher_train = Arc::new(MNISTBatcher::<B>::new(device));
    let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend>::new(device));
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::test()));

    // Model
    let optim = Adam::new(&config.optimizer);
    let model = Model::new(&config, 784, 10);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer::<f32>(2)
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim);

    let _model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();
}
