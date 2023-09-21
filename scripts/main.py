import numpy as np
from torchvision import datasets

import ray
from ray import air, tune

from raytest.my_trainable import Trainable


def main(config):
    # Download the dataset first
    datasets.MNIST("~/data", train=True, download=True)
    # Setup Tuner
    tune_config = tune.TuneConfig(
        metric='mean_accuracy', mode='max', num_samples=4
    )
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=40
    )
    run_config = air.RunConfig(
        stop={'training_iteration': 400},
        checkpoint_config=checkpoint_config
    )
    # Run Tuner
    tuner = tune.Tuner(
        Trainable,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config
    )
    return tuner.fit()


if __name__ == '__main__':
    ray.init(address='auto')
    search_space = {
        "lr": tune.uniform(0.001, 0.1),
        "momentum": tune.uniform(0.001, 0.9),
    }
    results = main(search_space)
    print("Best config is:", results.get_best_result().config)
    # print(results.get_best_result().metrics)
    # dfs = {result.log_dir: result.metrics_dataframe for result in results}
    # print([d.mean_accuracy for d in dfs.values()])
