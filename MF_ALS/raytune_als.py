from implicit_als import train_als
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':
    config={
        "factors": tune.choice([50,55,60]),
        "iterations": tune.choice([30,35,40,45,50,55,60])
    }

    asha = ASHAScheduler(
        time_attr='training_iteration',
        max_t=200,
        grace_period=10,
        reduction_factor=3,
        brackets=3)

    ray.shutdown()
    ray.init(num_cpus=3, num_gpus=1)

    result = tune.run(
        tune.with_parameters(train_als),
        local_dir='/opt/ml/input/melon/results/',
        resources_per_trial={"cpu": 3, "gpu": 1},
        config=config,
        metric="recall",
        mode="max",
        num_samples=20,
        scheduler=asha
    )

    
