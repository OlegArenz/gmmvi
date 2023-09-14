from gmmvi.gmmvi_runner import GmmviRunner
from gmmvi.configs import get_default_algorithm_config, update_config
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt

# For creating a custom environment, we need to extend
# gmmvi.experiments.target_distributions.lnpdf.LNPDF:
from gmmvi.experiments.target_distributions.lnpdf import LNPDF
class Rosenbrock(LNPDF):
    """ We treat the negative Rosenbrock function as unnormalized target distribution.
    We implement it in numpy and do not allow GMMVI to backpropagate through log_density().
    Here we use MORE to estimate the natural gradient, which does not require the target distribution to
    be differentiable.
    """
    def __init__(self):
        super(Rosenbrock, self).__init__(safe_for_tf_graph=False)
        self.a = 1
        self.b = 100

    def get_num_dimensions(self) -> int:
        return 2

    def log_density(self, samples: tf.Tensor) -> tf.Tensor:
        x = samples[:, 0].numpy().astype(np.float32)
        y = samples[:, 1].numpy().astype(np.float32)
        my_log_density = -((self.a - x)**2 + self.b * (y - x**2)**2)
        return tf.convert_to_tensor(my_log_density, dtype=tf.float32)

# We can also use the GmmviRunner, when using custom environments, but we have
# to put the LNPDF object into the dict. Furthermore, we need to define the other
# environment-specific settings that would otherwise be defined in
# the corresponding config in gmmvi/config/experiment_configs:
environment_config = {
    "target_fn": Rosenbrock(),
    "start_seed": 0,
    "environment_name": "Rosenbrock",
    "model_initialization": {
        "use_diagonal_covs": True,
        "num_initial_components": 1,
        "prior_mean": 0.,
        "prior_scale": 1.,
        "initial_cov": 1.,
    },
    "ng_estimator_config": {
        "only_use_own_samples": False,
    },
    "num_component_adapter_config": {
        "add_iters": 5,
        "del_iters": 100,
    },
    "sample_selector_config": {
        "desired_samples_per_component": 5,
    },
    "gmmvi_runner_config": {
        "log_metrics_interval": 100
    },
    "use_sample_database": True,
    "max_database_size": int(1e6),
    "temperature": 1.
}

# If the target distribution is not differentiable, we have to pick MORE (Design Choice Z) as NgEstimator.
algorithm_config = get_default_algorithm_config("ZAMTRUX")

# Now we just need to merge the configs and use GmmviRunner as before:
merged_config = update_config(algorithm_config, environment_config)
gmmvi_runner = GmmviRunner.build_from_config(merged_config)

for n in range(1000):
    gmmvi_runner.iterate_and_log(n)

# Plot samples from our "Rosenbrock-distribution"
test_samples = gmmvi_runner.gmmvi.model.sample(10000)[0]
plt.plot(test_samples[:, 0], test_samples[:, 1], 'x')
plt.show()
plt.pause(0.1)
