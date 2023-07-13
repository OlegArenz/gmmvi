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
    As we want to use Stein's Lemma for estimating the natural gradient (Codeletter "S"),
    we need to implement the gradient ourselves, and, therefore, we set
    use_log_density_and_grad=True and implement the corresponding method.
    """
    def __init__(self):
        super(Rosenbrock, self).__init__(use_log_density_and_grad=True,
                                         safe_for_tf_graph=False)
        self.a = 1
        self.b = 100

    def get_num_dimensions(self) -> int:
        return 2

    def log_density(self, samples: tf.Tensor) -> tf.Tensor:
        x = samples[:, 0].numpy().astype(np.float32)
        y = samples[:, 1].numpy().astype(np.float32)
        my_log_density = -((self.a - x)**2 + self.b * (y - x**2)**2)
        return tf.convert_to_tensor(my_log_density, dtype=tf.float32)

    def log_density_and_grad(self, samples: tf.Tensor) -> tf.Tensor:
        x = samples[:, 0].numpy().astype(np.float32)
        y = samples[:, 1].numpy().astype(np.float32)
        my_log_density = -((self.a - x)**2 + self.b * (y - x**2)**2)
        my_grad_x = -(-2 * (self.a - x) - 4 * self.b * (y - x**2) * x)
        my_grad_y = -(2 * self.b * (y - x**2))
        my_grad = np.vstack((my_grad_x, my_grad_y)).T
        return [tf.convert_to_tensor(my_log_density, dtype=tf.float32),
               tf.convert_to_tensor(my_grad, dtype=tf.float32)]

# We can also use the GmmviRunner, when using custom environments, but we have
# to put the LNPDF object into the dict. Furthermore, we need to define the other
# environment-specific settings that would otherwise be defined in
# the corresponding config in gmmvi/config/experiment_configs:
environment_config = {
    "target_fn": Rosenbrock(),
    "start_seed": 0,
    "environment_name": "Rosenbrock",
    "model_initialization": {
        "use_diagonal_covs": False,
        "num_initial_components": 1,
        "prior_mean": 0.,
        "prior_scale": 1.,
        "initial_cov": 1.,
    },
    "gmmvi_runner_config": {
        "log_metrics_interval": 100
    },
    "use_sample_database": True,
    "max_database_size": int(1e6),
    "temperature": 1.
}

# We will again use the automatically generated config for the algorithm,
# but this time, we will use "SAMTRUX". The default settings are reasonable for
# SAMTRUX, so we do not make any modifications to the hyperparameters.
algorithm_config = get_default_algorithm_config("SAMTRUX")

# Now we just need to merge the configs and use GmmviRunner as before:
merged_config = update_config(algorithm_config, environment_config)
gmmvi_runner = GmmviRunner.build_from_config(merged_config)

for n in range(500):
    gmmvi_runner.iterate_and_log(n)

# Plot samples from our "Rosenbrock-distribution"
test_samples = gmmvi_runner.gmmvi.model.sample(10000)[0]
plt.plot(test_samples[:, 0], test_samples[:, 1], 'x')
plt.show()
plt.pause(0.1)
