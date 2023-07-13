import os
import logging
# Tensorflow may give warnings when the Cholesky decomposition fails.
# However, these warning can usually be ignored because the NgBasedOptimizer
# will handle them by rejecting the update and decreasing the stepsize for
# the failing component. To keep the console uncluttered, we suppress warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf

from gmmvi.optimization.gmmvi import GMMVI
from gmmvi.configs import load_yaml
from gmmvi.experiments.target_distributions.logistic_regression import make_breast_cancer
from gmmvi.models.full_cov_gmm import FullCovGMM
from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.experiments.setup_experiment import construct_initial_mixture

#For creating a GMMVI object using GMMVI.build_from_config, we need:
# 1. A dictionary containing the hyperparameters
my_path = os.path.dirname(os.path.realpath(__file__))
config = load_yaml(os.path.join(my_path, "example_config.yml"))

# 2. A target distribution
target_distribution = make_breast_cancer()

# 3. An (wrapped) initial model
dims = target_distribution.get_num_dimensions()
initial_weights = tf.ones(1, tf.float32)
initial_means = tf.zeros((1, dims), tf.float32)
initial_covs = tf.reshape(100 * tf.eye(dims), [1, dims, dims])
model = FullCovGMM(initial_weights, initial_means, initial_covs)
# Above config contains a section model_initialization, and, therefore,
# we could also create the initial model using:
# model = construct_initial_mixture(dims, **config["model_initialization"])
wrapped_model = GmmWrapper.build_from_config(model=model, config=config)


# Now we can create the GMMVI object and start optimizing
gmmvi = GMMVI.build_from_config(config=config,
                                target_distribution=target_distribution,
                                model=wrapped_model)
max_iter = 1001
for n in range(max_iter):
    gmmvi.train_iter()

    if n % 100  == 0:
        samples = gmmvi.model.sample(1000)[0]
        elbo = tf.reduce_mean(target_distribution.log_density(samples)
                              - model.log_density(samples))
        print(f"{n}/{max_iter}: "
              f"The model now has {gmmvi.model.num_components} components "
              f"and an elbo of {elbo}.")
