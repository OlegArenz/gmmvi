import tensorflow as tf

from gmmvi.models.diagonal_gmm import DiagonalGMM
from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.models.full_cov_gmm import FullCovGMM
import numpy as np

from gmmvi.experiments.target_distributions.lnpdf import LNPDF

def init_experiment(config: dict) -> [LNPDF, GmmWrapper]:
    """
    Creates the target distribution and initial model based on the provided config.

    Parameters:
        config: dict
            The config dictionary

    Returns:
        tuple(:py:class:`LNPDF:, :py:class:`GmmWrapper`)

    """
    if "environment_config" in config.keys():
        target_fn = get_target_lnpdf(experiment=config["environment_name"],
                                     environment_config=config["environment_config"],
                                     seed=config["seed"])
    elif "target_fn" in config.keys():
        target_fn = config.pop("target_fn")
    else:
        ValueError("No target distribution was specified")

    gmm = construct_initial_mixture(
        num_dimensions=target_fn.get_num_dimensions(),
        **config["model_initialization"]
    )

    if "initial_l2_regularizer" in config["ng_estimator_config"]:
        initial_l2_regularizer = config["ng_estimator_config"]['initial_l2_regularizer']
    else:
        initial_l2_regularizer = 1e-12
    gmm_wrapper = GmmWrapper(gmm, config["component_stepsize_adapter_config"]["initial_stepsize"],
                             initial_l2_regularizer, max_reward_history_length=10000)

    return target_fn, gmm_wrapper


def get_target_lnpdf(experiment, environment_config, seed):
    if experiment == "breastCancer":
        from gmmvi.experiments.target_distributions.logistic_regression import make_breast_cancer
        target_fn = make_breast_cancer()
    elif experiment == "breastCancer_mb":
        from gmmvi.experiments.target_distributions.logistic_regression import make_breast_cancer_mb
        target_fn = make_breast_cancer_mb(**environment_config)
    elif experiment == "GermanCredit":
        from gmmvi.experiments.target_distributions.logistic_regression import make_german_credit
        target_fn = make_german_credit()
    elif experiment == "GermanCredit_mb":
        from gmmvi.experiments.target_distributions.logistic_regression import make_german_credit_mb
        target_fn = make_german_credit_mb(**environment_config)
    elif experiment == "PlanarRobot4":
        from gmmvi.experiments.target_distributions.planar_robot import make_four_goal
        target_fn = make_four_goal()
    elif experiment == "PlanarRobot1":
        from gmmvi.experiments.target_distributions.planar_robot import make_single_goal
        target_fn = make_single_goal()
    elif experiment == "STM":
        from gmmvi.experiments.target_distributions.student_t_mixture import make_target
        target_fn = make_target(**environment_config)
    elif experiment.startswith("GMM"):
        from gmmvi.experiments.target_distributions.gmm import make_target
        target_fn = make_target(**environment_config)
    elif experiment.startswith("DIAGGMM"):
        from gmmvi.experiments.target_distributions.diag_gmm import make_target
        target_fn = make_target(**environment_config)
    elif experiment.startswith("MNIST"):
        from gmmvi.experiments.target_distributions.bnn import make_MNIST_target
        target_fn = make_MNIST_target(**environment_config)
    elif experiment.startswith("WINE"):
        from gmmvi.experiments.target_distributions.bnn import make_WINE_target
        target_fn = make_WINE_target(dataset_seed=seed, **environment_config)
    elif experiment.startswith("Talos"):
        from gmmvi.experiments.target_distributions.talos_ik import make_talos_target
        target_fn = make_talos_target(**environment_config)
    else:
        raise ValueError(f"get_target_lnpdf() was called with unknown experiment name: {experiment}")

    return target_fn

def construct_initial_mixture(num_dimensions, num_initial_components, prior_mean, prior_scale, use_diagonal_covs,
                              initial_cov=None):
    """
    Initializes a GMM with the given number of dimensions and number of components.
    The component means are sampled from a diagonal prior distribution with mean *prior_mean* and
    standard deviation *prior_scale*. The initial covariance is given by *initial_cov*.

    Parameters:
        num_dimensions: int
            number of dimensions of the model

        num_initial_components: int
            initial number of components

        prior_mean: int or tf.Tensor
            mean of the Gaussian for sampling the mixture means.

        prior_scale: int or tf.Tensor
            standard deviation for sampling the mixture means. Can be a scalar, or a tensor of size *num_dimensions*

        use_diagonal_covs: bool
            if True, the initial mixture will be a :py:class:`DiagonalGMM`,
            otherwise it will be a :py:class:`FullCovGMM`.

        initial_cov: tf.Tensor
            the initial covariance matrix.

    Returns:
        :py:class:`GMM`: The initial model.

    """
    if np.isscalar(prior_mean):
        prior_mean = prior_mean * np.ones(num_dimensions)

    if np.isscalar(prior_scale):
        prior_scale = prior_scale * np.ones(num_dimensions)
    prior = np.array(prior_scale) ** 2

    weights = np.ones(num_initial_components, dtype=np.float32) / num_initial_components
    means = np.zeros((num_initial_components, num_dimensions), dtype=np.float32)

    if use_diagonal_covs:
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * np.ones(num_dimensions)

        covs = np.ones((num_initial_components, num_dimensions), dtype=np.float32)
        for i in range(0, num_initial_components):
            if num_initial_components == 1:
                means[i] = prior_mean
            else:
                means[i] = prior_mean + np.sqrt(prior) * np.random.standard_normal([num_dimensions])
            covs[i] = initial_cov
    else:
        prior = np.diag(prior)
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * np.eye(num_dimensions)

        covs = np.ones((num_initial_components, num_dimensions, num_dimensions), dtype=np.float32)
        for i in range(0, num_initial_components):
            if num_initial_components == 1:
                means[i] = prior_mean
            else:
                means[i] = prior_mean + np.linalg.cholesky(prior) @ np.random.standard_normal([num_dimensions,1])[:,0]
            covs[i] = initial_cov

    if use_diagonal_covs:
        return DiagonalGMM(weights, tf.convert_to_tensor(means, tf.float32), tf.convert_to_tensor(covs, tf.float32))
    else:
        return FullCovGMM(weights, tf.convert_to_tensor(means, tf.float32), tf.convert_to_tensor(covs, tf.float32))
