import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from gmmvi.experiments.target_distributions.logistic_regression import LNPDF
from gmmvi.models.gmm_wrapper import GmmWrapper


class GMM_LNPDF(LNPDF):
    """Implements a target distribution that is given by a Gaussian mixture model.

    Parameters
    ----------
    target_weights: tf.Tensor of tf.float32
        a one-dimensional vector of size number_of_components containing the mixture weights.

    target_means: tf.Tensor of tf.float32
        a two-dimensional vector of size number_of_components x dimensions containing the mixture means.

    target_covs: tf.Tensor of tf.float32
        a three-dimensional vector of size number_of_components x dimensions x dimensions containing the covariance
        matrices.
    """

    def __init__(self, target_weights: tf.Tensor, target_means: tf.Tensor, target_covs: tf.Tensor):
        super(GMM_LNPDF, self).__init__(use_log_density_and_grad=False, safe_for_tf_graph=True)
        self.target_weights = tf.cast(target_weights, dtype=tf.float32)
        self.target_means = tf.cast(target_means, dtype=tf.float32)
        self.target_covs = tf.cast(target_covs, dtype=tf.float32)
        self.gmm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(logits=tf.math.log(self.target_weights)),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=self.target_means, scale_tril=tf.linalg.cholesky(self.target_covs)))

    def log_density(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return self.gmm.log_prob(x)

    def marginal_log_density(self, x, dim):
        """ Computes the marginal distribution along the given dimensions.

        Parameters
        ----------
        x: tf.Tensor of tf.float32
            a one-dimensional vector of size number_of_samples containing the samples we want to evaluate

        dim: an int
            Specifies the dimension used for constructing the marginal GMM.

        Returns:
            tf.Tensor - a one-dimensional Tensor of shape number_of_samples containing the marginal log densities.
        """
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(logits=tf.math.log(self.target_weights)),
            components_distribution=tfp.distributions.Normal(loc=self.target_means[:, dim],
                                                             scale=tf.math.sqrt(self.target_covs[:, dim, dim])))
        x = tf.cast(x, dtype=tf.float32)
        return mixture.log_prob(x)

    def get_num_dimensions(self):
        return len(self.target_means[0])

    def can_sample(self):
        """
        Returns:
            bool: We can sample from a GMM, so this method will return True.
        """
        return True

    def sample(self, n):
        """ Draws n samples from this GMM.

        Parameters:
            n: int
                The number of samples we want to draw.

        Returns:
            tf.Tensor: The sample, a tensor of size n x dimensions.
        """
        return self.gmm.sample(n)

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ This method computes the number of detected modes (by testing how many modes of this target distribution
        are close to a component in the learned model) and a figure that shows plots comparing the marginal
        distributions of the model with the true marginals of this target distribution.

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing two items (the number of detected modes, and a figure showing the plots of marginals.
        """
        expensive_metrics = dict()
        x_vals = tf.linspace(-70, 70, 1000)
        x_vals = tf.reshape(tf.repeat(x_vals, [self.get_num_dimensions()]), [-1, self.get_num_dimensions()])
        x_vals = tf.cast(x_vals, dtype=tf.float32)
        fig, axs = plt.subplots(4, 5)
        for dim in range(20):
            true_densities = tf.exp(self.marginal_log_density(x_vals, dim))
            start = tf.reduce_min(tf.where(true_densities > 1e-4))
            end = tf.reduce_max(tf.where(true_densities > 1e-4))
            axs[dim // 5, dim % 5].plot(x_vals[start:end, dim], true_densities[start:end], color='b')
            axs[dim // 5, dim % 5].plot(x_vals[start:end, dim], tf.exp(model.marginal_log_density(x_vals[start:end],
                                                                                                  dim)), color='r')
        dists_to_target_means = tf.reduce_min(
            tf.linalg.norm(tf.expand_dims(self.target_means, 1)
                           - tf.expand_dims(model.means, 0),
                           axis=2), 1)
        detection_threshold = tf.linalg.norm(6. * tf.ones(model.num_dimensions))
        num_detected = tf.where(dists_to_target_means < detection_threshold).shape[0]
        print(f"Found {num_detected} components.")
        expensive_metrics.update({"num_detected_modes": num_detected, "marginals": fig})
        return expensive_metrics


def make_target(num_dimensions):
    """
    Create a :py:class:`GMM target distribution<gmmvi.experiments.target_distributions.gmm.GMM_LNPDF>` using the
    same procedure as :cite:t:`Arenz2020` for initializing the weights, means and covariance matrices.

    Parameters:
        num_dimensions: int
            The number of dimensions of the target GMM

    Returns:
        :py:class:`GMM_LNPDF<gmmvi.experiments.target_distributions.gmm.GMM_LNPDF>`: the instantiated object
    """
    num_true_components = 10
    weights = np.ones(num_true_components) / num_true_components
    means = np.empty((num_true_components, num_dimensions))
    covs = np.empty((num_true_components, num_dimensions, num_dimensions))
    for i in range(0, num_true_components):
        means[i] = 100 * (np.random.random(num_dimensions) - 0.5)
        covs[i] = 0.1 * np.random.normal(0, num_dimensions, (num_dimensions * num_dimensions)).reshape(
            (num_dimensions, num_dimensions))
        covs[i] = covs[i].transpose().dot(covs[i])
        covs[i] += 1 * np.eye(num_dimensions)
    return GMM_LNPDF(weights, means, covs)


def make_target_with_scale(num_dimensions, num_components, scale):
    num_true_components = num_components
    weights = np.ones(num_true_components) / num_true_components
    means = np.empty((num_true_components, num_dimensions))
    covs = np.empty((num_true_components, num_dimensions, num_dimensions))
    for i in range(0, num_true_components):
        means[i] = 100 * (np.random.random(num_dimensions) - 0.5)
        covs[i] = np.random.normal(0, np.sqrt(scale), (num_dimensions * num_dimensions)).reshape(
            (num_dimensions, num_dimensions))
        covs[i] = covs[i].transpose().dot(covs[i])
        covs[i] += 1 * np.eye(num_dimensions)

    # print("target_means:", means)
    # print("target_covs:", covs)
    return GMM_LNPDF(weights, means, covs)


def U(theta):
    return np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )


def make_simple_target():
    pi = math.pi

    # weights
    w_true = np.array([0.5, 0.3, 0.2])

    # means
    mu_true = np.array(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [0.0, 2.0],
        ]
    )

    # covs
    cov1 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov1 = U(pi / 4) @ cov1 @ np.transpose(U(pi / 4))
    cov2 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov2 = U(-pi / 4) @ cov2 @ np.transpose(U(-pi / 4))
    cov3 = np.array([[1.0, 0.0], [0.0, 2.0]])
    cov3 = U(pi / 2) @ cov3 @ np.transpose(U(pi / 2))
    cov_true = np.stack([cov1, cov2, cov3], axis=0)

    # generate target dist
    target_dist = GMM_LNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covs=cov_true,
    )

    return target_dist


def make_star_target(num_components):
    # Source: Lin et al.
    K = num_components

    ## weights
    w_true = np.ones((K,)) / K

    ## means and precs
    # first component
    mus = [np.array([1.5, 0.0])]
    precs = [np.diag([1.0, 100.0])]
    # other components are generated through rotation
    theta = 2 * math.pi / K
    for _ in range(K - 1):
        mus.append(U(theta) @ mus[-1])
        precs.append(U(theta) @ precs[-1] @ np.transpose(U(theta)))
    assert len(w_true) == len(mus) == len(precs) == K

    mu_true = np.stack(mus, axis=0)
    prec_true = np.stack(precs, axis=0)
    cov_true = np.linalg.inv(prec_true)

    # generate target dist
    target_dist = GMM_LNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covs=cov_true,
    )

    return target_dist
