import numpy as np
import tensorflow as tf
from gmmvi.models.diagonal_gmm import DiagonalGMM

from gmmvi.experiments.target_distributions.logistic_regression import LNPDF



class DIAGGMM_LNPDF(LNPDF):
    def __init__(self, target_weights, target_means, target_covs):
        super(DIAGGMM_LNPDF, self).__init__(use_log_density_and_grad=False)
        self.target_weights = target_weights
        self.target_means = target_means
        self.target_covs = target_covs
        self.target_means = target_means.astype(np.float32)
        self.target_covs = target_covs.astype(np.float32)
        self.gmm = DiagonalGMM(target_weights, target_means, target_covs)

    def log_density(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return self.gmm.log_density(x)

    def get_num_dimensions(self):
        return len(self.target_means[0])

    def can_sample(self):
        return True

    def sample(self, n):
        return self.gmm.sample(n)


def make_target(num_dimensions):
    num_true_components = 10
    weights = np.ones(num_true_components) / num_true_components
    means = np.empty((num_true_components, num_dimensions))
    covs = np.empty((num_true_components, num_dimensions))
    for i in range(0, num_true_components):
        means[i] = 100 * (np.random.random(num_dimensions) - 0.5)
        covs[i] = 10 * np.random.random(num_dimensions)
    return DIAGGMM_LNPDF(tf.cast(weights, tf.float32).numpy(),
                         tf.cast(means, tf.float32).numpy(),
                         tf.cast(covs, tf.float32).numpy())

