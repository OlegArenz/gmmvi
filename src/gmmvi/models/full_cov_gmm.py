import tensorflow as tf
from gmmvi.models.gmm import GMM
from math import pi


class FullCovGMM(GMM):
    """ A Gaussian mixture model with full covariance matrices.

    Parameters:
        weights: tf.Tensor
            a one-dimensional tensor containing the initial weights of the GMM.

        means: tf.Tensor
            a two-dimensional tensor containing the component means.

        covs: tf.Tensor
            a three-dimensional tensor containing the component covariance matrices.
    """
    def __init__(self, weights: tf.Tensor, means: tf.Tensor, covs: tf.Tensor):
        self.diagonal_covs = False
        num_dimensions = len(means[0])
        log_weights = tf.Variable(tf.math.log(weights), shape=[None], dtype=tf.float32)
        chol_covs = tf.Variable(tf.stack([tf.linalg.cholesky(cov) for cov in covs]),
                                      shape=[None, num_dimensions, num_dimensions],
                                      dtype=tf.float32)
        means = tf.Variable(tf.convert_to_tensor(means), shape=[None, num_dimensions], dtype=tf.float32)
        super(FullCovGMM, self).__init__(log_weights, means, chol_covs)

    @property
    def covs(self) -> tf.Tensor:
        return self.chol_cov @ tf.transpose(self.chol_cov, [0, 2, 1])

    def gaussian_entropy(self, chol: tf.Tensor) -> tf.Tensor:
        return 0.5 * self.num_dimensions * (tf.math.log(2 * pi) + 1) + tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(chol)))

    def sample_from_component(self, index: int, num_samples: int) -> tf.Tensor:
        return tf.transpose(tf.expand_dims(self.means[index], axis=-1)
                            + self.chol_cov[index] @ tf.random.normal((self.num_dimensions, num_samples),
                                                                      mean=0., stddev=1.))

    def component_log_density(self, index: int, samples: tf.Tensor) -> tf.Tensor:
        diffs = samples - self.means[index]
        sqrts = tf.linalg.triangular_solve(self.chol_cov[index], tf.transpose(diffs))
        mahalas = - 0.5 * tf.reduce_sum(sqrts * sqrts, axis=0)
        const_parts = - 0.5 * tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(self.chol_cov[index])))) \
                      - 0.5 * self.num_dimensions * tf.math.log(2 * pi)
        return mahalas + const_parts

    def component_marginal_log_densities(self, samples: tf.Tensor, dim: int) -> tf.Tensor:
        diffs = tf.expand_dims(samples[:, dim], 0) - tf.expand_dims(self.means[:, dim], 1)
        mahalas = - 0.5 * diffs * diffs / tf.expand_dims(self.covs[:, dim, dim], 1)
        const_parts = - 0.5 * tf.math.log(self.covs[:, dim, dim]) \
                      - 0.5 * tf.math.log(2 * pi)
        return mahalas + tf.expand_dims(const_parts, axis=1)

    def component_log_densities(self, samples: tf.Tensor) -> tf.Tensor:
        diffs = tf.expand_dims(samples, 0) - tf.expand_dims(self.means, 1)
        sqrts = tf.linalg.triangular_solve(self.chol_cov, tf.transpose(diffs, [0, 2, 1]))
        mahalas = - 0.5 * tf.reduce_sum(sqrts * sqrts, axis=1)
        const_parts = - 0.5 * tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(self.chol_cov))), axis=1) \
                      - 0.5 * self.num_dimensions * tf.math.log(2 * pi)
        return mahalas + tf.expand_dims(const_parts, axis=1)

    def add_component(self, initial_weight: tf.Tensor, initial_mean: tf.Tensor, initial_cov: tf.Tensor):
        self.means.assign(tf.concat((self.means, tf.expand_dims(initial_mean, axis=0)), axis=0))
        self.chol_cov.assign(
            tf.concat((self.chol_cov, tf.expand_dims(tf.linalg.cholesky(initial_cov), axis=0)), axis=0))
        self.replace_weights(tf.concat((self.log_weights, tf.expand_dims(tf.math.log(initial_weight), axis=0)), axis=0))