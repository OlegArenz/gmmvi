import tensorflow as tf
from gmmvi.models.gmm import GMM
from math import pi


class DiagonalGMM(GMM):
    """ A Gaussian mixture model with diagonal covariance matrices.

    Parameters:
        weights: tf.Tensor
            a one-dimensional tensor containing the initial weights of the GMM.

        means: tf.Tensor
            a two-dimensional tensor containing the component means.

        covs: tf.Tensor
            a two-dimensional tensor containing the diagonal entries of the component covariances.
    """

    def __init__(self, weights: tf.Tensor, means: tf.Tensor, covs: tf.Tensor):
        num_dimensions = len(means[0])
        log_weights = tf.Variable(tf.math.log(weights), shape=[None], dtype=tf.float32)
        chol_covs = tf.Variable(tf.stack([tf.math.sqrt(cov) for cov in covs]),
                                  shape=[None, num_dimensions],
                                  dtype=tf.float32)
        means = tf.Variable(tf.convert_to_tensor(means), shape=[None, num_dimensions], dtype=tf.float32)
        super(DiagonalGMM, self).__init__(log_weights, means, chol_covs)
        self.diagonal_covs = True     # ToDo: Check why this is necessary

    @staticmethod
    def diagonal_gaussian_log_pdf(dim: int, mean: tf.Tensor, chol: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        constant_part = - 0.5 * dim * tf.math.log(2 * pi) - tf.reduce_sum(tf.math.log(chol))
        return constant_part - 0.5 * tf.reduce_sum(
            tf.square(tf.expand_dims(1. / chol, 0) * (tf.expand_dims(mean, 0) - x)), axis=1)

    @property
    def covs(self) -> tf.Tensor:
        return tf.math.square(self.chol_cov)

    def gaussian_entropy(self, chol: tf.Tensor) -> tf.Tensor:
        return 0.5 * self.num_dimensions * (tf.math.log(2 * pi) + 1) + tf.reduce_sum(tf.math.log(chol))

    def sample_from_component(self, index: int, num_samples: int) -> tf.Tensor:
        return tf.transpose(tf.expand_dims(self.means[index], 1) + tf.expand_dims(self.chol_cov[index], 1)
                            * tf.random.normal((self.num_dimensions, num_samples), mean=0., stddev=1.))

    def component_log_densities(self, samples: tf.Tensor) -> tf.Tensor:
        log_pdfs = tf.TensorArray(tf.float32, size=self.num_components)
        for i in range(self.num_components):
            this_log_pdf = self.diagonal_gaussian_log_pdf(self.num_dimensions, self.means[i], self.chol_cov[i], samples)
            log_pdfs = log_pdfs.write(i, this_log_pdf)
        log_pdfs = log_pdfs.stack()
        return log_pdfs

    def add_component(self, initial_weight: tf.Tensor, initial_mean: tf.Tensor, initial_cov: tf.Tensor):
        self.means.assign(tf.concat((self.means, tf.expand_dims(initial_mean, axis=0)), axis=0))
        self.chol_cov.assign(
            tf.concat((self.chol_cov, tf.expand_dims(tf.math.sqrt(initial_cov), axis=0)), axis=0))
        self.replace_weights(tf.concat((self.log_weights, tf.expand_dims(tf.math.log(initial_weight), axis=0)), axis=0))
