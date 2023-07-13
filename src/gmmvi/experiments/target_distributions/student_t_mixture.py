import os.path
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gmmvi.experiments.target_distributions.logistic_regression import LNPDF
from gmmvi.models.gmm_wrapper import GmmWrapper

tfd = tfp.distributions


class StudentTMixture_LNPDF(LNPDF):
    """Implements a target distribution that is given by a mixture of Student-T distributions.

    Parameters
    ----------
    target_weights: tf.Tensor of tf.float32
        a one-dimensional vector of size number_of_components containing the mixture weights.

    target_means: tf.Tensor of tf.float32
        a two-dimensional vector of size number_of_components x dimensions containing the mixture means.

    target_covs: tf.Tensor of tf.float32
        a three-dimensional vector of size number_of_components x dimensions x dimensions containing the covariance
        matrices.

    alpha: int
        The number of degrees of freedom.
    """

    def __init__(self, target_weights: tf.Tensor, target_means: tf.Tensor, target_covs: tf.Tensor, alpha=2):
        super(StudentTMixture_LNPDF, self).__init__(use_log_density_and_grad=False)
        self.alpha = alpha
        self.target_weights = tf.cast(target_weights, dtype=tf.float32)
        self.target_means = tf.cast(target_means, dtype=tf.float32)
        self.target_covs = tf.cast(target_covs, dtype=tf.float32)
        self.mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=tf.math.log(self.target_weights)),
            components_distribution=tfd.MultivariateStudentTLinearOperator(
                df=alpha, loc=self.target_means,
                scale=tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(self.target_covs))))

    def marginal_log_density(self, x, dim):
        """ Computes the marginal distribution along the given dimensions.

        Parameters:
            x: tf.Tensor of tf.float32
                a one-dimensional vector of size number_of_samples containing the samples we want to evaluate

            dim: an int
                Specifies the dimension used for constructing the marginal mixture of Student-Ts.

        Returns:
            tf.Tensor: a one-dimensional tensor of shape number_of_samples containing the marginal log densities.
        """
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(logits=tf.math.log(self.target_weights)),
            components_distribution=tfp.distributions.StudentT(df=self.alpha, loc=self.target_means[:, dim],
                                                               scale=tf.math.sqrt(self.target_covs[:, dim, dim])))
        x = tf.cast(x, dtype=tf.float32)
        return mixture.log_prob(x[:, dim])

    def log_density(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return self.mixture.log_prob(x)

    def get_num_dimensions(self):
        return len(self.target_means[0])

    def can_sample(self):
        """
        Returns:
            bool: We can sample from a mixture of Student-T, so this method will return True.
        """
        return True

    def sample(self, n):
        """ Draws n samples from this mixture of Student-T.

        Parameters:
            n: int
                The number of samples we want to draw.

        Returns:
            tf.Tensor: The sample, a tensor of size n x dimensions.
        """
        return self.mixture.sample(n)

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ This method computed the number of detected modes (by testing how many modes of this mixture of Student-T
        are close to a component in the learned model) and a figure that shows plots comparing the marginal
        distributions of the model with the true marginals of this mixture of Student-T.

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing two items (the number of detected modes, and a figure showing the plots of
            marginals.
        """
        expensive_metrics = dict()
        x_vals = tf.linspace(-25, 25, 1000)
        x_vals = tf.reshape(tf.repeat(x_vals, [self.get_num_dimensions()]), [-1, self.get_num_dimensions()])
        x_vals = tf.cast(x_vals, dtype=tf.float32)
        fig, axs = plt.subplots(5, 4, sharex=True)
        fig.tight_layout(pad=0.2)
        plt.subplots_adjust(wspace=0.2, hspace=0.1, right=0.995, left=0.04, top=0.995)
        for dim in range(20):  # Only plot first 20 dimensions, also for 300D STM
            true_densities = tf.exp(self.marginal_log_density(x_vals, dim))
            start = tf.reduce_min(tf.where(true_densities > 1e-4))
            end = tf.reduce_max(tf.where(true_densities > 1e-4))
            axs[dim // 4, dim % 4].plot(x_vals[start:end, dim], true_densities[start:end], color='b', linewidth=0.1)
            axs[dim // 4, dim % 4].plot(x_vals[start:end, dim],
                                        tf.exp(model.marginal_log_density(x_vals[start:end], dim)),
                                        color='r', linewidth=0.1)
            axs[dim // 4, dim % 4].tick_params(axis="y",direction="inout", pad=2, labelsize=5.)
            axs[dim // 4, dim % 4].tick_params(axis="x", labelsize=5.)
            # axs[dim // 4, dim % 4].set_xticks([])
            # axs[dim // 4, dim % 4].set_yticks([])
        dists_to_target_means = tf.reduce_min(
            tf.linalg.norm(tf.expand_dims(self.target_means, 1)
                           - tf.expand_dims(model.means, 0),
                           axis=2), 1)
        detection_threshold = tf.linalg.norm(6. * tf.ones(model.num_dimensions))
        num_detected = tf.where(dists_to_target_means < detection_threshold).shape[0]
        print(f"Found {num_detected} components.")
        expensive_metrics.update({"num_detected_modes": num_detected, "marginals": fig})
        return expensive_metrics


def make_target(num_dimensions, harder_setting, use_matlab_target=False):
    """
    Create a :py:class:`mixture of Student-T target distribution<gmmvi.experiments.target_distributions.student_t_mixture.StudentTMixture_LNPDF>`
    using the same procedure as :cite:t:`Lin2020` for initializing the weights, means and covariance matrices.

    Parameters:
        num_dimensions: int
            The number of dimensions of the target mixture

        harder_setting: bool
            if True, we use the same procedure that :cite:t:`Lin2020` used for initializing their 300-dimensional
            mixture. If False, we use the procedure for the 20-dimensional setting.

    Returns:
        :py:class:`StudentTMixture_LNPDF`: the instantiated object.
    """
    if harder_setting:
        s = 25
        num_components = 20
    else:
        s = 20
        num_components = 10
    weights = np.ones(num_components) / num_components
    means = np.empty((num_components, num_dimensions))
    covs = np.empty((num_components, num_dimensions, num_dimensions))
    for i in range(0, num_components):
        means[i] = tf.random.uniform(means[i].shape, minval=0, maxval=1) * (2 * s) - s
        covs[i] = 0.1 * num_dimensions * np.random.normal(0, 1, (num_dimensions * num_dimensions)).reshape(
            (num_dimensions, num_dimensions))
        covs[i] = covs[i].transpose().dot(covs[i])
        covs[i] += 1 * np.eye(num_dimensions)
        covs[i] = np.linalg.inv(covs[i])

    if use_matlab_target:
        import scipy.io as spio

        my_dir = pathlib.Path(__file__).parent.resolve()
        mat_dir = os.path.join(my_dir, "tests", "STM matlab data")
        if num_dimensions == 20 and not harder_setting:
            targetdist = spio.loadmat(os.path.join(mat_dir, "target_dist20D2.mat"))
            samps_and_densities = spio.loadmat(os.path.join(mat_dir, "target_dist20d_samples_and_densities2.mat"))
        elif num_dimensions == 300 and harder_setting:
            targetdist = spio.loadmat(os.path.join(mat_dir, "target_dist300D.mat"))
            samps_and_densities = spio.loadmat(os.path.join(mat_dir, "target_dist300d_samples_and_densities.mat"))
        else:
            raise ValueError("Matlab data is not available for the mixture of Student-T experiment with" 
                             f"num_dimensions: {num_dimensions} and harder_setting: {harder_setting}.")
        _, _, _, mweights, mmeans, mprecs = targetdist.values()
        mat_weights = tf.exp(mweights[0])
        mat_means = np.stack([m[0][:, 0] for m in mmeans])
        mat_precs = np.stack([np.linalg.inv(p[0]) for p in mprecs])
        test_samples = np.transpose(samps_and_densities['xSampled'])
        test_densities = samps_and_densities["target_lnpdf"][:, 0]
        target_mixture = StudentTMixture_LNPDF(mat_weights, mat_means, mat_precs)
        assert np.all(np.isclose(target_mixture.log_density(test_samples).numpy(), test_densities, rtol=1e-6))
        return target_mixture
    return StudentTMixture_LNPDF(weights, means, covs)


if __name__ == "__main__":
    stm = make_target(num_dimensions=20)
    print(stm.sample(100))
