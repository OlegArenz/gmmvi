from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from gmmvi.optimization.least_squares import QuadFunc
from gmmvi.models.gmm_wrapper import GmmWrapper


class NgEstimator:
    """ This class provides a common interface for estimating the natural gradient for a Gaussian component.

    There are currently two options for estimating the natural gradient:

    1. The :py:class:`MoreNgEstimator` uses compatible function approximation to estimate the natural gradient from a
       quadratic reward surrogate :cite:p:`Pajarinen2019,Abdolmaleki2015,Peters2008,Sutton1999`.

    2. The :py:class:`SteinNgEstimator` uses Stein's Lemma to estimate the natural gradient using first-order
       information :cite:p:`Lin2019Stein`.

    Parameters:
        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        requires_gradient: bool
            Does this object require first-order information?

        only_use_own_samples: bool
            If true, we do not use importance sampling to update one component based on samples from a different component.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting.
    """

    def __init__(self, temperature, model: GmmWrapper, requires_gradient: bool,
                 only_use_own_samples: bool, use_self_normalized_importance_weights: bool):
        self._model = model
        self._temperature = temperature
        self._requires_gradients = requires_gradient
        self._only_use_own_samples = only_use_own_samples
        self._use_self_normalized_importance_weights = use_self_normalized_importance_weights

    @staticmethod
    def build_from_config(config, temperature, gmm_wrapper):
        """This static method provides a convenient way to create a
        :py:class:`MoreNgEstimator`, or :py:class:`SteinNgEstimator` depending on the provided config.

        Parameters:
            temperature: float
                Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.
        """
        if config["ng_estimator_type"] == "Stein":
            return SteinNgEstimator(temperature=temperature, model=gmm_wrapper, **config['ng_estimator_config'])
        elif config["ng_estimator_type"] == "MORE":
            return MoreNgEstimator(temperature=temperature, model=gmm_wrapper, **config['ng_estimator_config'])
        else:
            raise ValueError(f"config['ng_estimator_type'] is '{config['ng_estimator_type']}' "
                             f"which is an unknown type")

    @property
    def requires_gradients(self) -> bool:
        return self._requires_gradients

    def get_expected_hessian_and_grad(self, samples: tf.Tensor, mapping: tf.Tensor, background_densities: tf.Tensor,
                                      target_lnpdfs: tf.Tensor, target_lnpdfs_grads: tf.Tensor):
        """ Perform the natural gradient estimation, needs to be implemented by the deriving class.

        Parameters:
            samples: tf.Tensor
                a tensor of shape num_samples x num_dimension containing the samples used for the approximation

            mapping: tf.Tensor
                a one-dimensional tensor of integers, storing for every sample from which component it was sampled.

            background_densities: tf.Tensor
                the log probability density of the background distribution (which was used for sampling the provided
                samples). A one-dimensional tensor of size num_samples.

            target_lnpdfs: tf.Tensor
                The rewards are given by the log-densities of the target-distribution,
                :math:`\\log p(\\mathbf{x})`.

            target_lnpdfs_grads: tf.Tensor
                The gradients of the target_lnpdfs with respect to the samples,
                :math:`\\nabla_{\\mathbf{x}}\\log p(\\mathbf{x})`.

        Returns:
            tuple(tf.Tensor, tf.Tensor):

            **expected_hessian_neg** - A tensor of shape num_components x num_dimensions x num_dimensions containing
            for each component an estimate of the (negated) expected Hessian
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}\\mathbf{x}} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}\\right]`

            **expected_gradient_neg** - A tensor of shape num_components x num_dimensions containing
            for each component an estimate of the (negated) expected gradient
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}\\right]`
        """
        raise NotImplementedError

    def get_rewards_for_comp(self, index: int, samples: tf.Tensor, mapping: tf.Tensor, component_log_densities,
                             log_ratios: tf.Tensor, log_ratio_grads: tf.Tensor, background_densities: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self._only_use_own_samples:
            own_sample_indices = tf.reshape(tf.where(mapping == index), [-1])
            my_samples = tf.gather(samples, own_sample_indices)
            my_rewards = tf.gather(log_ratios, own_sample_indices)
            my_reward_grads = tf.gather(log_ratio_grads, own_sample_indices)
        #    my_background_densities = tf.gather(background_densities, own_sample_indices)
            my_background_densities = tf.gather(component_log_densities[index], own_sample_indices)
            my_component_log_densities = tf.gather(component_log_densities[index], own_sample_indices)
            return my_samples, my_rewards, my_reward_grads, my_background_densities, my_component_log_densities
        else:
            return samples, log_ratios, log_ratio_grads, background_densities, component_log_densities[index]


class SteinNgEstimator(NgEstimator):
    """ Use Stein's Lemma to estimate the natural gradient using first-order information.
    See :cite:p:`Lin2019Stein`.

    Parameters:
        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        only_use_own_samples: bool
            If true, we do not use importance sampling to update one component based on samples from a different component.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting.
    """

    def __init__(self, temperature, model, only_use_own_samples: bool, use_self_normalized_importance_weights: bool):
        super(SteinNgEstimator, self).__init__(temperature, model, True, only_use_own_samples,
                                               use_self_normalized_importance_weights)

    @staticmethod
    def _stable_expectation(log_weights, log_values):
        n = tf.cast(tf.shape(log_weights)[0], tf.float32)
        lswe, signs = tfp.math.reduce_weighted_logsumexp(
            tf.expand_dims(log_weights, 1) + tf.math.log(tf.math.abs(log_values)),
            w=tf.math.sign(log_values), axis=0, return_sign=True)
        return 1 / n * signs * tf.exp(lswe)

    def _get_expected_gradient_and_hessian_standard_iw(self, chol_cov, mean, component_log_densities, samples,
                                                              background_mixture_densities, log_ratio_grads):
        log_importance_weights = component_log_densities - background_mixture_densities
        expected_gradient = self._stable_expectation(log_importance_weights, log_ratio_grads)

        if self._model.diagonal_covs:
            prec_times_diff = tf.expand_dims(1 / (chol_cov ** 2), 1) \
                            * tf.transpose(samples - mean)
            prec_times_diff_times_grad = tf.transpose(prec_times_diff) * log_ratio_grads
        else:
            prec_times_diff = tf.linalg.cholesky_solve(chol_cov, tf.transpose(samples - mean))
            prec_times_diff_times_grad = \
                tf.expand_dims(tf.transpose(prec_times_diff), 1) * tf.expand_dims(log_ratio_grads, -1)
            log_importance_weights = tf.expand_dims(log_importance_weights, 1)
        expected_hessian = self._stable_expectation(log_importance_weights, prec_times_diff_times_grad)
        return expected_gradient, expected_hessian

    def _get_expected_gradient_and_hessian_self_normalized_iw(self, chol_cov, mean, component_log_densities, samples,
                                                              background_mixture_densities, log_ratio_grads):
        log_weights = component_log_densities - background_mixture_densities
        log_weights -= tf.reduce_logsumexp(log_weights, axis=0, keepdims=True)
        weights = tf.exp(log_weights)
        importance_weights = weights / tf.reduce_sum(weights, axis=0, keepdims=True)
        weighted_gradients = tf.expand_dims(importance_weights, 1) * log_ratio_grads
        if self._model.diagonal_covs:
            prec_times_diff = tf.expand_dims(1 / (chol_cov ** 2), 1) \
                            * tf.transpose(samples - mean)
            expected_hessian = tf.reduce_sum(tf.transpose(prec_times_diff) * weighted_gradients, 0)
        else:
            prec_times_diff = tf.linalg.cholesky_solve(chol_cov, tf.transpose(samples - mean))
            expected_hessian = tf.reduce_sum(
                tf.expand_dims(tf.transpose(prec_times_diff), 1) * tf.expand_dims(weighted_gradients, -1), 0)
            expected_hessian = 0.5 * (expected_hessian + tf.transpose(expected_hessian))
        expected_gradient = tf.reduce_sum(weighted_gradients, 0)
        return expected_gradient, expected_hessian

    def _get_expected_gradient_and_hessian_for_comp(self, i, my_component_log_densities, my_samples,
                                                my_background_densities, my_log_ratios_grad):
        if self._use_self_normalized_importance_weights:
            expected_gradient, expected_hessian = \
                self._get_expected_gradient_and_hessian_self_normalized_iw(
                    self._model.chol_cov[i], self._model.means[i], my_component_log_densities, my_samples,
                    my_background_densities, my_log_ratios_grad)
        else:
            expected_gradient, expected_hessian = \
                self._get_expected_gradient_and_hessian_standard_iw(
                    self._model.chol_cov[i], self._model.means[i], my_component_log_densities, my_samples,
                    my_background_densities, my_log_ratios_grad)
        return expected_gradient, expected_hessian

    def get_expected_hessian_and_grad(self, samples: tf.Tensor, mapping: tf.Tensor, background_densities: tf.Tensor,
                                      target_lnpdfs: tf.Tensor, target_lnpdfs_grads: tf.Tensor):
        """ Estimates the natural gradient using Stein's Lemma :cite:p:`Lin2019Stein`. The expected gradient is
        a simple importance-weighted Monte-Carlo estimate based on the provided *target_lnpdfs_grads* and the gradients
        of the component log-densities. The expected Hessians are estimated as
        :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}\\mathbf{x}} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}\\right] \\approx - \\sum_{\\mathbf{x}_i} w_i \\Sigma^{-1} (\\mathbf{x}_i - \\mu) \\nabla_{\\mathbf{x}_i} g_{\\mathbf{x}_i}^T`,
        where :math:`g_{\\mathbf{x}_i} = \\nabla_{\\mathbf{x}_i} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}` is the
        gradient of the log-ratio with respect to the corresponding sample.

        Parameters:
            samples: tf.Tensor
                a tensor of shape num_samples x num_dimension containing the samples used for the approximation

            mapping: tf.Tensor
                a one-dimensional tensor of integers, storing for every sample from which component it was sampled.

            background_densities: tf.Tensor
                the log probability density of the background distribution (which was used for sampling the provided
                samples). A one-dimensional tensor of size num_samples.

            target_lnpdfs: tf.Tensor
                The rewards are given by the log-densities of the target-distribution,
                :math:`\\log p(\\mathbf{x})`.

            target_lnpdfs_grads: tf.Tensor
                The gradients of the target_lnpdfs with respect to the samples,
                :math:`\\nabla_{\\mathbf{x}}\\log p(\\mathbf{x})`.

        Returns:
            tuple(tf.Tensor, tf.Tensor):
            **expected_hessian_neg** - A tensor of shape num_components x num_dimensions x num_dimensions containing
            for each component an estimate of the (negated) expected Hessian
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}\\mathbf{x}} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}\\right]`

            **expected_gradient_neg** - A tensor of shape num_components x num_dimensions  containing
            for each component an estimate of the (negated) expected gradient
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}} \\log \\frac{p(\\mathbf{x}}{q(\\mathbf{x}}\\right]`
        """

        num_components = self._model.num_components
        relative_mapping = mapping - tf.reduce_max(mapping) + num_components - 1

        model_densities, model_densities_grad, component_log_densities = self._model.log_density_and_grad(samples)
        log_ratios = target_lnpdfs - model_densities
        log_ratio_grads = target_lnpdfs_grads - model_densities_grad

        expected_hessian_neg = tf.TensorArray(tf.float32, size=num_components)
        expected_gradient_neg = tf.TensorArray(tf.float32, size=num_components)
        for i in tf.range(num_components):
            my_samples, my_log_ratios, my_log_ratios_grad, my_background_densities, my_component_log_densities = \
                self.get_rewards_for_comp(i, samples, relative_mapping, component_log_densities,
                                          log_ratios, log_ratio_grads, background_densities)
            expected_gradient, expected_hessian = \
                self._get_expected_gradient_and_hessian_for_comp(i, my_component_log_densities, my_samples,
                                                                 my_background_densities, my_log_ratios_grad)
            expected_hessian_neg = expected_hessian_neg.write(i, -expected_hessian)
            expected_gradient_neg = expected_gradient_neg.write(i, -expected_gradient)
        expected_hessian_neg = tf.convert_to_tensor(expected_hessian_neg.stack())
        expected_gradient_neg = tf.convert_to_tensor(expected_gradient_neg.stack())
        return expected_hessian_neg, expected_gradient_neg


class MoreNgEstimator(NgEstimator):
    """ Use compatible function approximation to estimate the natural gradient from a quadratic reward surrogate.
    See :cite:p:`Pajarinen2019,Abdolmaleki2015,Peters2008,Sutton1999`.

    Parameters:
        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        only_use_own_samples: bool
            If true, we do not use importance sampling to update one component based on samples from a different component.

        initial_l2_regularizer: float
            The l2_regularizer is as regularizer during weighted least-squares (ridge regression) for fitting the
            compatible surrogate.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting.
    """

    def __init__(self, temperature, model, only_use_own_samples: bool,
                 initial_l2_regularizer: float, use_self_normalized_importance_weights: bool):
        super(MoreNgEstimator, self).__init__(temperature, model, True, only_use_own_samples,
                                              use_self_normalized_importance_weights)
        tf.assert_equal(self._model.l2_regularizers, initial_l2_regularizer)
        self.least_square_fitter = QuadFunc(self._model.num_dimensions)

    def get_expected_hessian_and_grad(self, samples: tf.Tensor, mapping: tf.Tensor, background_densities: tf.Tensor,
                                      target_lnpdfs: tf.Tensor, target_lnpdfs_grads: tf.Tensor) \
            -> [tf.Tensor, tf.Tensor]:
        """ Estimates the natural gradient using compatible function approximation. This method does not require / make
        use of the provided gradients, but only uses the function evaluations *target_lnpdfs* for estimating the
        natural gradient. The method fits a quadratic reward function
        :math:`\\tilde{R}(\\mathbf{x}) = \\mathbf{x}^T \\mathbf{R} \\mathbf{x} + \\mathbf{x}^T \\mathbf{r} + r_0`
        to approximate the target distribution  using importance-weighted least squares where the targets are given by
        *target_lnpdfs*, :math:`\\log p(\\mathbf{x})`.
        The natural gradient estimate, can then be computed from the coefficients :math:`\\mathbf{R}`
        and :math:`\\mathbf{r}`.


        Parameters:
            samples: tf.Tensor
                a tensor of shape num_samples x num_dimension containing the samples used for the approximation

            mapping: tf.Tensor
                a one-dimensional tensor of integers, storing for every sample from which component it was sampled.

            background_densities: tf.Tensor
                the log probability density of the background distribution (which was used for sampling the provided
                samples). A one-dimensional tensor of size num_samples.

            target_lnpdfs: tf.Tensor
                The rewards are given by the (unnormalized) log-densities of the target-distribution,
                :math:`\\log p(\\mathbf{x})`.

            target_lnpdfs_grads: tf.Tensor
                The gradients of the target_lnpdfs with respect to the samples (not used),
                :math:`\\nabla_{\\mathbf{x}}\\log p(\\mathbf{x})`.

        Returns:
            tuple(tf.Tensor, tf.Tensor):

            **expected_hessian_neg** - A tensor of shape num_components x num_dimensions x num_dimensions containing
            for each component an estimate of the (negated) expected Hessian
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}\\mathbf{x}} \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`

            **expected_gradient_neg** - A tensor of shape num_components x num_dimensions  containing
            for each component an estimate of the (negated) expected gradient
            :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}} \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
        """
        num_components = tf.shape(self._model.means)[0]

        expected_hessian_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        expected_gradient_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        relative_mapping = mapping - tf.reduce_max(mapping) + num_components - 1

        model_densities, component_log_densities = self._model.log_densities_also_individual(samples)

        log_ratios = target_lnpdfs - model_densities
        log_ratio_grads = tf.zeros(tf.shape(samples[0]))

        for i in tf.range(num_components):
            my_samples, my_rewards, _, my_background_densities, my_component_log_densities = \
                self.get_rewards_for_comp(i, samples, relative_mapping, component_log_densities, log_ratios,
                                          log_ratio_grads, background_densities)

            log_weights = my_component_log_densities - my_background_densities
            if self._use_self_normalized_importance_weights:
                log_weights -= tf.reduce_logsumexp(log_weights, axis=0, keepdims=True)
                weights = tf.exp(log_weights)
                my_importance_weights = weights / tf.reduce_sum(weights, axis=0, keepdims=True)
            else:
                my_importance_weights = tf.exp(log_weights)
            reward_quad, reward_lin, const_term = self.least_square_fitter.fit_quadratic(self._model.l2_regularizers[i],
                                                                                         tf.shape(my_samples)[0],
                                                                                         my_samples, my_rewards,
                                                                                         my_importance_weights,
                                                                                         self._model.means[i],
                                                                                         self._model.chol_cov[i])

            this_G_hat = reward_quad
            expected_hessian_neg = expected_hessian_neg.write(i, this_G_hat)
            this_g_hat = tf.reshape(reward_quad @ tf.expand_dims(self._model.means[i], axis=1)
                                    - tf.expand_dims(reward_lin, axis=1), [self._model.num_dimensions])
            expected_gradient_neg = expected_gradient_neg.write(i, this_g_hat)
        expected_hessian_neg = tf.convert_to_tensor(expected_hessian_neg.stack())
        expected_gradient_neg = tf.convert_to_tensor(expected_gradient_neg.stack())
        return expected_hessian_neg, expected_gradient_neg
