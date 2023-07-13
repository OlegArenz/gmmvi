import tensorflow as tf
import tensorflow_probability as tfp

from gmmvi.models.gmm_wrapper import GmmWrapper


class WeightUpdater:
    """ This class provides a common interface for updating the weights of the mixture model.

    It currently supports two options:

    1. The :py:class:`DirectWeightUpdater` straightforwardly applies a natural gradient update using the given stepsize.

    2. The :py:class:`TrustRegionBasedWeightUpdater` treats the stepsize as a trust-region constraint between the
       current distribution over weights and the updated distribution, and performs the largest step in the direction
       of the natural gradient that confines to this constraint.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the weights.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting for estimating the natural gradient.
    """
    def __init__(self, model: GmmWrapper, temperature: float, use_self_normalized_importance_weights: bool):
        self.model = model
        self.temperature = temperature
        self.use_self_normalized_importance_weights = use_self_normalized_importance_weights

    @staticmethod
    def build_from_config(config, gmm_wrapper):
        """This static method provides a convenient way to create a
        :py:class:`DirectWeightUpdater` or :py:class:`TrustRegionBasedWeightUpdater` depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model for which we want to update the weights.
        """
        if config["weight_updater_type"] == "direct":
            return DirectWeightUpdater(gmm_wrapper, temperature=config['temperature'],
                                       **config["weight_updater_config"])
        elif config["weight_updater_type"] == "trust-region":
            return TrustRegionBasedWeightUpdater(gmm_wrapper, temperature=config['temperature'],
                                                 **config["weight_updater_config"])
        else:
            raise ValueError(
                f"config['weight_updater_type'] is '{config['weight_updater_type']}' which is an unknown type")

    def _get_expected_log_ratios(self, samples, background_mixture_densities, target_lnpdfs):
        model_densities, component_log_densities = self.model.log_densities_also_individual(samples)
        log_ratios = target_lnpdfs - self.temperature * model_densities
        if self.use_self_normalized_importance_weights:
            log_weights = component_log_densities - background_mixture_densities
            log_weights -= tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
            weights = tf.exp(log_weights)
            importance_weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
            expected_log_ratios = tf.linalg.matvec(importance_weights, log_ratios)
        else:
            n = tf.cast(tf.shape(samples)[0], tf.float32)
            log_importance_weights = component_log_densities - background_mixture_densities
            lswe, signs = tfp.math.reduce_weighted_logsumexp(
                log_importance_weights + tf.math.log(tf.math.abs(log_ratios)),
                w=tf.math.sign(log_ratios), axis=1, return_sign=True)
            expected_log_ratios = 1 / n * signs * tf.exp(lswe)

        component_rewards = self.temperature * self.model.log_weights + expected_log_ratios
        self.model.store_rewards(component_rewards)
        return expected_log_ratios

    def update_weights(self, samples: tf.Tensor, background_mixture_densities: tf.Tensor,
                       target_lnpdfs: tf.Tensor, stepsize: float):
        """
        Computes the importance weights and uses them to estimate the natural gradient. Performs a natural gradient step
        using the given stepsize.

        Parameters:
            samples: tf.Tensor
                The samples for which the *background_mixture_densities* and *target_lnpdfs* were evaluated. Needed
                for computing the importance weights.

            background_mixture_densities: tf.Tensor
                The log_densities of the *samples* for the distribution that was effectively used for obtain the
                provided *samples*. Needed for computing the importance weights.

            target_lnpdfs: tf.Tensor
                The log densities of the target distribution evaluated for the provided *samples*,
                :math:`\\log p(\\mathbf{x})`.

            stepsize: float
                The stepsize that should be used for performing the weight update.
        """
        expected_log_ratios = self._get_expected_log_ratios(samples, background_mixture_densities, target_lnpdfs)
        self._update_weights_from_expected_log_ratios(expected_log_ratios, stepsize)

    def _update_weights_from_expected_log_ratios(self, expected_log_ratios: tf.Tensor, stepsize: float):
        raise NotImplementedError


class DirectWeightUpdater(WeightUpdater):
    """ This class can be used for directly updating the weights along the natural gradient, using the given stepsize.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the weights.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting for estimating the natural gradient.
    """
    def __init__(self, model: GmmWrapper, temperature: float, use_self_normalized_importance_weights: bool):
        super(DirectWeightUpdater, self).__init__(model, temperature, use_self_normalized_importance_weights)

    def _update_weights_from_expected_log_ratios(self, expected_log_ratios: tf.Tensor, stepsize: tf.float32):
        """ Directly uses the stepsize to update the weights towards the expected_log_ratios

        Parameters:
            expected_log_ratios: tf.Variable(tf.float32)
                A vector containing an (MC-)estimate of
                :math:`\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
                , for every component o.

            stepsize: tf.float32
                The stepsize :math:`\\beta`, the new weights are proportional to
                :math:`\\exp(\\text{old\\_log\\_weights} + \\beta * \\text{expected\\_log\\_ratios})`.
        """
        if self.model.num_components > 1:
            unnormalized_weights = self.model.log_weights + stepsize / self.temperature * expected_log_ratios
            new_log_probs = unnormalized_weights - tf.reduce_logsumexp(unnormalized_weights)
            new_log_probs = tf.math.maximum(new_log_probs, -69.07)  # lower bound weights to 1e-30
            new_log_probs -= tf.reduce_logsumexp(new_log_probs)
            self.model.replace_weights(new_log_probs)


class TrustRegionBasedWeightUpdater(WeightUpdater):
    """ This class can be used for performing the weight update by treating the stepsize as a KL constraint.

    Constrains the KL between the updated weights and the current weights :math:`\\text{KL}(q_\\text{new}(o) || q(o))`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the weights.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.

        use_self_normalized_importance_weights: bool
            if true, use self-normalized importance weighting (normalizing the importance weights such they sum to one),
            rather than standard importance weighting for estimating the natural gradient.
    """

    def __init__(self, model: GmmWrapper, temperature: float, use_self_normalized_importance_weights: bool):
        super(TrustRegionBasedWeightUpdater, self).__init__(model, temperature, use_self_normalized_importance_weights)

    def kl(self, eta: tf.float32, component_rewards: tf.Tensor) -> [tf.float32, tf.Tensor]:
        """
        Computes the Kullback-Leibler divergence between the updated component and current component, when updating
        with stepsize eta along the natural gradient.

        Parameters:
            eta: tf.float32
                The stepsize for which the KL divergence should be computed.

            component_rewards: tf.float32
                A tensor containing an MC-estimate of the expected reward (expected logratios) of each component,
                :math:`R(o)=\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`

        Returns:
            tuple(float, tf.Tensor):

            **kl** - a float corresponding to the KL between the updated and previous weight distribution.

            **new_log_weights** - log of the updated weights, :math:`\\log(q_\\text{new}(o))`.
        """
        unnormalized_weights = (eta + 1)/(self.temperature + eta) * self.model.log_weights \
                               + 1. / (self.temperature + eta) * component_rewards
        new_log_weights = unnormalized_weights - tf.reduce_logsumexp(unnormalized_weights)
        new_log_weights = tf.math.maximum(new_log_weights, -69.07)  # lower bound weights to 1e-30
        new_log_weights -= tf.reduce_logsumexp(new_log_weights)

        kl = tf.reduce_sum(tf.exp(new_log_weights) * (new_log_weights - self.model.log_weights))
        return kl, new_log_weights

    def _bracketing_search(self, expected_log_ratios: tf.Tensor, kl_bound: tf.float32, lower_bound: tf.float32,
                           upper_bound: tf.float32) -> [tf.float32, tf.float32, tf.Tensor]:
        """ This method finds the largest stepsize eta, such that the updated weight distribution stays within a
        KL-constrained trust-region around the current distribution. Here we use a simple bracketing search, which
        can be efficiently performed within a Tensorflow graph. The procedure simple keeps track of a
        lower bound and an upper bound on the optimal stepsize and recursively evaluates the arithmetic mean of both
        bounds. If this mean-stepsize results in a too large KL divergence, it becomes the new lower bound;
        otherwise the new upper bound.

        Parameters:
            expected_log_ratios: tf.Tensor
                A vector containing an (MC-)estimate of
                :math:`\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
                , for every component o.

            kl_bound: tf.float32
                The trust region constraint

            lower_bound: tf.float32
                The initial lower bound on the stepsize

            upper_bound:
                The initial upper bound on the stepsize

        Returns:
            tuple(tf.float32, tf.float32, tf.Tensor):

            **new_lower_bound** - The lower bound after a stopping criterion was reached.

            **new_upper_bound** - The upper bound after a stopping criterion was reached.

            **new_log_weights** - log of the updated weights, :math:`\\log(q_\\text{new}(o))`.

        """
        log_eta = 0.5 * (upper_bound + lower_bound)
        upper_bound_satisfies_constraint = False
        kl = -1.
        eta = -1.
        new_log_weights = self.model.log_weights
        for _ in tf.range(50):
            eta = tf.exp(log_eta)
            diff = tf.math.abs(tf.exp(upper_bound) - tf.exp(lower_bound))
            if diff < 1e-1:
                break

            kl, new_log_weights = self.kl(eta, expected_log_ratios)

            if tf.abs(kl_bound - kl) < 1e-1 * kl_bound:
                # we indicate that we already found a sufficiently good eta, by setting:
                lower_bound = upper_bound
                break

            if kl_bound > kl:
                upper_bound = log_eta
                upper_bound_satisfies_constraint = True
            else:
                lower_bound = log_eta
            log_eta = 0.5 * (upper_bound + lower_bound)

        if lower_bound == upper_bound:
            return kl, eta, new_log_weights
        # We could not find the optimal multiplier, but if the upper bound is large enough to satisfy the constraint,
        # we can still make an update
        if upper_bound_satisfies_constraint:
            kl, new_log_weights = self.kl(tf.exp(upper_bound), expected_log_ratios)
            return kl, tf.exp(upper_bound), new_log_weights

        return -1., -1., self.model.log_weights

    def _update_weights_from_expected_log_ratios(self, expected_log_ratios, kl_bound):
        """Perform the weight update, treating the stepsize as constraint on the KL-divergence.

        Parameters:
            expected_log_ratios: tf.Variable(tf.float32)
                A vector containing an (MC-)estimate of
                :math:`\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
                , for every component o.

            stepsize: tf.float32
                The stepsize :math:`\\epsilon`, the new weights will satisfy
                :math:`\\text{KL}(q_\\text{new}(o) || q(o)) < \\epsilon`.
        """
        if self.model.num_components > 1:
            lower_bound = tf.constant(-45.)
            upper_bound = tf.constant(45.)
            kl, eta, new_log_weights = self._bracketing_search(expected_log_ratios, kl_bound, lower_bound, upper_bound)
            self.model.replace_weights(new_log_weights)
            # tf.print(kl)
            #  tf.print(eta)
