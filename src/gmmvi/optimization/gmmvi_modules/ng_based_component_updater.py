import tensorflow as tf
from gmmvi.models.gmm_wrapper import GmmWrapper

class NgBasedComponentUpdater:
    """ This class provides a common interface for updating the Gaussian components along the natural gradient.

    The Gaussian components of the mixture model, are updated by updating their parameters (their mean and covariance)
    based on previously computed estimates of the natural gradient (see :py:class:`NgEstimator`) and stepsizes
    (see :py:class:`ComponentStepsizeAdaptation`).

    There are currently three options for updating the components:

    1. The :py:class:`DirectNgBasedComponentUpdater` straightforwardly performs the natural gradient with the given
       stepsize.

    2. The :py:class:`NgBasedComponentUpdaterIblr` uses the improved Bayesian learning rate to ensure positive definite
       covariance matrices.

    3. The :py:class:`KLConstrainedNgBasedComponentUpdater` treats the stepsize as a trust-region constraint.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.
    """

    def __init__(self, model: GmmWrapper, temperature: float):
        self.model = model
        self.temperature = temperature

    @staticmethod
    def build_from_config(config, gmm_wrapper):
        """This static method provides a convenient way to create a
        :py:class:`DirectNgBasedComponentUpdater`, :py:class:`NgBasedComponentUpdaterIblr`
        or :py:class:`KLConstrainedNgBasedComponentUpdater` depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model for which we want to update the components
        """
        if config["ng_based_updater_type"] == "trust-region":
            return KLConstrainedNgBasedComponentUpdater(gmm_wrapper, temperature=config['temperature'],
                                                        **config["ng_based_updater_config"])
        elif config["ng_based_updater_type"] == "direct":
            return DirectNgBasedComponentUpdater(gmm_wrapper, temperature=config['temperature'],
                                                 **config["ng_based_updater_config"])
        elif config["ng_based_updater_type"] == "iBLR":
            return NgBasedComponentUpdaterIblr(gmm_wrapper, temperature=config['temperature'],
                                               **config["ng_based_updater_config"])
        else:
            raise ValueError(
                f"config['ng_based_updater_type'] is '{config['ng_based_updater_type']}' which is an unknown type")

    def apply_NG_update(self, expected_hessians_neg: tf.Tensor, expected_gradients_neg: tf.Tensor,
                        stepsizes: tf.Tensor):
        """ Update the components based on the estimates of the natural gradients (provided in terms of the negated
        expected Hessian and expected gradients) and the given component-specific stepsizes.

        Parameters:
            expected_hessians_neg: tf.Tensor
                A three-dimensional tensor of shape num_components x num_dimensions x num_dimensions, containing an
                estimate of
                :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}\\mathbf{x}} \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
                for each component.

            expected_gradients_neg: tf.Tensor
                A two-dimensional tensor of shape num_components x num_dimensions x num_dimensions, containing an
                estimate of
                :math:`-\\mathbb{E}_{q(\\mathbf{x}|o)}\\left[ \\nabla_{\\mathbf{x}} \\log \\frac{p(\\mathbf{x})}{q(\\mathbf{x})}\\right]`
                for each component.

            stepsizes: tf.Tensor
                A one-dimensional tensor of shape num_components, containing the stepsize for each component.
        """
        raise NotImplementedError


class DirectNgBasedComponentUpdater(NgBasedComponentUpdater):
    """ This class straightforwardly performs the natural gradient update with the given stepsize.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.
    """

    def __init__(self, model: GmmWrapper, temperature: float):
        super(DirectNgBasedComponentUpdater, self).__init__(model, temperature)

    def apply_NG_update(self, expected_hessians_neg: tf.Tensor, expected_gradients_neg: tf.Tensor,
                        stepsizes: tf.Tensor):
        means = tf.TensorArray(tf.float32, size=self.model.num_components)
        chols = tf.TensorArray(tf.float32, size=self.model.num_components)
        successes = tf.TensorArray(tf.bool, size=self.model.num_components)

        for i in range(self.model.num_components):
            old_chol = self.model.chol_cov[i]
            old_mean = self.model.means[i]
            old_inv_chol = tf.linalg.inv(old_chol)
            old_precision = tf.transpose(old_inv_chol) @ old_inv_chol
            old_lin = old_precision @ tf.expand_dims(old_mean, 1)

            delta_precision = expected_hessians_neg[i]
            delta_lin = expected_hessians_neg[i] @ tf.expand_dims(old_mean, 1) \
                        - tf.expand_dims(expected_gradients_neg[i], 1)

            new_lin = old_lin + stepsizes[i] * delta_lin
            new_precision = old_precision + stepsizes[i] * delta_precision
            new_mean = tf.reshape(tf.linalg.solve(new_precision, new_lin), [-1])
            new_cov = tf.linalg.inv(new_precision)
            new_chol = tf.linalg.cholesky(new_cov)

            if tf.reduce_any(tf.math.is_nan(new_chol)):
                success = False
                new_mean = old_mean
                new_chol = old_chol
            else:
                success = True

            means = means.write(i, new_mean)
            chols = chols.write(i, new_chol)
            successes = successes.write(i, success)

        chols = chols.stack()
        means = means.stack()
        successes = successes.stack()

        updated_l2_reg = tf.where(successes,
                                  tf.maximum(0.5 * self.model.l2_regularizers, self.model.initial_regularizer),
                                  tf.minimum(1e-6, 10 * self.model.l2_regularizers))
        self.model.l2_regularizers.assign(updated_l2_reg)

        self.model.replace_components(means, chols)
        self.model.num_received_updates.assign_add(tf.ones(self.model.num_components))


class NgBasedComponentUpdaterIblr(NgBasedComponentUpdater):
    """ This class updates the component using the improved Bayesian learning rule.

     The iBLR performs Riemannian gradient descent and ensures positive definite covariance matrices :cite:p:`Lin2020`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.
    """

    def __init__(self, model: GmmWrapper, temperature: float):
        super(NgBasedComponentUpdaterIblr, self).__init__(model, temperature)

    def apply_NG_update(self, expected_hessians_neg: tf.Tensor, expected_gradients_neg: tf.Tensor,
                        stepsizes: tf.Tensor):
        means = tf.TensorArray(tf.float32, size=self.model.num_components)
        chols = tf.TensorArray(tf.float32, size=self.model.num_components)
        successes = tf.TensorArray(tf.bool, size=self.model.num_components)

        for i in range(self.model.num_components):
            old_chol = self.model.chol_cov[i]
            old_mean = self.model.means[i]

            if self.model.diagonal_covs:
                correction_term = stepsizes[i] / 2 * expected_hessians_neg[i] * old_chol \
                                  * old_chol * expected_hessians_neg[i]
                old_inv_chol = 1. / old_chol
                old_precision = old_inv_chol * old_inv_chol
            else:
                correction_term = stepsizes[i] / 2 * expected_hessians_neg[i] @ old_chol \
                                  @ tf.transpose(old_chol) @ expected_hessians_neg[i]
                old_inv_chol = tf.linalg.inv(old_chol)
                old_precision = tf.transpose(old_inv_chol) @ old_inv_chol

            delta_precision = expected_hessians_neg[i] + correction_term
            delta_mean = - expected_gradients_neg[i]

            if self.model.num_received_updates[i] == 0:
                # first time: no mean update
                new_mean = old_mean
            else:
                if self.model.diagonal_covs:
                    new_mean = old_mean + stepsizes[i] * old_chol * old_chol * delta_mean
                else:
                    new_mean = old_mean + tf.squeeze(
                        stepsizes[i] * old_chol @ tf.transpose(old_chol) @ tf.expand_dims(delta_mean, 1))

            new_precision = old_precision + stepsizes[i] * delta_precision
            if self.model.diagonal_covs:
                new_cov = 1. / new_precision
                new_chol = tf.math.sqrt(new_cov)
            else:
                new_cov = tf.linalg.inv(new_precision)
                new_chol = tf.linalg.cholesky(new_cov)

            if tf.reduce_any(tf.math.is_nan(new_chol)):
                success = False
                new_mean = old_mean
                new_chol = old_chol
            else:
                success = True

            means = means.write(i, new_mean)
            chols = chols.write(i, new_chol)
            successes = successes.write(i, success)

        chols = chols.stack()
        means = means.stack()
        successes = successes.stack()

        updated_l2_reg = tf.where(successes,
                                  tf.maximum(0.5 * self.model.l2_regularizers, self.model.initial_regularizer),
                                  tf.minimum(1e-6, 10 * self.model.l2_regularizers))
        self.model.l2_regularizers.assign(updated_l2_reg)

        self.model.replace_components(means, chols)
        self.model.num_received_updates.assign_add(tf.ones(self.model.num_components))


class KLConstrainedNgBasedComponentUpdater(NgBasedComponentUpdater):
    """ Updates the component by treating the stepsize as a constraint on the KL-divergence to the current component.

    This class updates the component by performing the largest update along the natural gradient direction, that
    confines with a trust-region constraint on the Kullback-Leibler divergence with respect to the current
    component :cite:p:`Arenz2020`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to update the components.

        temperature: float
            Usually temperature=1., can be used to scale the importance of maximizing the model entropy.
    """

    def __init__(self, model: GmmWrapper, temperature: float):
        super(KLConstrainedNgBasedComponentUpdater, self).__init__(model, temperature)

    def kl(self, eta: tf.float32, old_lin_term: tf.Tensor, old_precision: tf.Tensor, old_inv_chol: tf.Tensor,
           reward_lin: tf.Tensor, reward_quad: tf.Tensor, kl_const_part: tf.float32, old_mean: tf.Tensor,
           eta_in_logspace: bool) -> [tf.float32, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Computes the Kullback-Leibler divergence between the updated component and current component, when updating
        with stepsize eta along the natural gradient.

        Parameters:
            eta: tf.float32
                The stepsize for which the KL divergence should be computed.

            old_lin_term: tf.Tensor
                The linear term of the canonical Gaussian form of the current component. A one-dimensional tensor
                of shape num_dimensions.

            old_precision: tf.Tensor
                The precision matrix of the current component. A two-dimensional tensor of shape
                num_dimensions x num_dimensions.

            old_inv_chol: tf.Tensor
                The inverse of the Cholesky matrix of the current component. A two-dimensional tensor of shape
                num_dimensions x num_dimensions.

            reward_lin: tf.Tensor
                When using :py:class:`MORE` to estimate the natural gradient, this tensor correspond to the linear
                coefficient of the quadratic reward model. When using Stein's Lemma, this term can be computed
                from the expected gradient and expected Hessian.

            reward_quad: tf.Tensor
                When using :py:class:`MORE` to estimate the natural gradient, this tensor correspond to the quadratic
                coefficient of the quadratic reward model. When using Stein's Lemma, this term can be computed
                from the expected Hessian.

            kl_const_part: tf.float32
                A term of the KL divergence that can be precomputed as it does not depend on the parameters of the
                updated component.

            old_mean: tf.Tensor
                The mean of the current component.

            eta_in_logspace: bool
                if true, the provided eta is given in log-space.

        Returns:
            tuple(float, tf.Tensor, tf.Tensor, tf.Tensor):

            **kl** - a float corresponding to the KL between the updated component and the old component.

            **new_mean** - a tensor specifying the mean of the updated component.

            **new_precision** - a tensor specifying the precision of the updated component.

            **inv_chol_inv** - a tensor specifying the inverse of the Cholesky of the precision matrix of the updated
            component.
        """
        if eta_in_logspace:
            eta = tf.exp(eta)

        new_lin = (eta * old_lin_term + reward_lin) / eta
        new_precision = (eta * old_precision + reward_quad) / eta
        if self.model.diagonal_covs:
            chol_precision = tf.math.sqrt(new_precision)
            new_mean = 1./new_precision * new_lin
            inv_chol_inv = 1./chol_precision
            diff = old_mean - new_mean
        #   new_logdet = -2 * tf.reduce_sum(tf.math.log(chol_precision))
        #   trace_term = tf.reduce_sum(old_precision / new_precision)
        #   kl = 0.5 * (kl_const_part - new_logdet + trace_term
        #               + tf.reduce_sum(tf.square(old_inv_chol * diff)))
            # this is numerically more stable:
            kl = 0.5 * (tf.maximum(0., tf.reduce_sum(tf.math.log(new_precision/old_precision)
                                                     + old_precision/new_precision)
                        - self.model.num_dimensions)
                        + tf.reduce_sum(tf.square(old_inv_chol * diff)))
        else:
            chol_precision = tf.linalg.cholesky(new_precision)
            if tf.reduce_any(tf.math.is_nan(chol_precision)):
                new_mean = old_mean
                inv_chol_inv = old_inv_chol
                new_precision = old_precision
                kl = tf.float32.max
            else:
                new_mean = tf.reshape(tf.linalg.cholesky_solve(chol_precision, tf.expand_dims(new_lin, 1)), [-1])
                new_logdet = -2 * tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(chol_precision)))
                inv_chol_inv = tf.linalg.inv(chol_precision)
                trace_term = tf.square(tf.norm(inv_chol_inv @ tf.transpose(old_inv_chol)))
                diff = old_mean - new_mean
                kl = 0.5 * (kl_const_part - new_logdet + trace_term
                            + tf.reduce_sum(tf.square(tf.linalg.matvec(old_inv_chol, diff))))
        return kl, new_mean, new_precision, inv_chol_inv

    def bracketing_search(self, kl_bound: tf.float32, lower_bound: tf.float32, upper_bound: tf.float32,
                          old_lin_term: tf.Tensor, old_precision: tf.Tensor, old_inv_chol: tf.Tensor,
                          reward_lin_term: tf.Tensor, reward_quad_term: tf.Tensor, kl_const_part: tf.float32,
                          old_mean: tf.Tensor, eta_in_logspace: tf.float32) -> [tf.float32, tf.float32]:
        """ This method finds the largest stepsize eta, such that the updated component stays within a KL-constrained
        trust-region around the current component. Whereas, :cite:p:`Arenz2020` used L-BFGS-B to solve this convex
        optimization problem, here we use a simple bracketing search, which seems to be more robust (by not relying on
        gradients, and can be efficiently performed within a Tensorflow graph. The procedure simple keeps track of a
        lower bound and an upper bound on the optimal stepsize and recursively evaluates the arithmetic mean of both
        bounds. If this mean-stepsize results in a two large KL divergence, or in a non-positive-definite covariance
        matrix, it becomes the new lower bound; otherwise the new upper bound.

        Parameters:
            kl_bound: tf.float32
                The trust region constraint

            lower_bound: tf.float32
                The initial lower bound on the stepsize

            upper_bound:
                The initial upper bound on the stepsize

            old_lin_term: tf.Tensor
                The linear term of the canonical Gaussian form of the current component. A one-dimensional tensor
                of shape num_dimensions.

            old_precision:
                The precision matrix of the current component. A two-dimensional tensor of shape
                num_dimensions x num_dimensions.

            old_inv_chol:
                The inverse of the Cholesky matrix of the current component. A two-dimensional tensor of shape
                num_dimensions x num_dimensions.

            reward_lin:
                When using :py:class:`MORE` to estimate the natural gradient, this tensor correspond to the linear
                coefficient of the quadratic reward model. When using Stein's Lemma, this term can be computed
                from the expected gradient and expected Hessian.

            reward_quad:
                When using :py:class:`MORE` to estimate the natural gradient, this tensor correspond to the quadratic
                coefficient of the quadratic reward model. When using Stein's Lemma, this term can be computed
                from the expected Hessian.

            kl_const_part:
                A term of the KL divergence that can be precomputed as it does not depend on the parameters of the
                updated component.

            old_mean:
                The mean of the current component.

            eta_in_logspace:
                if true, the bracketing search should be performed in log-space (requires fewer iterations)

        Returns:
            tuple(tf.float32, tf.float32):

            **new_lower_bound** - The lower bound after a stopping criterion was reached.

            **new_upper_bound** - The upper bound after a stopping criterion was reached.

        """
        eta = 0.5 * (upper_bound + lower_bound)
        upper_bound_satisfies_constraint = False
        for _ in tf.range(1000):
            if eta_in_logspace:
                diff = tf.math.minimum(tf.exp(upper_bound) - tf.exp(eta), tf.exp(eta) - tf.exp(lower_bound))
            else:
                diff = tf.math.minimum(upper_bound - eta, eta - lower_bound)
            if diff < 1e-1:
                break

            kl = self.kl(eta, old_lin_term, old_precision, old_inv_chol, reward_lin_term, reward_quad_term,
                         kl_const_part, old_mean, eta_in_logspace)[0]

            if tf.abs(kl_bound - kl) < 1e-1 * kl_bound:
                # we indicate that we already found a sufficiently good eta, by setting:
                lower_bound = upper_bound = eta
                break

            if kl_bound > kl:
                upper_bound = eta
                upper_bound_satisfies_constraint = True
            else:
                lower_bound = eta
            eta = 0.5 * (upper_bound + lower_bound)
        # We could not find the optimal multiplier, but if the upper bound is large enough to satisfy the constraint,
        # we can still make an update
        if upper_bound_satisfies_constraint:
            lower_bound = upper_bound

        if eta_in_logspace:
            return tf.exp(lower_bound), tf.exp(upper_bound)
        else:
            return lower_bound, upper_bound

    def apply_NG_update(self, expected_hessians_neg: tf.Tensor, expected_gradients_neg: tf.Tensor,
                        stepsizes: tf.Tensor):
        means = tf.TensorArray(tf.float32, size=self.model.num_components)
        successes = tf.TensorArray(tf.bool, size=self.model.num_components)
        chols = tf.TensorArray(tf.float32, size=self.model.num_components)
        kls = tf.TensorArray(tf.float32, size=self.model.num_components)
        etas = tf.TensorArray(tf.float32, size=self.model.num_components)

        for i in range(self.model.num_components):
            old_chol = self.model.chol_cov[i]
            old_mean = self.model.means[i]
            last_eta = self.model.last_log_etas[i]
            eps = stepsizes[i]
            eta_offset = self.temperature

            reward_quad = expected_hessians_neg[i]
            if self.model.diagonal_covs:
                reward_lin = reward_quad * old_mean - expected_gradients_neg[i]
                old_logdet = 2 * tf.reduce_sum(tf.math.log(old_chol))
                old_inv_chol = 1./(old_chol)
                old_precision = old_inv_chol**2
                old_lin_term = old_precision * old_mean
                kl_const_part = old_logdet - self.model.num_dimensions
            else:
                reward_lin = tf.squeeze(reward_quad @ tf.expand_dims(old_mean, 1)) - expected_gradients_neg[i]
                old_logdet = 2 * tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(old_chol)))
                old_inv_chol = tf.linalg.inv(old_chol)
                old_precision = tf.transpose(old_inv_chol) @ old_inv_chol
                old_lin_term = tf.linalg.matvec(old_precision, old_mean)
                kl_const_part = old_logdet - self.model.num_dimensions

            if last_eta < 0:
                # No Warmstarting
                bracketing_search_in_logspace = True
                lower_bound = tf.constant(-20.)
                upper_bound = tf.constant(80.)
            else:
                # Warmstarting, define a bracket around the eta found during the last update.
                bracketing_search_in_logspace = True
                lower_bound = tf.maximum(0., tf.math.log(last_eta) - 3)
                upper_bound = tf.math.log(last_eta) + 3

            new_lower, new_upper = self.bracketing_search(eps, lower_bound, upper_bound, old_lin_term, old_precision,
                                                          old_inv_chol, reward_lin, reward_quad, kl_const_part,
                                                          old_mean, eta_in_logspace=bracketing_search_in_logspace)
            eta = tf.maximum(new_lower, eta_offset)
            success = False
            if new_lower == new_upper:
                success = True
                kl, new_mean, new_precision, new_inv_chol_inv = self.kl(eta, old_lin_term, old_precision, old_inv_chol,
                                                                        reward_lin, reward_quad, kl_const_part,
                                                                        old_mean, False)
                if self.model.diagonal_covs:
                    new_cov = tf.math.square(new_inv_chol_inv)
                else:
                    new_cov = tf.transpose(new_inv_chol_inv) @ new_inv_chol_inv

                if kl < tf.float32.max:
                    if self.model.diagonal_covs:
                        new_chol = tf.math.sqrt(new_cov)
                    else:
                        new_chol = tf.linalg.cholesky(new_cov)
                        if tf.reduce_any(tf.math.is_nan(new_chol)):
                            success = False
                else:
                    success = False
                    new_chol = old_chol

                if success:
                    chols = chols.write(i, new_chol)
                    means = means.write(i, new_mean)
                    kls = kls.write(i, kl)
                    successes = successes.write(i, True)
                    etas = etas.write(i, eta)

            if not success:
                chols = chols.write(i, old_chol)
                means = means.write(i, old_mean)
                kls = kls.write(i, -1.)
                successes = successes.write(i, False)
                etas = etas.write(i, -1)

        chols = chols.stack()
        means = means.stack()
        successes = successes.stack()
        etas = etas.stack()

        self.model.replace_components(means, chols)
        self.model.num_received_updates.assign_add(tf.ones(self.model.num_components))
        updated_l2_reg = tf.where(successes,
                                  tf.maximum(0.5 * self.model.l2_regularizers, self.model.initial_regularizer),
                                  tf.minimum(1e-6, 10 * self.model.l2_regularizers))
        self.model.l2_regularizers.assign(updated_l2_reg)
        self.model.last_log_etas.assign(etas)

    #    kls = kls.stack()
    #    tf.print(kls)
