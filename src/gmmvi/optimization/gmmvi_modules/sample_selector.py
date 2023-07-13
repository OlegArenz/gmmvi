import tensorflow as tf
from gmmvi.optimization.sample_db import SampleDB
from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.experiments.target_distributions.lnpdf import LNPDF

class SampleSelector:
    """ Provides the interface for selecting samples for performing the updates at the beginning of every iteration.

    The samples are evaluated on the target distribution and used for updating the weights, means and covariance of the
    GMM.

    There are currently two options for estimating the natural gradient:

    1. The :py:class:`VipsSampleSelector` use the procedure described by :cite:t:`Arenz2018,Arenz2020` to ensure that we
       have samples in the vicinity of every component, enabling us to perform a stable update on every component.

    2. The :py:class:`LinSampleSelector` uses the procedure described by :cite:t:`Lin2019` which draws samples
        according to the weights of the current mixture model, aiming for better sample efficiency.

    Parameters:
        target_distribution: :py:class:`LNPDF`
            The target distribution is used for evaluating the newly drawn samples.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model is used for drawing the samples.

        sample_db: :py:class:`SampleDB`
            The new samples and their target_densities (and gradients) are stored in the sample database.
    """

    def __init__(self, target_distribution: LNPDF, model: GmmWrapper, sample_db: SampleDB):
        self.target_distribution = target_distribution
        self.model = model
        self.sample_db = sample_db
        if self.target_distribution.safe_for_tf_graph:
            self.get_target_grads = tf.function(self.get_target_grads, experimental_relax_shapes=True)

    @staticmethod
    def build_from_config(config, gmm_wrapper, sample_db, target_distribution):
        """This static method provides a convenient way to create a
        :py:class:`VipsSampleSelector`, or :py:class:`LinSampleSelector` depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model is used for drawing the samples.

            sample_db: :py:class:`SampleDB`
                The new samples and their target_densities (and gradients) are stored in the sample database.

            target_distribution: :py:class:`LNPDF`
                The target distribution is used for evaluating the newly drawn samples.
        """
        if config["sample_selector_type"] == "component-based":
            return VipsSampleSelector(target_distribution, gmm_wrapper, sample_db,
                                                 **config['sample_selector_config'])
        elif config["sample_selector_type"] == "mixture-based":
            return LinSampleSelector(target_distribution, gmm_wrapper, sample_db,
                                                **config['sample_selector_config'])
        else:
            raise ValueError(
                f"config['sample_selector_type'] is '{config['sample_selector_type']}' which is an unknown type")

    def target_uld(self, samples: tf.Tensor) -> tf.Tensor:
        return self.target_distribution.log_density(samples)

    def get_target_grads(self, samples: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        if self.target_distribution.use_log_density_and_grad:
            # useful if we can't backprop through target_uld
            target, gradient = self.target_distribution.log_density_and_grad(samples)
        else:
            with tf.GradientTape(persistent=False) as tape:
                tape.watch(samples)
                target = self.target_distribution.log_density(samples)
            gradient = tape.gradient(target, samples)
        return gradient, target

    def select_samples(self) -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Select the samples for current learning iteration and stores the data in the sample database.

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):

            **samples** - a tensor of shape number_of_selected_samples x number_of_dimensions

            **old_samples_pdf** - a tensor of shape number_of_selected_samples, containing the log-densities
            of the distribution that was effectively used to obtain the selected samples.
            Needed for importance weighting.

            **target_lnpdfs** - a tensor of shape number_of_selected_samples, containing the log-densities of the
            target distrbution for each selected sample, :math:`\\log p(\\mathbf{x})`.

            **target_grads** - a tensor  of shape number_of_selected_samples x num_dimensions, containing the gradients
            of the log-densities of the target distrbution for each selected sample,
            :math:`\\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})`.

        """
        raise NotImplementedError

class VipsSampleSelector(SampleSelector):
    """ Selects the samples according to the procedure described by :cite:t:`Arenz2018,Arenz2020`.

    This class uses the procedure described by :cite:t:`Arenz2018,Arenz2020` to ensure that we have samples in the
    vicinity of every component. It uses two passes. In the first pass, it selects a given number of samples from the
    sample database. In the second pass, it computes the effective sample size for every component (based on the
    importance weights) and compares the effective sample size with a given desired number of samples. It then draws
    from every component the respective missing number of samples.


    Parameters:
        target_distribution: :py:class:`LNPDF`
            The target distribution is used for evaluating the newly drawn samples.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model is used for drawing the samples.

        sample_db: :py:class:`SampleDB`
            The database is used for reusing samples from previous iterations and for storing the new samples and their
            target_densities (and gradients).

        desired_samples_per_component: int
            The desired number of samples for every component.

        ratio_reused_samples_to_desired: float
            In the first pass, we reuse the number_of_components * *ratio_reused_samples_to_desired* *
            *desired_samples_per_component* freshest samples from the database.
    """

    def __init__(self, target_distribution: LNPDF, model: GmmWrapper, sample_db: SampleDB,
                 desired_samples_per_component: int, ratio_reused_samples_to_desired: float):
        super(VipsSampleSelector, self).__init__(target_distribution, model, sample_db)
        self.sample_db = sample_db
        self.desired_samples_per_component = desired_samples_per_component
        self.reused_samples_per_component = tf.cast(tf.math.floor(
            ratio_reused_samples_to_desired * desired_samples_per_component), dtype=tf.int32)

    def get_effective_samples(self, model_densities: tf.Tensor, oldsamples_pdf: tf.Tensor) -> tf.Tensor:
        """ Computes the effective sample size based on the log-densities of the target distribution and the
        log-densities of the background distribution.

        Parameters:
            model_densities: tf.Tensor
                The log-densities of the individual components, :math:`\\log q(\\mathbf{x}|o)`

            oldsamples_pdf: tf.Tensor
                The log-densities of the distribution that was effectively used for obtaining the selected samples

        Returns:
            float: the effective number of samples
        """
        log_weight = model_densities - tf.expand_dims(oldsamples_pdf, axis=0)
        log_weight = log_weight - tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)
        weights = tf.exp(log_weight)
        num_effective_samples = 1. / tf.reduce_sum(weights * weights, axis=1)
        return num_effective_samples

    def sample_where_needed(self, samples: tf.Tensor, oldsamples_pdf: tf.Tensor, num_desired_samples: int = None)\
            -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Computes the components' effective sample sizes for the given set of samples and draws, for every component i,
        :math:`n_{\\text{des}} - n_{\\text{eff,i}}` new samples.

        Parameters:
            samples: tf.Tensor
                the samples that were chosen during the first pass

            oldsamples_pdf: tf.Tensor
                The log-densities of the distribution that was effectively used for obtaining the selected samples

            num_desired_samples: int
                The number of desired samples per component

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):

            **new_samples** - a tf.Tensor, the newly drawn samples

            **new_target_lnpdfs** - a tf.Tensor, the log-densities of the target distribution on the newly drawn
            samples, :math:`\\log p(\\mathbf{x})`.

            **new_target_grads** - a tf.Tensor, the gradients of  the log-densities for the newly drawn samples,
            :math:`\\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})`.

            **mapping** - a tf.Tensor, for every sample the one-dimensional tensor contains the index of the component
            that was used for drawing that sample.
        """

        if num_desired_samples is None:
            num_desired_samples = self.desired_samples_per_component
        if tf.shape(samples)[0] == 0:
            num_effective_samples = tf.zeros((self.model.num_components), dtype=tf.int32)
        else:
            model_logpdfs = self.model.component_log_densities(samples)
            num_effective_samples = tf.cast(tf.math.floor(self.get_effective_samples(model_logpdfs, oldsamples_pdf)),
                                            dtype=tf.int32)
        num_additional_samples = tf.math.maximum(1, num_desired_samples - num_effective_samples)
        new_samples, mapping = self.model.sample_from_components_no_shuffle(num_additional_samples)
        new_target_grads, new_target_lnpdfs = self.get_target_grads(new_samples)
        return new_samples, new_target_lnpdfs, new_target_grads, mapping

    def select_samples(self) -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get old samples from the database
        num_samples_to_reuse = self.reused_samples_per_component * self.model.num_components
        oldsamples_pdf, samples, _, _, _ = self.sample_db.get_newest_samples(num_samples_to_reuse)
        num_reused_samples = tf.shape(samples)[0]

        # Get additional samples to ensure a desired effective sample size for every component
        new_samples, new_target_lnpdfs, new_target_grads, mapping = self.sample_where_needed(samples, oldsamples_pdf)
        self.sample_db.add_samples(new_samples, self.model.means, self.model.chol_cov, new_target_lnpdfs,
                                   new_target_grads, mapping)
        num_new_samples = tf.shape(new_samples)[0]

        # We call get_newest_samples again in order to recompute the background densities
        oldsamples_pdf, samples, mapping, target_lnpdfs, target_grads = self.sample_db.get_newest_samples(num_reused_samples +
                                                                                                 num_new_samples)
        return samples, mapping, oldsamples_pdf, target_lnpdfs, target_grads

class LinSampleSelector(SampleSelector):
    """ Selects the samples according to the procedure described by :cite:t:`Lin2019`.

    This class uses the procedure described by :cite:t:`Lin2019` by drawing new samples for the current mixture
    model. We also implemented the two-phase procedure of the :py:class:`VipsSampleSelector` to reuse samples from
    the database and redraw samples based on a desired number of samples. However, in contrast to the
    :py:class:`VipsSampleSelector`, we compute the effective sample size not per component, but for the whole mixture,
    and redraw samples n_eff - *desired_samples_per_component* new samples from the mixture model.
    The exact procedure of :cite:t:`Lin2019` can be reproduced, when choosing *ratio_reused_samples_to_desired* = 0,
    where always a fixed number of new samples is drawn from the mixture model.

    Parameters:
        target_distribution: :py:class:`LNPDF`
            The target distribution is used for evaluating the newly drawn samples.

        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model is used for drawing the samples.

        sample_db: :py:class:`SampleDB`
            The database is used for reusing samples from previous iterations and for storing the new samples and their
            target_densities (and gradients).

        desired_samples_per_component: int
            The desired number for the mixture update.

        ratio_reused_samples_to_desired: float
            In the first pass, we reuse the *ratio_reused_samples_to_desired* *
            *desired_samples_per_component* freshest samples from the database.
    """

    def __init__(self, target_distribution: LNPDF, model: GmmWrapper, sample_db: SampleDB,
                 desired_samples_per_component: int, ratio_reused_samples_to_desired: float):
        super(LinSampleSelector, self).__init__(target_distribution, model, sample_db)
        self.desired_samples_per_component = desired_samples_per_component
        self.reused_samples_per_component = tf.cast(tf.math.floor(
            ratio_reused_samples_to_desired * desired_samples_per_component), dtype=tf.int32)

    def get_effective_samples(self, model_densities: tf.Tensor, oldsamples_pdf: tf.Tensor) -> tf.Tensor:
        """
        Computes the effective sample size of the mixture model based on the log-densities of the target distribution
        and the log-densities of the background distribution.

        Parameters:
            model_densities: tf.Tensor
                The log-densities of the mixture model, :math:`\\log q(\\mathbf{x})`.

            oldsamples_pdf: tf.Tensor
                The log-densities of the distribution that was effectively used for obtaining the selected samples

        Returns:
            float: the effective number of samples
        """
        log_weight = model_densities - tf.expand_dims(oldsamples_pdf, axis=0)
        log_weight = log_weight - tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)
        weights = tf.exp(log_weight)
        num_effective_samples = 1. / tf.reduce_sum(weights * weights, axis=1)
        return num_effective_samples

    @tf.function
    def sample_where_needed(self) -> [tf.Tensor, tf.Tensor, int]:
        """
        Computes the mixture model's effective sample size for the given set of samples and draws
        :math:`n_{\\text{des}} - n_{\\text{eff}}` new samples from the mixture model.

        Parameters:
            samples: tf.Tensor
                the samples that were chosen during the first pass

            oldsamples_pdf: tf.Tensor
                The log-densities of the distribution that was effectively used for obtaining the selected samples

            num_desired_samples: int
                The number of desired samples per component

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):

            **new_samples** - a tensor containing the newly drawn samples

            **new_target_lnpdfs** - a tensor containing the log-densities of the target distribution on the
            newly drawn samples, :math:`\\log p(\\mathbf{x})`.

            **new_target_grads** - a tensor containing the gradients of  the log-densities for the newly drawn
            samples.

            **mapping** - a tensor containing for every sample the one-dimensional tensor contains the index of the
            component that was used for drawing that sample.
        """

        # Get old samples from the database
        num_samples_to_reuse = self.reused_samples_per_component * self.model.num_components
        oldsamples_pdf, old_samples, _, _, _ = self.sample_db.get_newest_samples(num_samples_to_reuse)
        num_reused_samples   = tf.shape(old_samples)[0]

        # Get additional samples to ensure a desired effective sample size for every component
        if tf.shape(old_samples)[0] == 0:
            num_effective_samples = tf.zeros((1), dtype=tf.int32)
        else:
            model_logpdfs = self.model.log_density(old_samples)
            num_effective_samples = tf.cast(tf.math.floor(self.get_effective_samples(model_logpdfs, oldsamples_pdf)),
                                            dtype=tf.int32)
        num_additional_samples = tf.math.maximum(1, self.desired_samples_per_component - num_effective_samples)
        new_samples, mapping = self.model.sample(tf.squeeze(num_additional_samples))

        return new_samples, mapping, num_reused_samples

    def select_samples(self) -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get additional samples to ensure a desired effective sample size for every component
        new_samples, mapping, num_reused_samples = self.sample_where_needed()

        new_target_grads, new_target_lnpdfs = self.get_target_grads(new_samples)

        self.sample_db.add_samples(new_samples, self.model.means, self.model.chol_cov, new_target_lnpdfs,
                                   new_target_grads, mapping)

        # We call get_newest_samples again in order to recompute the background densities
        samples_this_iter = num_reused_samples + tf.shape(new_samples)[0]
        oldsamples_pdf, samples, mapping, target_lnpdfs, target_grads = self.sample_db.get_newest_samples(samples_this_iter)
        return samples, mapping, oldsamples_pdf, target_lnpdfs, target_grads

