from math import pi
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp

from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.optimization.sample_db import SampleDB
from gmmvi.models.diagonal_gmm import DiagonalGMM
from gmmvi.experiments.target_distributions.lnpdf import LNPDF

class ComponentAdaptation:
    """ This class provides a common interface for adapting the number of components.

    There are currently only two options:

    1. The :py:class:`FixedComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.FixedComponentAdaptation>`
    a dummy-class, that does not do anything.

    2. The :py:class:`VipsComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.VipsComponentAdaptation>`
    uses the procedure of VIPS :cite:p:`Arenz2020` to add and delete components.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        sample_db: :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>`
            The sample database can be used to select candidate locations for adding a new component, without having
            to perform additional queries to the target distribution.

        target_distribution: :py:class:`LNPDF<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
             The target distribution can be used to evaluate candidate locations for adding a new component.

        prior_mean: tf.Tensor
            A one dimensional tensor of size num_dimensions, specifying the mean of the Gaussian that we can use to
            sample candidate locations for adding a new component.

        initial_cov: tf.Tensor
            A two-dimensional tensor of size num_dimensions x num_dimensions, specifying the covariance of the Gaussian
            that we can use to sample candidate locations for adding a new component.
    """
    def __init__(self):
        pass

    @staticmethod
    def build_from_config(config, gmm_wrapper, sample_db, target_distribution, prior_mean, initial_cov):
        """This static method provides a convenient way to create a
        :py:class:`FixedComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.FixedComponentAdaptation>`
        or :py:class:`VipsComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.VipsComponentAdaptation>`
        depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model where we want to adapt the number of components.

            sample_db: :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>`
                The sample database can be used to select candidate locations for adding a new component, without having
                to perform additional queries to the target distribution.

            target_distribution: :py:class:`LNPDF<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
                 The target distribution can be used to evaluate candidate locations for adding a new component.

            prior_mean: tf.Tensor
                A one dimensional tensor of size num_dimensions, specifying the mean of the Gaussian that we can use to sample
                candidate locations for adding a new component.

            initial_cov: tf.Tensor
                A two-dimensional tensor of size num_dimensions x num_dimensions, specifying the covariance of the Gaussian
                that we can use to sample candidate locations for adding a new component.
        """
        if config["num_component_adapter_type"] == "adaptive":
            return VipsComponentAdaptation(gmm_wrapper, sample_db, target_distribution, prior_mean, initial_cov,
                                           **config["num_component_adapter_config"])
        elif config["num_component_adapter_type"] == "fixed":
            return FixedComponentAdaptation(**config["num_component_adapter_config"])
        else:
            raise ValueError(
                f"config['num_component_adapter_type'] is '{config['num_componenter_adapter_type']}' "
                f"which is an unknown type")

    def adapt_number_of_components(self, iteration):
        raise NotImplementedError


class FixedComponentAdaptation(ComponentAdaptation):
    """ This is a dummy class, used when we do not want to adapt the number of components during learning. """
    def __init__(self):
        super(FixedComponentAdaptation, self).__init__()

    def adapt_number_of_components(self, iteration):
        """ As we do not want to change the number of components, this method does not do anything.

        Parameters:

             iteration: int
                The current iteration (ignored).
        """
        pass


class VipsComponentAdaptation(ComponentAdaptation):
    """ This class implements the component adaptation procedure used by VIPS.

    See :cite:p:`Arenz2020`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        sample_db: :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>`
            The sample database can be used to select candidate locations for adding a new component, without having
            to perform additional queries to the target distribution.

        target_distribution: :py:class:`LNPDF<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
             The target distribution can be used to evaluate candidate locations for adding a new component.

        prior_mean: tf.Tensor
            A one dimensional tensor of size num_dimensions, specifying the mean of the Gaussian that we can use to sample
            candidate locations for adding a new component.

        initial_cov: tf.Tensor
            A two-dimensional tensor of size num_dimensions x num_dimensions, specifying the covariance of the Gaussian
            that we can use to sample candidate locations for adding a new component.

        del_iters: int
            minimum number of updates a component needs to have received, before it is considered as candidate for deletion.

        add_iters: int
            a new component will be added every *add_iters* iterations

        max_components: int
            do not add components, if the model has at least *max_components* components

        num_database_samples: int
            number of samples from the :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>` that are used for
            selecting a good initial mean when adding a new component.

        num_prior_samples: int
            number of samples from the prior distribution that are used for selecting a good initial mean when adding a
            new component.
    """
    def __init__(self, model: GmmWrapper, sample_db: SampleDB, target_lnpdf: LNPDF, prior_mean: Union[float, tf.Tensor],
                 initial_cov: Union[float, tf.Tensor], del_iters: int, add_iters: int, max_components: int,
                 thresholds_for_add_heuristic: float, min_weight_for_del_heuristic: float,
                 num_database_samples: int, num_prior_samples: int):
        super(VipsComponentAdaptation, self).__init__()
        self.model = model
        if (prior_mean is not None) and (initial_cov is not None):
            if tf.rank(initial_cov) == 0:
                initial_cov = initial_cov * tf.ones(model.num_dimensions)
            if tf.rank(prior_mean) == 0:
                prior_mean = prior_mean * tf.ones(model.num_dimensions)
            self.prior = DiagonalGMM(tf.ones(1), tf.expand_dims(prior_mean, 0), tf.expand_dims(initial_cov, 0))
        else:
            self.prior = None

        self.num_prior_samples = num_prior_samples
        self.target_lnpdf = target_lnpdf
        self.sample_db = sample_db
        self.del_iters = del_iters
        self.add_iters = add_iters
        self.max_components = max_components
        self.num_db_samples = num_database_samples
        self.num_calls_to_add_heuristic = tf.Variable(0)
        self.thresholds_for_addHeuristic = tf.convert_to_tensor(thresholds_for_add_heuristic, dtype=tf.float32)
        self.min_weight_for_del_heuristic = min_weight_for_del_heuristic

        self.reward_improvements = tf.Variable(tf.zeros(0), shape=[None], dtype=tf.float32)
        self.filter_delay = tf.cast(tf.floor(self.del_iters / 3), dtype=tf.int32)
        gaussian = tfp.distributions.Normal(tf.constant(0, tf.float32), tf.constant(self.del_iters / 8., tf.float32))
        kernel = gaussian.prob(tf.range(start=-self.filter_delay, limit=self.filter_delay, dtype=tf.float32))
        self.kernel = tf.reshape(kernel / tf.reduce_sum(kernel), [-1, 1, 1])

    def adapt_number_of_components(self, iteration: int):
        """ This method may change the number of components, either by deleting bad components that have low weights,
         or by adding new components.

         Parameters:
             iteration: int
                The current iteration, used to decide whether a new component should be added.
         """

        if iteration > self.del_iters:
            self.delete_bad_components()
        if iteration > 1 and iteration % self.add_iters == 0:
            if self.model.num_components < self.max_components:
                self.add_new_component()

    @tf.function(experimental_relax_shapes=True)
    def add_at_best_location(self, samples, target_lnpdfs):
        """ Find the most promising :cite:p:`Arenz2020` location among the provided samples for adding a new component,
        that is, a new component will be added with mean given by one of the provided samples.

        Parameters:
            samples: tf.Tensor
                candidate locations for initializing the mean of the new component

            target_lnpdfs: tf.Tensor
                for each candidate location, this tensor contains the log-density under the (unnormalized) target
                distribution.
        """
        iter = self.num_calls_to_add_heuristic % len(self.thresholds_for_addHeuristic)
        model_log_densities = self.model.log_density(samples)
        init_weight = 1e-29
        a = tf.random.uniform([1])
        if self.prior is not None:
            des_entropy = self.model.get_average_entropy() * a + self.prior.get_average_entropy() * (1 - a)
        else:
            des_entropy = self.model.get_average_entropy()
        max_logdensity = tf.reduce_max(model_log_densities)
        rewards = target_lnpdfs - tf.maximum(
            max_logdensity - self.thresholds_for_addHeuristic[iter], model_log_densities
        )
        new_mean = samples[tf.argmax(rewards)]
        H_unscaled = 0.5 * self.model.num_dimensions * (tf.math.log(2.0 * pi) + 1)
        c = tf.math.exp((2 * (des_entropy - H_unscaled)) / self.model.num_dimensions)
        if self.model.diagonal_covs:
            new_cov = c * tf.ones(self.model.num_dimensions)
        else:
            new_cov = c * tf.eye(self.model.num_dimensions)

        self.model.add_component(init_weight, new_mean, new_cov, tf.reshape(self.thresholds_for_addHeuristic[iter], [1]),
                                 tf.reshape(des_entropy, [1]))

    @tf.function
    def select_samples_for_adding_heuristic(self):
        """ Select a set of samples used as candidates for initializing the mean of the new component.

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor):

            **samples** - the selected candidate locations

            **target_lnpdfs** - log-densities of the *samples* under the unnormalized target distribution

            **prior_samples** - additional samples drawn from a prior, which have not yet been evaluated on the
            target distribution.
        """
        self.num_calls_to_add_heuristic.assign_add(1)
        samples, target_lnpdfs = self.sample_db.get_random_sample(self.num_db_samples)
        prior_samples = tf.zeros((0, self.model.num_dimensions), tf.float32)

        if self.num_prior_samples > 0:
            prior_samples = self.prior.sample(self.num_prior_samples)[0]
            self.sample_db.num_samples_written.assign_add(self.num_prior_samples)
        return samples, target_lnpdfs, prior_samples

    def add_new_component(self):
        """ This method adds a new component by first selecting a set of candidate locations and the choosing the most
        promising one using the procedure of VIPS :cite:p:`Arenz2020`.
        """
        samples, target_lnpdfs, prior_samples = self.select_samples_for_adding_heuristic()
        if self.num_prior_samples > 0:
            samples = tf.concat((samples, prior_samples), 0)
            target_lnpdfs = tf.concat((target_lnpdfs, self.target_lnpdf.log_density(prior_samples)), 0)
        self.add_at_best_location(samples, target_lnpdfs)

    def delete_bad_components(self):
        """ Components are deleted, if all the following criteria are met received:

        1. It must have received at least *del_iters* updates

        2. It must not have improved significantly during the last iterations. In contrast to VIPS, we use a Gaussian
           filter to smooth the rewards of the component, to be more robust with respect to noisy target distributions.

        3. It must have very low weight, such that the effects on the model are negligible.
        """

        # estimate the relative improvement for every component with respect to
        # the improvement it would need to catch up (assuming linear improvement) with the best component
        current_smoothed_reward = tf.reduce_mean(
            self.model.reward_history[:, -tf.size(self.kernel):] * tf.reshape(self.kernel, [1, -1]), axis=1)
        old_smoothed_reward = tf.reduce_mean(
            self.model.reward_history[:, -tf.size(self.kernel)-self.del_iters:-self.del_iters]
            * tf.reshape(self.kernel, [1, -1]), axis=1)

        old_smoothed_reward -= tf.reduce_max(current_smoothed_reward)
        current_smoothed_reward -= tf.reduce_max(current_smoothed_reward)
        reward_improvements = (current_smoothed_reward - old_smoothed_reward) / tf.abs(old_smoothed_reward)
        self.reward_improvements.assign(reward_improvements)
        # compute for each component the maximum weight it had within the last del_iters,
        # or that it would have gotten when we used greedy updates
        max_actual_weights = tf.reduce_max(self.model.weight_history[:, -tf.size(self.kernel)-self.del_iters:-1], axis=1)
        max_greedy_weights = tf.reduce_max(tf.exp(
            self.model.reward_history[:, -tf.size(self.kernel)-self.del_iters:] - tf.math.reduce_logsumexp(self.model.reward_history[:, -tf.size(self.kernel)-self.del_iters:],
                                                                             axis=0, keepdims=True)), axis=1)
        max_weights = tf.math.maximum(max_actual_weights, max_greedy_weights)

        is_stagnating = reward_improvements <= 0.4
        is_low_weight = max_weights < self.min_weight_for_del_heuristic
        is_old_enough = self.model.reward_history[:, -self.del_iters] != -tf.float32.max
        is_bad = tf.reduce_all((is_stagnating, is_low_weight, is_old_enough), axis=0)
        bad_component_indices = tf.squeeze(tf.where(is_bad), axis=1)

        if tf.size(bad_component_indices) > 0:
            for idx in tf.sort(bad_component_indices, direction='DESCENDING'):
                self.model.remove_component(idx)


