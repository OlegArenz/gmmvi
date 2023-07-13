# -*- coding: utf-8 -*-
import tensorflow as tf

from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.experiments.target_distributions.lnpdf import LNPDF
from gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation import ComponentStepsizeAdaptation
from gmmvi.optimization.gmmvi_modules.component_adaptation import ComponentAdaptation
from gmmvi.optimization.gmmvi_modules.ng_based_component_updater import NgBasedComponentUpdater
from gmmvi.optimization.gmmvi_modules.ng_estimator import NgEstimator
from gmmvi.optimization.gmmvi_modules.sample_selector import SampleSelector
from gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation import WeightStepsizeAdaptation
from gmmvi.optimization.gmmvi_modules.weight_updater import WeightUpdater
from gmmvi.optimization.sample_db import SampleDB


class GMMVI:
    """The main class of this framework, which provides the functionality to perform a complete update step for the GMM.

    Responsibilities for performing the necessary sub-steps (sample selection, natural gradient estimation, etc.)
    and for keeping track of data are delegated to the :py:mod:`GMMVI Modules<gmmvi.optimization.gmmvi_modules>`,
    the :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>` and
    :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`.
    Hence, this class acts mainly as a manager between these components.

    Parameters:
        model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The (wrapped) model that we are optimizing.

        sample_db: :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>`
            The database for storing samples.

        temperature: tf.float32
            The temperature parameter :math:`\\beta` for weighting the model entropy :math:`H(q)`
            in the optimization problem
            :math:`\\arg\\max_q \\mathbb{E}\\left[ \\log(\\tilde{p}(x)) \\right] + \\beta H(q)`.

        sample_selector: :py:class:`SampleSelector<gmmvi.optimization.gmmvi_modules.sample_selector.SampleSelector>`
            The SampleSelector for selecting the samples that are used during each iteration.

        num_component_adapter: :py:class:`NumComponentAdaptation\
                <gmmvi.optimization.gmmvi_modules.component_adaptation.ComponentAdaptation>`
            The NumComponentAdapter used for adding and deleting components.

        component_stepsize_adapter: :py:class:`ComponentStepsizeAdaptation\
                <gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation.ComponentStepsizeAdaptation>`
            The ComponentStepsizeAdapter for choosing the learning rates for the component update.

        ng_estimator: :py:class:`NgEstimator<gmmvi.optimization.gmmvi_modules.ng_estimator.NgEstimator>`
            The NgEstimator for estimating the natural gradient for the component update.

        ng_based_updater: :py:class:`NgBasedComponentUpdater\
                <gmmvi.optimization.gmmvi_modules.ng_based_component_updater.NgBasedComponentUpdater>`
            The NgBasedComponentUpdater for updating the components based on the estimated natural gradients.

        weight_stepsize_adapter: :py:class:`WeightStepsizeAdaptation\
                <gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation.WeightStepsizeAdaptation>`
            The WeightStepsizeAdapter for choosing the learning rate for updating the mixture weights.

        weight_updater: :py:class:`WeightUpdater<gmmvi.optimization.gmmvi_modules.weight_updater.WeightUpdater>`
            The NgBasedComponentUpdater for updating the components based on the estimated natural gradients.
    """

    def __init__(
            self,
            model: GmmWrapper,
            sample_db: SampleDB,
            temperature: tf.float32,
            sample_selector: SampleSelector,
            num_component_adapter: ComponentAdaptation,
            component_stepsize_adapter: ComponentStepsizeAdaptation,
            ng_estimator: NgEstimator,
            ng_based_updater: NgBasedComponentUpdater,
            weight_stepsize_adapter: WeightStepsizeAdaptation,
            weight_updater: WeightUpdater
    ):
        self.temperature = temperature
        self.model = model
        self.num_dimensions = self.model.num_dimensions
        self.sample_db = sample_db
        self.sample_selector = sample_selector
        self.num_component_adapter = num_component_adapter
        self.component_stepsize_adapter = component_stepsize_adapter
        self.ng_estimator = ng_estimator
        self.ng_based_updater = ng_based_updater
        self.weight_stepsize_adapter = weight_stepsize_adapter
        self.weight_updater = weight_updater
        self.num_updates = tf.Variable(0, dtype=tf.int32)

        if self.sample_selector.target_distribution.safe_for_tf_graph:
            # it is fine to query the target_distribution within a tf.function(),
            # so we can compile the whole training-step to a graph
            self.train_iter = tf.function(self.train_iter, experimental_follow_type_hints=True)
        else:
            # Everything not related to querying the target_distribution will be wrapped in a graph.
            # Needed when the target_distribution is not implemented in tensorflow
            self._run_updates = tf.function(
                self._run_updates, input_signature=[
                    tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # samples
                    tf.TensorSpec(shape=[None], dtype=tf.int32),  # mapping
                    tf.TensorSpec(shape=[None], dtype=tf.float32),  # sample_dist_densities
                    tf.TensorSpec(shape=[None], dtype=tf.float32),  # target_lnpdfs
                    tf.TensorSpec(shape=[None, None], dtype=tf.float32)]  # target_lnpdf_grads
            )

    @staticmethod
    def build_from_config(config: dict, target_distribution: LNPDF, model: GmmWrapper):
        """Create a :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>` instance from a configuration dictionary.

        This static method provides a convenient way to create a :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>`
        instance, based on an initial GMM ( a :py:class:`wrapped model<gmmvi.models.gmm_wrapper.GmmWrapper>` ),
        a :py:class:`target_distribution<gmmvi.experiments.target_distributions.lnpdf.LNPDF>` and a dictionary
        containing the types and parameters of the :py:mod:`GMMVI modules<gmmvi.optimization.gmmvi_modules>`.

        Parameters:
            config: dict
                The dictionary should contain for each :py:mod:`GMMVI module<gmmvi.optimization.gmmvi_modules>`
                an entry of the form XXX_type (a string) and XXX_config (a dict) for specifying the type of each module,
                and the module-specific hyperparameters.
                For example, the dictionary could contain sample_selector_type={"component-based"} and
                sample_selector_config={"desired_samples_per_component": 100, "ratio_reused_samples_to_desired": 2.}.
                Refer to the example yml-configs, or to the individual GMMVI module for the expected parameters, and
                type-strings.

            target_distribution: :py:class:`LNPDF<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
                The (unnormalized) target distribution that we want to approximate.

            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The (wrapped) model that we are optimizing.
        """
        sample_db = SampleDB.build_from_config(config, model.num_dimensions)
        ng_estimator = NgEstimator.build_from_config(config, config['temperature'], model)
        ng_based_updater = NgBasedComponentUpdater.build_from_config(config, model)
        num_component_adapter = ComponentAdaptation.build_from_config(
            config, model, sample_db, target_distribution=target_distribution,
            prior_mean=config["model_initialization"]["prior_mean"],
            initial_cov=config["model_initialization"]["initial_cov"])
        component_stepsize_adapter = ComponentStepsizeAdaptation.build_from_config(config, model)
        sample_selector = SampleSelector.build_from_config(config, model, sample_db, target_distribution)
        weight_updater = WeightUpdater.build_from_config(config, model)
        weight_stepsize_adapter = WeightStepsizeAdaptation.build_from_config(config, model)

        return GMMVI(model, sample_db, config['temperature'], sample_selector,
                     num_component_adapter, component_stepsize_adapter,
                     ng_estimator, ng_based_updater, weight_stepsize_adapter, weight_updater)

    def train_iter(self):
        """Perform a single training iteration.

        This method does not take any parameters, nor does it return anything.
        However, it may have several effects, such as

        * drawing new samples from the :py:attr:`~model` and evaluating them on the target distribution,

        * updating the :py:attr:`gmmvi.optimization.gmmvi.GMMVI.model` parameters,

        * adapting learning rates, etc.
        """
        samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads \
            = self.sample_selector.select_samples()
        self._run_updates(samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads)
        self.num_component_adapter.adapt_number_of_components(self.num_updates)

    def _run_updates(self, samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads):
        # Update components
        new_component_stepsizes = self.component_stepsize_adapter.update_stepsize(self.model.stepsizes)
        self.model.update_stepsizes(new_component_stepsizes)
        expected_hessian_neg, expected_grad_neg = self.ng_estimator.get_expected_hessian_and_grad(
            samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads)
        self.ng_based_updater.apply_NG_update(expected_hessian_neg, expected_grad_neg, self.model.stepsizes)

        # update weights
        weight_stepsize = self.weight_stepsize_adapter.update_stepsize()
        self.weight_updater.update_weights(samples, sample_dist_densities, target_lnpdfs, weight_stepsize)
        self.num_updates.assign_add(1)

