import tensorflow as tf
from gmmvi.models.gmm import GMM

class GmmWrapper():
    """ This method wraps around the :py:class:`model<gmmvi.models.gmm.GMM>` to keep track of component-specific
    meta-information used by the learner (e.g. component-specific stepsizes).
    This class can be used just like a :py:class:`model<gmmvi.models.gmm.GMM>`, because any methods not implemented
    within the GmmWrapper are forwarded to the encapuslated model. However, some functions have slightly different
    behavior, for example, when removing a component, not only the component in the encapuslated model will be removed,
    but also the meta-information, stored in this GmmWrapper. Hence, the model should always be accessed through the
    GmmWrapper.

    Whenever adding a new component (via py:meth:`gmmvi.models.gmm.GmmWrapper.add_component`), the GmmWrapper will
    initialize the meta-information (stepsize and l2-regularizer) with the provided initial_values.

    Parameters:
        model: :py:class:`gmmvi.models.gmm.GMM`
            The model to be encapuslated.

        initial_stepsize: float
            The stepsize, that is assigned to a newly added component

        initial_regularizer: float
            The l2 regularizer, that is assigned to a newly added component (only used when using a
            py:class:`MoreNgEstimator<gmmvi.optimization.gmmvi_modules.ng_estimator.MoreNgEstimator>` for estimating the
            natural gradients).

        max_reward_history_length: int
            The GmmWrapper also keeps track how much reward each component obtained at the previous iterations. This
            parameter controls after how many iterations the component rewards are forgotten (to save memory).
    """

    @staticmethod
    def build_from_config(model: GMM, config: dict):
        """Create a :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>` instance from a configuration dictionary.

        This static method provides a convenient way to create a
        :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>` instance, based on an initial
        :py:class:`GMM<gmmvi.models.gmm.GMM>`, and a dictionary containing the parameters.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.
                The max_reward_history_length, which is needed for instantiating the
                :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>` is typically not provided directly,
                but chosen depending on whether
                :py:class:`VipsComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.VipsComponentAdaptation>`
                is used or not.

            model: :py:class:`GMM<gmmvi.models.gmm.GMM>`
                The model that we want to encapsulate.
        """
        max_reward_history_length = 2 * max(2, config["num_component_adapter_config"]["del_iters"]) \
            if "del_iters" in config["num_component_adapter_config"] else 2
        initial_regularizer = config["ng_estimator_config"]["initial_l2_regularizer"] \
            if "initial_l2_regularizer" in config["ng_estimator_config"] else 1e-12
        initial_stepsize = config["component_stepsize_adapter_config"]["initial_stepsize"]
        return GmmWrapper(model, initial_stepsize, initial_regularizer, max_reward_history_length)

    def __init__(self, model: GMM, initial_stepsize: float, initial_regularizer: float, max_reward_history_length: int):
        self.model = model

        self.initial_regularizer = initial_regularizer
        self.initial_last_eta = -1
        self.initial_stepsize = initial_stepsize
        self.max_reward_history_length = max_reward_history_length

        self.l2_regularizers = tf.Variable(self.initial_regularizer * tf.ones(model.num_components), shape=[None])
        self.last_log_etas = tf.Variable(self.initial_last_eta * tf.ones(model.num_components), shape=[None])
        self.num_received_updates = tf.Variable(tf.zeros(model.num_components), shape=[None])
        self.stepsizes = tf.Variable(self.initial_stepsize * tf.ones(model.num_components), shape=[None])
        self.reward_history = tf.Variable(tf.float32.min * tf.ones((model.num_components, max_reward_history_length)),
                                          shape=[None, None])
        self.weight_history = tf.Variable(tf.float32.min * tf.ones((model.num_components, max_reward_history_length)),
                                          shape=[None, None])
        self.unique_component_ids = tf.Variable(tf.range(model.num_components), shape=[None])
        self.max_component_id = tf.Variable(tf.reduce_max(self.unique_component_ids), shape=[])

        self.adding_thresholds = tf.Variable(-tf.ones(model.num_components), shape=[None])
        self.initial_entropies = tf.Variable(self.model.component_entropies(), shape=[None])


    def __getattr__(self, name):
        """
        This method forwards all calls to methods and member-variables not implemented in this class to "self.model".
        Hence, we can treat the GmmWrapper like the model, e.g. call gmm_wrapper.sample(N)
        """
        return self.model.__getattribute__(name)

    def add_component(self, initial_weight: tf.float32, initial_mean: tf.Tensor, initial_cov: tf.Tensor,
                      adding_threshold: tf.Tensor, initial_entropy: tf.Tensor):
        """ Adds a new component to the encapuslated model (see :py:meth:`GMM<gmmvi.models.gmm.GMM.add_component>`,
        but also stores / intializes meta-information.

        Parameters:
            initial_weight : tf.Tensor
                The weight of the new component (before re-normalization)

            initial_mean : tf.Tensor
                The mean of the new component

            initial_cov : tf.Tensor
                The covariance matrix of the new component

            adding_threshold: tf.Tensor
                The threshold used by
                :py:class:`VipsComponentAdaptation<gmmvi.optimization.gmmvi_modules.component_adaptation.VipsComponentAdaptation>`,
                stored for debugging.

            initial_entropy: tf.Tensor
                The initial entropy of the new component (can be computed from initial_cov).
        """
        self.model.add_component(initial_weight, initial_mean, initial_cov)
        self.max_component_id.assign_add(1)
        self.unique_component_ids.assign(tf.concat((self.unique_component_ids,
                                                    tf.ones(1, dtype=tf.int32) * self.max_component_id), axis=0))
        self.l2_regularizers.assign(tf.concat((self.l2_regularizers, tf.ones(1) * self.initial_regularizer), axis=0))
        self.last_log_etas.assign(tf.concat((self.last_log_etas, tf.ones(1) * self.initial_last_eta), axis=0))
        self.num_received_updates.assign(tf.concat((self.num_received_updates, tf.zeros(1)), axis=0))
        self.stepsizes.assign(tf.concat((self.stepsizes, tf.ones(1) * self.initial_stepsize), axis=0))
        self.reward_history.assign(
            tf.concat((self.reward_history, tf.ones((1, self.max_reward_history_length)) * tf.float32.min), axis=0))
        self.weight_history.assign(
            tf.concat((self.weight_history, tf.ones((1, self.max_reward_history_length)) * initial_weight), axis=0))
        self.adding_thresholds.assign(
            tf.concat((self.adding_thresholds, adding_threshold), axis=0))
        self.initial_entropies.assign(tf.concat((self.initial_entropies, initial_entropy), axis=0))

    def remove_component(self, idx: int):
        """ Deletes the given component in the encapuslated model (see :py:meth:`GMM<gmmvi.models.gmm.GMM.remove_component>`),
        but also deletes the corresponding meta-information.

        Parameters:
            idx: int
                The idx of the component to be removed.
        """
        self.model.remove_component(idx)
        self.unique_component_ids.assign(tf.concat((self.unique_component_ids[:idx],
                                                    self.unique_component_ids[idx + 1:]), axis=0))
        self.l2_regularizers.assign(tf.concat((self.l2_regularizers[:idx], self.l2_regularizers[idx + 1:]), axis=0))
        self.last_log_etas.assign(tf.concat((self.last_log_etas[:idx], self.last_log_etas[idx + 1:]), axis=0))
        self.num_received_updates.assign(
            tf.concat((self.num_received_updates[:idx], self.num_received_updates[idx + 1:]), axis=0))
        self.stepsizes.assign(tf.concat((self.stepsizes[:idx], self.stepsizes[idx + 1:]), axis=0))
        self.reward_history.assign(tf.concat((self.reward_history[:idx], self.reward_history[idx + 1:]), axis=0))
        self.weight_history.assign(tf.concat((self.weight_history[:idx], self.weight_history[idx + 1:]), axis=0))
        self.adding_thresholds.assign(tf.concat((self.adding_thresholds[:idx], self.adding_thresholds[idx + 1:]), axis=0))
        self.initial_entropies.assign(tf.concat((self.initial_entropies[:idx], self.initial_entropies[idx + 1:]), axis=0))

    def store_rewards(self, rewards: tf.Tensor):
        """
        Store the provided reward of each component.

        Parameters:
            rewards: tf.Tensor
            A one dimensional tensor of size number_of_components, containing the reward for each component
        """
        self.reward_history.assign(tf.concat((self.reward_history[:, 1:], tf.expand_dims(rewards, 1)), axis=1))

    def update_stepsizes(self, new_stepsizes: tf.Tensor):
        """
        This method updates the stepsize for each component.

        Parameters:
            new_stepsizes: tf.Tensor
            A one dimensional tensor of size number_of_components, containing the new stepsize for each component
        """
        self.stepsizes.assign(new_stepsizes)

    def replace_weights(self, new_log_weights: tf.Tensor):
        """
        Overwrites the weights of the encapuslated model, see
        (see :py:meth:`GMM<gmmvi.models.gmm.GMM.remove_component>`), but also keeps track of each component's weight
        from previous iterations.

        Parameters:
            new_log_weights: tf.Tensor
                A one dimensional tensor of size number_of_components, containing the log of the new weight for each
                component.
        """
        self.model.replace_weights(new_log_weights)
        self.weight_history.assign(tf.concat((self.weight_history[:, 1:], tf.expand_dims(self.weights, 1)), axis=1))
