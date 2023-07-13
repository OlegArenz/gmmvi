import tensorflow as tf

from gmmvi.models.gmm_wrapper import GmmWrapper


class WeightStepsizeAdaptation:
    """ This class provides a common interface for adapting the stepsize for the weight update.

    There are currently three options for weight stepsize adpatation:

    1. The :py:class:`FixedWeightStepsizeAdaptation` is a dummy-class, that does not do anything.

    2. The :py:class:`DecayingWeightStepsizeAdaptation` uses exponential decay.

    3. The :py:class:`ImprovementBasedWeightStepsizeAdaptation` uses a procedure similar to VIPS :cite:p:`Arenz2020` to
       increase the stepsize if the mixture improved during the last updates, and to decrease it otherwise.

    Parameters:
        initial_stepsize: float
            The initial stepsize for the weight update.
    """

    def __init__(self, initial_stepsize: tf.float32):
        self.stepsize = tf.Variable(initial_stepsize, dtype=tf.float32)

    @staticmethod
    def build_from_config(config, gmm_wrapper):
        """This static method provides a convenient way to create a :py:class:`FixedWeightStepsizeAdaptation`,
        :py:class:`DecayingWeightStepsizeAdaptation` or :py:class:`ImprovementBasedWeightStepsizeAdaptation`
        depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model.
        """
        if config["weight_stepsize_adapter_type"] == "fixed":
            return FixedWeightStepsizeAdaptation(**config['weight_stepsize_adapter_config'])
        elif config["weight_stepsize_adapter_type"] == "decaying":
            return DecayingWeightStepsizeAdaptation(**config['weight_stepsize_adapter_config'])
        elif config["weight_stepsize_adapter_type"] == "improvement_based":
            return ImprovementBasedWeightStepsizeAdaptation(gmm_wrapper, **config['weight_stepsize_adapter_config'])
        else:
            raise ValueError(
                f"config['weight_stepsize_adapter_type'] is '{config['weight_stepsize_adapter_type']}' "
                f"which is an unknown type")

    def _update_stepsize(self):
        pass

    def update_stepsize(self):
        """
        Update the stepsizes, according to the chosen procedure.

        Returns:
            float: the updated stepsize.
        """
        self._update_stepsize()
        return self.stepsize


class FixedWeightStepsizeAdaptation(WeightStepsizeAdaptation):
    """ This class is a dummy class, that can be used when we want to keep the stepsize for the weight update constant.

    Parameters:
        initial_stepsize: float
            The initial stepsize for the weight update.
    """
    def __init__(self, initial_stepsize: tf.float32):
        super(FixedWeightStepsizeAdaptation, self).__init__(initial_stepsize)


class DecayingWeightStepsizeAdaptation(WeightStepsizeAdaptation):
    """ This class implements an exponentially decaying stepsize schedule.

    See :cite:p:`Khan2018a`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped.

        annealing_exponent: float
            controls how fast the stepsize decays

        initial_stepsize: float
            The initial stepsize for the weight update.
    """
    def __init__(self, initial_stepsize: tf.float32, annealing_exponent: tf.float32):
        super(DecayingWeightStepsizeAdaptation, self).__init__(initial_stepsize)
        self.initial_stepsize = tf.constant(initial_stepsize)
        self.annealing_exponent = annealing_exponent
        self.num_weight_updates = tf.Variable(0, dtype=tf.float32)

    def _update_stepsize(self):
        """ Updates the stepsize using exponential decay. More specifially, the new stepsize is given by
        :math:`\\frac{\\text{initial\\_stepsize}}{1 + \\text{num\\_iterations}^\\text{annealing\\_exponent}}`.

        Returns:
            float: the updated stepsize.
        """
        self.stepsize.assign(self.initial_stepsize / (1. + tf.math.pow(self.num_weight_updates,
                                                                       self.annealing_exponent)))
        self.num_weight_updates.assign_add(1.)


class ImprovementBasedWeightStepsizeAdaptation(WeightStepsizeAdaptation):
    """ Increases the stepsize if the last weight update increased its reward, decreases it otherwise.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model.

        initial_stepsize: float
            The initial stepsize for the weight update.

        min_stepsize: float
            Do not not decrease the stepsize below this point

        max_stepsize: float
            Do not increase the stepsize above this point

        stepsize_inc_factor: float
            Factor (>1) for increasing the stepsize

        stepsize_dec_factor: float
            Factor in the range [0, 1] for decreasing the stepsize
    """
    def __init__(self, model: GmmWrapper, initial_stepsize: tf.float32, min_stepsize: tf.float32,
                 max_stepsize: tf.float32, stepsize_inc_factor: tf.float32, stepsize_dec_factor: tf.float32):
        super(ImprovementBasedWeightStepsizeAdaptation, self).__init__(initial_stepsize)
        self.initial_stepsize = tf.constant(initial_stepsize)
        self.model = model
        self.min_stepsize = min_stepsize
        self.max_stepsize = max_stepsize
        self.stepsize_inc_factor = stepsize_inc_factor
        self.stepsize_dec_factor = stepsize_dec_factor
        self.elbo_history = tf.Variable([tf.float32.min], shape=[None], dtype=tf.float32)

    def _update_stepsize(self):
        """ Updates the stepsize of each component based on the mixture model improvement.

        Returns:
            float: the updated stepsize.
        """
        elbo = tf.reduce_sum(self.model.weights * self.model.reward_history[:,-1]) - tf.reduce_sum(self.model.weights * self.model.log_weights)
        self.elbo_history.assign(tf.concat((self.elbo_history, tf.expand_dims(elbo,0)), axis=0))
        if self.elbo_history[-1] > self.elbo_history[-2]:
            self.stepsize.assign(tf.math.minimum(
                self.stepsize_inc_factor * self.stepsize,
                self.max_stepsize))
        else:
            self.stepsize.assign(tf.math.maximum(
                self.stepsize_dec_factor * self.stepsize,
                self.min_stepsize))

