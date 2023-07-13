import tensorflow as tf

from gmmvi.models.gmm_wrapper import GmmWrapper

class ComponentStepsizeAdaptation:
    """ This class provides a common interface for adapting the individual stepsizes for the component updates.

    There are currently three options for component stepsize adpatation:

    1. The :py:class:`FixedComponentStepsizeAdaptation` is a dummy-class, that does not do anything.

    2. The :py:class:`DecayingComponentStepsizeAdaptation` uses exponential decay.

    3. The :py:class:`ImprovementBasedComponentStepsizeAdaptation` uses the procedure of VIPS :cite:p:`Arenz2020` to
       increase the stepsize if a component improved during the last updates, and to decrease it otherwise.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        initial_stepsize: float
            The stepsize used when the component receives its first update
    """

    def __init__(self, gmm_wrapper: GmmWrapper, initial_stepsize: float):
        self.gmm_wrapper = gmm_wrapper
        self.initial_stepsize = initial_stepsize
        tf.assert_equal(gmm_wrapper.stepsizes, initial_stepsize)

    @staticmethod
    def build_from_config(config, gmm_wrapper):
        """This static method provides a convenient way to create a :py:class:`FixedComponentStepsizeAdaptation`,
        :py:class:`DecayingComponentStepsizeAdaptation` or :py:class:`ImprovementBasedComponentStepsizeAdaptation`
        depending on the provided config.

        Parameters:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The wrapped model.
        """
        if config["component_stepsize_adapter_type"] == "improvement-based":
            return ImprovementBasedComponentStepsizeAdaptation(gmm_wrapper, **config["component_stepsize_adapter_config"])
        elif config["component_stepsize_adapter_type"] == "decaying":
            return DecayingComponentStepsizeAdaptation(gmm_wrapper, **config["component_stepsize_adapter_config"])
        elif config["component_stepsize_adapter_type"] == "fixed":
            return FixedComponentStepsizeAdaptation(gmm_wrapper, **config["component_stepsize_adapter_config"])
        else:
            raise ValueError(
                f"config['component_stepsize_adapter_type'] is '{config['component_stepsize_adapter_type']}' "
                f"which is an unknown type")


    def update_stepsize(self, current_stepsizes: tf.Tensor) -> tf.Tensor:
        """
        Update the stepsizes, according to the chosen procedure.

        Parameters:
            current_stepsizes: tf.Tensor
                A tensor that contains the stepsize of each component

        Returns:
            tf.Tensor: a tensor of same size as *current_stepsizes* that contains the updates stepsizes.
        """
        raise NotImplementedError


class FixedComponentStepsizeAdaptation(ComponentStepsizeAdaptation):
    """ This class is a dummy class, that can be used when we want to keep the stepsizes constant.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        initial_stepsize: float
            The stepsize used for all component updates
    """
    def __init__(self, gmm_wrapper: GmmWrapper, initial_stepsize: float):
        super(FixedComponentStepsizeAdaptation, self).__init__(gmm_wrapper, initial_stepsize)

    def update_stepsize(self, current_stepsizes: tf.Tensor) -> tf.Tensor:
        """ This dummy function does nothing.

        Parameters:
            current_stepsizes: tf.Tensor
                A tensor that contains the stepsize of each component

        Returns:
            tf.Tensor: the same *current_stepsizes* tensor, that it was called with.
        """
        return current_stepsizes


class DecayingComponentStepsizeAdaptation(ComponentStepsizeAdaptation):
    """ This class implements an exponentially decaying stepsize schedule.
    See :cite:p:`Khan2018a`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model.

        annealing_exponent: float
            controls how fast the stepsize decays

        initial_stepsize: float
            The stepsize used for all component updates
    """
    def __init__(self, gmm_wrapper: GmmWrapper, annealing_exponent: float, initial_stepsize: float):
        super(DecayingComponentStepsizeAdaptation, self).__init__(gmm_wrapper, initial_stepsize)
        self.annealing_exponent = annealing_exponent
        self.num_received_updates = gmm_wrapper.num_received_updates

    def update_stepsize(self, current_stepsizes: tf.Tensor) -> tf.Tensor:
        """ Updates the stepsize using exponential decay. More specifially, the new stepsize is given by
        :math:`\\frac{\\text{initial\\_stepsize}}{1 + \\text{num\\_received\\_updates}^\\text{annealing\\_exponent}}`.

        Parameters:
            current_stepsizes: tf.Tensor
                A tensor that contains the stepsize of each component

        Returns:
            tf.Tensor: a tensor of same size as *current_stepsizes* that contains the updates stepsizes.
        """
        new_stepsizes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in range(tf.shape(current_stepsizes)[0]):
            new_stepsize = self.initial_stepsize \
                           / (1 + tf.math.pow(float(self.num_received_updates[i]), self.annealing_exponent))
            new_stepsizes = new_stepsizes.write(i, new_stepsize)
        return new_stepsizes.stack()


class ImprovementBasedComponentStepsizeAdaptation(ComponentStepsizeAdaptation):
    """ Increases the stepsize if the last component update increased its reward, decreases it otherwise.
    See :cite:p:`Arenz2020`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        initial_stepsize: float
            The stepsize used for all component updates

        min_stepsize: float
            Do not not decrease the stepsize below this point

        max_stepsize: float
            Do not increase the stepsize above this point

        stepsize_inc_factor: float
            Factor (>1) for increasing the stepsize

        stepsize_dec_factor: float
            Factor in the range [0, 1] for decreasing the stepsize
    """
    def __init__(self, gmm_wrapper: GmmWrapper, initial_stepsize: float, min_stepsize: float, max_stepsize: float,
                 stepsize_inc_factor: float, stepsize_dec_factor: float):
        super(ImprovementBasedComponentStepsizeAdaptation, self).__init__(gmm_wrapper, initial_stepsize)
        self.reward_history = gmm_wrapper.reward_history
        self.min_stepsize = min_stepsize
        self.max_stepsize = max_stepsize
        self.stepsize_inc_factor = stepsize_inc_factor
        self.stepsize_dec_factor = stepsize_dec_factor

    def update_stepsize(self, current_stepsizes: tf.Tensor) -> tf.Tensor:
        """ Updates the stepsize of each component based on previous reward improvements :cite:p:`Arenz2020`.

        Parameters:
            current_stepsizes: tf.Tensor
                A tensor that contains the stepsize of each component

        Returns:
            tf.Tensor: a tensor of same size as *current_stepsizes* that contains the updates stepsizes.
        """
        new_stepsizes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in range(tf.shape(current_stepsizes)[0]):
            if self.reward_history[i][-2] >= self.reward_history[i][-1]:
                new_stepsize = tf.math.maximum(
                    self.stepsize_dec_factor * current_stepsizes[i],
                    self.min_stepsize,
                )
            else:
                new_stepsize = tf.math.minimum(
                    self.stepsize_inc_factor * current_stepsizes[i],
                    self.max_stepsize,
                )
            new_stepsizes = new_stepsizes.write(i, new_stepsize)
        return new_stepsizes.stack()
