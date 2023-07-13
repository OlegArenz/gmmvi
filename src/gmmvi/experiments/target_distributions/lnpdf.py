import tensorflow as tf

from gmmvi.models.gmm_wrapper import GmmWrapper


class LNPDF():
    """This class defines the interface for target distributions.
    Every target distribution needs to implement the method log_density(x) for computing the unnormalized
    log-density of the target distribution.

    This function can be also used to wrap target distributions that are not implemented in Tensorflow, by setting
    use_log_density_and_grad to True and safe_for_tf_graph to False.

    Parameters:
        use_log_density_and_grad: bool
            if False, user is allowed to backprob through self.log_density(), otherwise the method
            :py:meth:`log_density_and_grad<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density_and_grad>`
            should be used (and needs to be implemented when using first-order estimates of the NG).

        safe_for_tf_graph: bool
            if True, we can call
            :py:meth:`log_density<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density>`
            and :py:meth:`log_density_and_grad<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density_and_grad>`
            within a tf.function().
    """

    def __init__(self, use_log_density_and_grad: bool=False, safe_for_tf_graph: bool=True):
        self._use_log_density_and_grad = use_log_density_and_grad
        self._safe_for_tf_graph = safe_for_tf_graph

    def log_density(self, x: tf.Tensor) -> tf.Tensor:
        """Returns the unnormalized log-density for each sample in x, :math:`\\log p(\\mathbf{x})`.

        Parameters:
            x: tf.Tensor
                The samples that we want to evaluate, a tf.Tensor of shape number_of_samples x dimensions.

        Returns:
            tf.Tensor: A one-dimensional tensor of shape number_of_samples containing the unnormalized log-densities.
        """
        raise NotImplementedError

    def log_density_and_grad(self, x: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """(May not be implemented) Returns the unnormalized log-density and its gradient for each sample in x.

        Parameters:
            x: tf.Tensor
                The samples that we want to evaluate, a tf.Tensor of shape number_of_samples x dimensions.

        Returns:
            tuple(tf.Tensor, tf.Tensor):

            **target_log_densities** - a one-dimensional tensor of shape number_of_samples
            containing the unnormalized log-densities.

            **log_density_grads** - a two-dimensional tensor of shape
            number_of_samples x dimensions containing the gradients of the log-densities with respect to the respective
            sample.
        """
        raise NotImplementedError

    def get_num_dimensions(self) -> int:
        """
        Returns:
            int: the number of dimensions
        """
        raise NotImplementedError

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ (May not be implemented) This method can be used for computing environment-specific metrics or plots
        that we want to log. It is called by the :py:class:`GmmviRunner<gmmvi.gmmvi_runner.GmmviRunner>`.

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing the name and value for each item we wish to log.
        """
        return dict()

    def can_sample(self) -> bool:
        """ If the target distribution can be sampled, and the respective
         :py:meth:`method<gmmvi.experiments.target_distributions.lnpdf.LNPDF.sample>` is implemented, you can overwrite
         this method to return True.

         Returns:
            bool: is it is safe to call :py:meth:`sample<gmmvi.experiments.target_distributions.lnpdf.LNPDF.sample>`?
         """
        return False

    @property
    def use_log_density_and_grad(self) -> bool:
        """
        if False, user is allowed to backprob through self.log_density(), otherwise the method
        :py:meth:`log_density_and_grad<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density_and_grad>`
        should be used (and needs to be implemented when using first-order estimates of the NG).
        """
        return self._use_log_density_and_grad

    @property
    def safe_for_tf_graph(self) -> bool:
        """
        if True, we can call
        :py:meth:`log_density<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density>`
        and :py:meth:`log_density_and_grad<gmmvi.experiments.target_distributions.lnpdf.LNPDF.log_density_and_grad>`
        within a tf.function().
        """
        return self._safe_for_tf_graph

    def sample(self, n: int) -> tf.Tensor:
        """
        (May not be implemented) If we can sample from the target distribution,
        this functionality can be implemented here. However, it is not used by the learning algorithm.

        Parameters:
            n: int
                The number of samples we want to draw

        Returns:
            tf.Tensor: the sample, a Tensor of shape n x dimensions

        """
        raise NotImplementedError
