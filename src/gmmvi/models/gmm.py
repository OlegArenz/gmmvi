import tensorflow as tf
from math import pi


class GMM:
    """An abstract class for Gaussian mixture models (GMMs).

    This class stores the parameters of a GMM (weights, means and Cholesky matrices) and provides functionality
    that is common for different types of GMMs (e.g., GMMs with full covariance matrices, and those with diagonal
    covariance matrices). For example, this class provides methods for sampling the *GMM*, evaluating its probability
    density function, and entropy, while relying on the subclass for sampling the *components*, etc.

    Parameters:
        log_weights: tf.Variable(float)
            A tensorflow Variable for storing the log-probabilities of the component weights.

        means: tf.Variable(float)
            A tensorflow Variable for storing the component means (number of components X number of dimensions)

        chol_covar: tf.Variable(float)
            A tensorflow Variable for storing the Cholesky matrix of the component's covariance matrix. The first
            dimension specifies the index of the components. The rank may vary depending on the subclass. For example,
            when storing the Cholesky matrix for a diagonal covariance matrix, it is possible to use a rank-2 Tensor
            (number of components X number of dimensions) for better memory efficiency.
    """

    def __init__(self, log_weights: tf.Variable, means: tf.Variable, chol_covs: tf.Variable):
        self.diagonal_covs = tf.rank(chol_covs) == 2
        self.num_dimensions = int(tf.shape(means)[1])
        self._const_log_det = tf.constant(0.5 * self.num_dimensions * tf.math.log(2 * pi))
        self.log_weights = log_weights
        self.means = means
        self.chol_cov = chol_covs
        self.replace_weights(self.log_weights)

    def sample_from_component(self, index: int, num_samples: int) -> tf.Tensor:
        """ draw samples from the specified components

        Parameters:
            index: int
                The index of the component from which we want to sample.

            num_samples: int
                The number of samples to be drawn.

        Returns:
            tf.Tensor: The drawn samples, tensor of size num_samples x dimensions.
        """
        raise NotImplementedError

    def component_log_density(self, index: int, samples: tf.Tensor) -> tf.Tensor:
        """ Use the specified component to evaluate the Gaussian log-density at the given samples.

        Parameters:
            index: int
                The index of the component of which we want to compute the log densities.

            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate.

        Returns:
            tf.Tensor: a one-dimensional tensor of size number_of_samples, containing the log-densities.
        """
        raise NotImplementedError

    def component_log_densities(self, samples: tf.Tensor) -> tf.Tensor:
        """ Evaluate the log densities for each mixture component on the given samples.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate.

        Returns:
            tf.Tensor: a two-dimensional tensor of size number_of_components x number_of_samples,
            containing the log-densities for each component.
        """
        raise NotImplementedError

    def component_marginal_log_densities(self, samples: tf.Tensor, dimension: int) -> tf.Tensor:
        """ Evaluate the marginal log densities for each mixture component along the given dimension for each sample.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate. Note that
                for providing an easier interface, each sample has the number of dimensions compatible with this GMM,
                although only a single entry is actually used for evaluating the marginal density.

            dimension: int
                The dimension of interest.

        Returns:
            tf.Tensor: a two-dimensional tensor of size number_of_components x number_of_samples,
            containing the marginal log-densities for each component.
        """
        raise NotImplementedError

    def gaussian_entropy(self, chol: tf.Tensor) -> tf.Tensor:
        """ Computes the entropy of Gaussian distribution with the given Cholesky matrix.

        Parameters:
            chol: tf.Tensor
                A two-dimensional tensor of shape number_of_dimensions x number_of_dimensions specifying the Cholesky matrix

        Returns:
            tf.float32: The entropy
        """
        raise NotImplementedError

    def add_component(self, initial_weight: tf.Tensor, initial_mean: tf.Tensor, initial_cov: tf.Tensor):
        """ Add a component to the mixture model. The weights will be automatically normalized.

        Parameters:
            initial_weight: tf.Tensor
                The weight of the new component (before re-normalization)

            initial_mean: tf.Tensor
                The mean of the new component

            initial_cov: tf.Tensor
                The covariance matrix of the new component
        """
        raise NotImplementedError

    def sample_categorical(self, num_samples: int) -> tf.Tensor:
        """ Sample components according to the weights

        Parameters:
            num_samples : int
                The number of components to be drawn

        Returns:
            tf.Tensor: a one-dimensional tensor of int, containing the component indices.
        """
        thresholds = tf.expand_dims(tf.cumsum(self.weights), 0)
        eps = tf.random.uniform(shape=[num_samples, 1])
        samples = tf.argmax(eps < thresholds, axis=-1, output_type=tf.int32)
        return samples

    def sample(self, num_samples: int) -> [tf.Tensor, tf.Tensor]:
        """ Draw samples from this GMM, also returns for every sample, the index of the component that was used for
        sampling it.

        Parameters:
            num_samples: int
                The number of samples to be drawn

        Returns:
            tuple(tf.Tensor, tf.Tensor):

            **drawn_samples** - a two-dimensional tensor of shape num_samples x num_dimensions,
            containing the drawn samples.

            **component_indices** - a one-dimensional tensor of int, containing the component indices.
        """
        sampled_components = self.sample_categorical(num_samples=num_samples)
        samples = tf.TensorArray(tf.float32, size=self.num_components, infer_shape=False)
        for i in range(self.num_components):
            n_samples = tf.reduce_sum(tf.cast(sampled_components == i, tf.int32))
            this_samples = self.sample_from_component(i, n_samples)
            samples = samples.write(i, this_samples)

        samples = samples.concat()
        return samples, sampled_components

    @property
    def weights(self) -> tf.Tensor:
        """
        Returns:
            tf.Tensor: a one-dimensional tensor of size num_components, containing the component weights.
        """
        return tf.math.exp(self.log_weights)

    def replace_weights(self, new_log_weights: tf.Tensor):
        """
        Overwrites the component log(weights). This method will take care of normalization.

        Parameters:
            new_log_weights: tf.Tensor
                a one-dimensional tensor of size num_components, containing the new log(weights)
        """
        self.log_weights.assign(new_log_weights - tf.reduce_logsumexp(new_log_weights))

    def log_densities_also_individual(self, samples: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """
        Evaluates the given samples on this GMM, but also returns individual log-densities for each Gaussian
        component.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate

        Returns:
            tuple(tf.Tensor, tf.Tensor):
            **model_log_densities** - a one-dimensional tnsor of shape num_samples containing the model log-densities.

            **component_log_densities** - a two-dimensional tensor of shape num_components x num_samples containing
            the component log-densities
        """
        component_log_densities = self.component_log_densities(samples)
        weighted_densities = component_log_densities + tf.expand_dims(self.log_weights, axis=1)
        return tf.reduce_logsumexp(weighted_densities, axis=0), component_log_densities

    def log_density(self, samples: tf.Tensor) -> tf.Tensor:
        """
        Evaluates the given samples on this GMM.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate.

        Returns:
            tf.Tensor: a one-dimensional tensor of shape num_samples containing the model log-densities.
        """
        log_densities = self.component_log_densities(samples)
        weighted_densities = log_densities + tf.expand_dims(self.log_weights, axis=1)
        return tf.reduce_logsumexp(weighted_densities, axis=0)

    def marginal_log_density(self, samples: tf.Tensor, dimension: int) -> tf.Tensor:
        """
        Evaluates this GMM on the given samples with respect to the marginal log-density along the given dimensions

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate

            dimension: int
                The dimension of interest

        Returns:
            tf.Tensor: a one-dimensional tensor of shape num_samples containing the marginal log-densities
        """
        log_densities = self.component_marginal_log_densities(samples, dimension)
        weighted_densities = log_densities + tf.expand_dims(self.log_weights, axis=1)
        return tf.reduce_logsumexp(weighted_densities, axis=0)

    def density(self, samples: tf.Tensor) -> tf.Tensor:
        """
        Evaluates the given samples on this GMM.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate

        Returns:
            tf.Tensor: a one-dimensional tensor of shape num_samples containing the model densities.
        """
        return tf.exp(self.log_density(samples))

    def component_entropies(self) -> tf.Tensor:
        """
        Computes the entropy of each component.

        Returns:
            tf.Tensor: a one-dimensional tensor of shape num_components containing the component entropies.
        """
        entropies = tf.TensorArray(tf.float32, size=self.num_components)
        for i in range(self.num_components):
            this_entropy = self.gaussian_entropy(self.chol_cov[i])
            entropies = entropies.write(i, this_entropy)
        return entropies.stack()

    def get_average_entropy(self) -> tf.float32:
        """
        Averages the entropies of the individual components based on their respective weights.

        Returns:
            tf.float32: the average component entropy
        """
        avg_entropy = 0.
        for i in range(self.num_components):
            avg_entropy += tf.exp(self.log_weights[i]) * self.gaussian_entropy(self.chol_cov[i])
        return avg_entropy

    def log_density_and_grad(self, samples: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Evaluates the given samples on this GMM, returns the log-densities of the whole model, their gradients, and
        also the individual log-densities for each Gaussian component.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate.

        Returns:
            tuple(tf.Tensor, tf.Tensor):

            **model_log_densities** - a one-dimensional tensor of shape num_samples containing the model log-densities.

            **model_log_density_grads** - a two-dimensional tf.Tensor of shape num_samples x num_dimensions containing
            the gradients of the model log-densities.

            **component_log_densities** - a two-dimensional tf.Tensor of shape num_components x num_samples containing
            the component log-densities.
        """
        with tf.GradientTape() as gfg:
            gfg.watch(samples)
            log_component_densities = self.component_log_densities(samples)
            log_densities = log_component_densities + tf.expand_dims(self.log_weights, axis=1)
            log_densities = tf.reduce_logsumexp(log_densities, axis=0)
        log_densities_grad = gfg.gradient(log_densities, samples)
        return log_densities, log_densities_grad, log_component_densities

    def component_log_density_and_grad(self, index: int, samples: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """
        Evaluates for the given component the log-density and its gradient.

        Parameters:
            samples: tf.Tensor
                A two-dimensional tensor of shape number_of_samples x num_dimensions, which we want to evaluate.

            Returns:
                **component_log_densities** - a one-dimensional tf.Tensor of shape num_samples containing the log-densities
                of the given component.

                **component_log_density_grads** - a two-dimensional tf.Tensor of shape num_samples x num_dimensions
                containing the gradients of the component's log-densities.
        """
        with tf.GradientTape() as gfg:
            gfg.watch(samples)
            log_component_density = self.component_log_density(index, samples)
        log_density_grad = gfg.gradient(log_component_density, samples)
        return log_component_density, log_density_grad

    @property
    def covs(self) -> tf.Tensor:
        """
        Returns:
            tf.Tensor: the covariance matrices as a 3-dimensional tensor of shape
            num_components x num_dimensions x num_dimensions
        """
        raise NotImplementedError

    @property
    def num_components(self) -> int:
        """
        Returns:
            int: the number of components of this GMM
        """
        return tf.shape(self.log_weights)[0]

    def sample_from_components(self, samples_per_component: tf.Tensor) -> tf.Tensor:
        """
        Draws from each component the corresponding number of samples (provided as a one-dimensional tensor).

        Parameters:
            samples_per_component : tf.Tensor
                a one-dimensional tensor of size number_of_component, containint for each component the number of
                samples to be drawn.

        Returns:
            tf.Tensor: a tensor of shape sum(samples_per_component) x num_dimensions containing the samples (shuffled).

        """
        samples = tf.TensorArray(tf.float32, size=self.num_components)
        for i in range(self.num_components):
            this_samples = self.sample_from_component(i, samples_per_component[i])
            samples = samples.write(i, this_samples)
        samples = tf.reshape(samples.stack(), [-1, self.num_dimensions])
        chosen_indices = tf.random.shuffle(tf.range(tf.shape(samples)[0]))
        return tf.gather(samples, chosen_indices)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def sample_from_components_no_shuffle(self, samples_per_component: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """
        Draws from each component the corresponding number of samples (provided as a one-dimensional tensor).
        Similar to :py:meth:`sample_from_components<gmmvi.models.gmm.GMM.sample_from_components>`, but the returned
        samples are not shuffled.

        Parameters:
            samples_per_component: tf.Tensor
                a one-dimensional tensor of size number_of_component, containint for each component the number of
                samples to be drawn.

        Returns:
            tf.Tensor: a tensor of shape sum(samples_per_component) x num_dimensions containing the samples
            (not shuffled).

        """
        mapping = tf.repeat(
            tf.range(self.num_components), samples_per_component
        )
        samples = tf.TensorArray(tf.float32, size=self.num_components)
        for i in range(self.num_components):
            this_samples = self.sample_from_component(i, samples_per_component[i])
            samples = samples.write(i, this_samples)
        samples = samples.concat()
        return samples, mapping

    def remove_component(self, idx: int):
        """
        Removes the specified component, and renormalizes the weights afterwards.

        Parameters:
            idx: int
                The idx of the component to be removed.
        """
        self.replace_weights(tf.concat((self.log_weights[:idx], self.log_weights[idx + 1:]), axis=0))
        self.means.assign(tf.concat((self.means[:idx], self.means[idx + 1:]), axis=0))
        self.chol_cov.assign(tf.concat((self.chol_cov[:idx], self.chol_cov[idx + 1:]), axis=0))


    def replace_components(self, new_means: tf.Tensor, new_chols: tf.Tensor):
        """
        Updates the means and covariances matrices (Cholesky) of the GMM.
        The weights and, therefore, the number of components can not be changed with this method.

        Parameters:
            new_means: tf.Tensor
                a two-dimensional tensor of shape current_number_of_components x dimensions, specifying the updated
                means.

            new_chols: tf.Tensor
                a three-dimensional tensor of shape current_number_of_components x dimensions x dimensions,
                specifying the updated Cholesky matrix.
        """
        new_means = tf.stack(new_means, axis=0)
        new_chols = tf.stack(new_chols, axis=0)
        self.means.assign(new_means)
        self.chol_cov.assign(new_chols)