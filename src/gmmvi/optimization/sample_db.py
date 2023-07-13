from math import pi
import tensorflow as tf

class SampleDB:
    """ A database for storing samples and meta-information.

    Along the samples, we also store

    1. The parameters of the Gaussian distribution that were used for obtaining each sample

    2. log-density evaluations of the target distribution, :math:`\\log p(\\mathbf{x})`

    3. (if available), gradients of the log-densites of the target distribution,
       :math:`\\nabla_\\mathbf{x} \\log p(\\mathbf{x})`

    Parameters:
        dim: int
            dimensionality of the samples to be stored

        diagonal_covariances: bool
            True, if the samples are always drawn from Gaussians with diagonal covariances (saves memory)

        keep_samples: bool
            If this is False, the samples are not actually stored

        max_samples: int
            Maximal number of samples that are stored. If adding new samples would exceed this limit, every N-th sample
            in the database gets deleted.
    """
    def __init__(self, dim, diagonal_covariances, keep_samples, max_samples=None):
        self._dim = dim
        self.diagonal_covariances = diagonal_covariances
        self.keep_samples = keep_samples
        self.samples = tf.Variable(tf.zeros((0, dim)), shape=[None, dim])
        self.means = tf.Variable(tf.zeros((0, dim)), shape=[None, dim])
        if diagonal_covariances:
            self.chols = tf.Variable(tf.zeros((0, dim)), shape=[None, dim])
            self.inv_chols = tf.Variable(tf.zeros((0, dim)), shape=[None, dim])
        else:
            self.chols = tf.Variable(tf.zeros((0, dim, dim)), shape=[None, dim, dim])
            self.inv_chols = tf.Variable(tf.zeros((0, dim, dim)), shape=[None, dim, dim])
        self.target_lnpdfs = tf.Variable(tf.zeros(0), shape=[None])
        self.target_grads = tf.Variable(tf.zeros((0, dim)), shape=[None, dim])
        self.mapping = tf.Variable(tf.zeros(0, dtype=tf.int32), shape=[None], dtype=tf.int32)
        self.num_samples_written = tf.Variable(0, shape=[], dtype=tf.int32)
        self.max_samples = max_samples

    @staticmethod
    def build_from_config(config, num_dimensions):
        """ A static method to conveniently create a :py:class:`SampleDB` from a given config dictionary.

        Parametes:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            num_dimensions: int
                dimensionality of the samples to be stored
        """
        return SampleDB(num_dimensions, config["model_initialization"]["use_diagonal_covs"],
                             config["use_sample_database"],
                             config["max_database_size"])

    @tf.function
    def remove_every_nth_sample(self, N: int):
        """ Deletes Every N-th sample from the database and the associated meta information.

        Parameters:
            N: int
                abovementioned N
        """
        self.samples.assign(self.samples[::N])
        self.target_lnpdfs.assign(self.target_lnpdfs[::N])
        self.target_grads.assign(self.target_grads[::N])
        self.mapping.assign(self.mapping[::N])
        used_indices, reduced_mapping = tf.unique(self.mapping)
        self.mapping.assign(reduced_mapping)
        self.means.assign(tf.gather(self.means, used_indices))
        self.chols.assign(tf.gather(self.chols, used_indices))
        self.inv_chols.assign(tf.gather(self.inv_chols, used_indices))

    @tf.function(experimental_relax_shapes=True)
    def add_samples(self, samples, means, chols, target_lnpdfs, target_grads, mapping):
        """ Add the given samples to the database.

        Parameters:
            samples: tf.Tensor
                a two-dimensional tensor of shape num_samples x num_dimensions containing the samples to be added.

            means: tf.Tensor
                a two-dimensional tensor containing for each Gaussian distribution that was used for obtaining the
                samples the corresponding mean. The first dimension of the tensor can be smaller than the number of
                samples, if several samples where drawn from the same Gaussian (see the parameter *mapping*).

            chols: tf.Tensor
                a three-dimensional tensor containing for each Gaussian distribution that was used for obtaining the
                samples the corresponding Cholesky matrix. The first dimension of the tensor can be smaller than the
                number of samples, if several samples where drawn from the same Gaussian (see the parameter *mapping*).

            target_lnpdfs: tf.Tensor
                a one-dimensional tensor containing the log-densities of the (unnormalized) target distribution,
                :math:`\\log p(\\mathbf{x})`.

            target_grads: tf.Tensor
                a two-dimensional tensor containing the gradients of the log-densities of the (unnormalized) target
                distribution, :math:`\\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})`.

            mapping: tf.Tensor
                a tensor of size number_of_samples, which corresponds for every sample the index to *means* and *chols*
                that corresponds to the Gaussian distribution that was used for drawing that sample.
        """
        if self.max_samples is not None and tf.shape(samples)[0] + tf.shape(self.samples)[0] > self.max_samples:
            self.remove_every_nth_sample(2)
        self.num_samples_written.assign_add(tf.shape(samples)[0])
        if self.keep_samples:
            self.mapping.assign(tf.concat((self.mapping, mapping + tf.shape(self.chols)[0]), axis=0))
            self.means.assign(tf.concat((self.means, means), axis=0))
            self.chols.assign(tf.concat((self.chols, chols), axis=0))
            if self.diagonal_covariances:
                self.inv_chols.assign(tf.concat((self.inv_chols, 1./chols), axis=0))
            else:
                self.inv_chols.assign(tf.concat((self.inv_chols, tf.linalg.inv(chols)), axis=0))
            self.samples.assign(tf.concat((self.samples, samples), axis=0))
            self.target_lnpdfs.assign(tf.concat((self.target_lnpdfs, target_lnpdfs), axis=0))
            self.target_grads.assign(tf.concat((self.target_grads, target_grads), axis=0))
        else:
            self.mapping.assign(mapping)
            self.means.assign(means)
            self.chols.assign(chols)
            if self.diagonal_covariances:
                self.inv_chols.assign(1./chols)
            else:
                self.inv_chols.assign(tf.linalg.inv(chols))
            self.samples.assign(samples)
            self.target_lnpdfs.assign(target_lnpdfs)
            self.target_grads.assign(target_grads)

    def get_random_sample(self, N: int):
        """ Get N random samples from the database.

        Parameters:
            N: int
                abovementioned N

        Returns:
            tuple(tf.Tensor, tf.Tensor)

            **samples** - the chosen samples

            **target_lnpdfs** - the corresponding log densities of the target distribution
        """
        chosen_indices = tf.random.shuffle(tf.range(tf.shape(self.samples)[0]))[:N]
        return tf.gather(self.samples, chosen_indices), tf.gather(self.target_lnpdfs, chosen_indices)

    def gaussian_log_pdf(self, mean, chol, inv_chol, x):
        if self.diagonal_covariances:
            constant_part = - 0.5 * self._dim * tf.math.log(2 * pi) - tf.reduce_sum(
                tf.math.log(chol))
            return constant_part - 0.5 * tf.reduce_sum(tf.square(tf.expand_dims(inv_chol, 1)
                                                                 * tf.transpose(tf.expand_dims(mean, 0) - x)), axis=0)
        else:
            constant_part = - 0.5 * self._dim * tf.math.log(2 * pi) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return constant_part - 0.5 * tf.reduce_sum(tf.square(inv_chol @ tf.transpose(mean - x)), axis=0)

    def evaluate_background(self, weights, means, chols, inv_chols, samples):
        """ Evaluates the log-densities of the given samples on a GMM with the given parametes. This function is
         implemented in a memory-efficient way to scale to mixture models with many components.

         Parameters:
             weights: tf.Tensor
                The weights of the GMM that should be evaluated

            means: tf.Tensor
                The means of the GMM that should be evaluated

            chols: tf.Tensor
                The Cholesky matrices of the GMM that should be evaluated

            inv_chols: tf.Tensor
                The inverse of abovementioned *chols*

            samples: tf.Tensor
                The samples to be evaluated.
         """
        log_weights = tf.math.log(weights)
        log_pdfs = self.gaussian_log_pdf(means[0], chols[0], inv_chols[0], samples) + log_weights[0]

        for i in range(1, len(weights)):
            log_pdfs = tf.reduce_logsumexp(tf.stack((
                log_pdfs,
                self.gaussian_log_pdf(means[i], chols[i], inv_chols[i], samples) + log_weights[i]
            ), axis=0), axis=0)
        return log_pdfs

    @tf.function()
    def get_newest_samples(self, N):
        """ Returns (up to) the N newest samples, and their meta-information.

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):

            **log_pdfs** - the log-density of the GMM that was effectively used for drawing the samples (used for
            importance sampling)

            **active_sample** - the selected samples

            **active_mapping** - contains for every sample the index of the component that was used for drawing it

            **active_target_lnpdfs** - log-density evaluations of the target distribution for the selected samples

            **active_target_grads** - gradients evaluations of the log-density of the target distribution for the
            selected samples
        """
        if tf.shape(self.samples)[0] == 0 or N == 0:
            return tf.zeros(0), tf.zeros((0, self._dim)), tf.zeros(0, dtype=tf.int32), tf.zeros(0), tf.zeros((0, self._dim))
        else:
            active_sample_index = tf.maximum(0, tf.shape(self.samples)[0] - N)
            active_sample = self.samples[active_sample_index:]
            active_target_lnpdfs = self.target_lnpdfs[active_sample_index:]
            active_target_grads = self.target_grads[active_sample_index:]
            active_mapping = self.mapping[active_sample_index:]
            active_components, _, count = tf.unique_with_counts(active_mapping)
            means = tf.gather(self.means, active_components)
            chols = tf.gather(self.chols, active_components)
            inv_chols = tf.gather(self.inv_chols, active_components)
            count = tf.cast(count, tf.float32)
            weight = count / tf.reduce_sum(count)
            log_pdfs = self.evaluate_background(weight, means, chols, inv_chols, active_sample)
            return log_pdfs, active_sample, active_mapping, active_target_lnpdfs, active_target_grads
