from pathlib import Path

import tensorflow as tf
import numpy as np
from .lnpdf import LNPDF
from math import pi

from gmmvi.models.gmm_wrapper import GmmWrapper


class LogisticRegression(LNPDF):
    """This class is used for implementing the logistic regression experiments based on the BreastCancer and
    GermanCredit dataset :cite:p:`UCI`, reimplementing the experiments used by :cite:t:`Arenz2020`.

    Parameters:
        dataset_id: a string
            Should be either "breast_cancer" or "german_credit"
    """

    def __init__(self, dataset_id):
        super(LogisticRegression, self).__init__(use_log_density_and_grad=False)
        self.const_term = tf.constant(tf.cast(0.5 * tf.math.log(2. * pi), dtype=tf.float32))
        root_path = Path(__file__).parent.resolve()
        if dataset_id == "breast_cancer":
            path = Path.joinpath(root_path, "datasets/breast_cancer.data")
            data = np.loadtxt(str(path))
            X = data[:, 2:]
            X /= np.std(X, 0)[np.newaxis, :]
            X = np.hstack((np.ones((len(X), 1)), X))
            self.data = tf.cast(X, tf.float32)
            self.labels = data[:, 1]
            self.num_dimensions = self.data.shape[1]
            self._prior_std = tf.constant(10., dtype=tf.float32)
            self.prior_mean = tf.constant(0., dtype=tf.float32)
        elif dataset_id == "german_credit":
            path = Path.joinpath(root_path, "datasets/german.data-numeric")
            data = np.loadtxt(str(path))
            X = data[:, :-1]
            X /= np.std(X, 0)[np.newaxis, :]
            X = np.hstack((np.ones((len(X), 1)), X))
            self.data = tf.cast(X, tf.float32)
            self.labels = data[:, -1] - 1
            self.num_dimensions = self.data.shape[1]
            self._prior_std = tf.constant(10., dtype=tf.float32)
            self.prior_mean = tf.constant(0., dtype=tf.float32)
        self.labels = tf.Variable(tf.expand_dims(self.labels.astype(np.float32), 1))

    def get_num_dimensions(self):
        return self.num_dimensions

    @property
    def prior_std(self):
        return self._prior_std

    def log_likelihood(self, x):
        features = -tf.matmul(self.data, tf.transpose(x))
        log_likelihoods = tf.where(self.labels == 1, tf.transpose(tf.math.log_sigmoid(features)),
                                   tf.transpose(tf.math.log_sigmoid(features) - features))
        return log_likelihoods

    def log_density(self, x):
        features = -tf.matmul(self.data, tf.transpose(x))
        log_likelihoods = tf.reduce_sum(tf.where(self.labels == 1, tf.math.log_sigmoid(features),
                                                 tf.math.log_sigmoid(features) - features), axis=0)
        log_prior = tf.reduce_sum(-tf.math.log(self.prior_std) - self.const_term - 0.5 * tf.math.square(
            (x - self.prior_mean) / self.prior_std), axis=1)
        log_posterior = log_likelihoods + log_prior
        return log_posterior

class LogisticRegression_minibatch(LogisticRegression):
    """This class is used for implementing minibatch-variants of the GermanCredit and BreastCancer
    :py:class:`experiments<gmmvi.experiments.target_distributions.logistic_regression.LogisticRegression>`

    Parameters:
        dataset_id: str
            Should be either "breast_cancer" or "german_credit"

        batchsize: int
            batchsize for evaluating the likelihood.

        size_test_set: int
            number of training data that should be held out.

        use_own_batch_per_samples: bool
            if True, a different minibatch is used for every sample for which we want to evaluate the target log-density,
            which reduces the variance (local reparameterization).
    """
    def __init__(self, dataset_id, batchsize, size_test_set, use_own_batch_per_sample):
        super(LogisticRegression_minibatch, self).__init__(dataset_id)
        self.data = tf.Variable(self.data)
        if size_test_set > 0:
            self.data_test = tf.Variable(self.data[-size_test_set:])
            self.labels_test = tf.Variable(self.labels[-size_test_set:])
            self.data = tf.Variable(self.data[:-size_test_set])
            self.labels = self.labels[:-size_test_set]

        self.num_data = tf.shape(self.data)[0]
        self.labels = tf.Variable(self.labels, dtype=tf.float32)
        self.batchsize = tf.Variable(batchsize, dtype=tf.int32)
        self.use_own_batch_per_sample = use_own_batch_per_sample
        self.last_start = tf.Variable(0, dtype=tf.int32)

    def shuffle_data(self):
        data, labels = tf.split(
             tf.random.shuffle(tf.concat((self.data, tf.cast(self.labels, tf.float32)), 1)),
            [self.num_dimensions, 1], axis=1)
        self.data.assign(data)
        self.labels.assign(labels)

    def likelihood_batch(self, x, data, labels):
        features = -tf.matmul(data, tf.transpose(x))
        log_likelihoods = tf.reduce_mean(tf.where(labels == 1, tf.math.log_sigmoid(features),
                                                  tf.math.log_sigmoid(features) - features), axis=0)
        return log_likelihoods

    def log_density_fb(self, x):
        """ Evaluate the log-density on the full data set (used for evaluation). If size_test_set=0, this function
        is equivalent to
        :py:meth:`gmmvi.experiments.target_distributions.logistic_regression.LogisticRegression.log_density`.
        """
        return LogisticRegression.log_density(self, x)

    def log_density(self, x):
        self.shuffle_data()
        if self.use_own_batch_per_sample:
            log_likelihoods = tf.TensorArray(tf.float32, size=tf.shape(x)[0])
            start = 0
            for i in tf.range(tf.shape(x)[0]):
                if start + self.batchsize > self.num_data:
                    start = 0
                indices = tf.slice(tf.range(self.num_data), [start], [self.batchsize])
                start = start + self.batchsize
                log_likelihoods = log_likelihoods.write(i, self.likelihood_batch(tf.expand_dims(x[i], axis=0), tf.gather(self.data, indices),
                                                                                 tf.gather(self.labels, indices)))
            log_likelihoods = log_likelihoods.concat()
        else:
            indices = tf.slice(tf.range(self.num_data), [0], [self.batchsize])
            log_likelihoods = self.likelihood_batch(x, tf.gather(self.data, indices), tf.gather(self.labels, indices))

        log_prior = tf.reduce_sum(-tf.math.log(self.prior_std) - 0.5 * tf.math.log(2. * pi) - 0.5 * tf.math.square((x - self.prior_mean) / self.prior_std), axis=1)
        log_posterior = tf.cast(self.num_data, tf.float32) * log_likelihoods + log_prior
        return log_posterior

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ As target-distribution specific metric, we estimate the full-batch ELBO.

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model and that are used for estimating the full-batch ELBO.

        Returns:
            dict: a dictionary with a single item containing the full-batch elbo.
        """
        expensive_metrics = dict()
        entropy = -tf.reduce_mean(model.log_density(samples))
        mean_reward = tf.reduce_mean(self.log_density_fb(samples))
        elbo_fb = mean_reward + entropy
        expensive_metrics.update({"elbo_fb:": elbo_fb})
        return expensive_metrics

def make_breast_cancer():
    return LogisticRegression("breast_cancer")

def make_german_credit():
    return LogisticRegression("german_credit")

def make_breast_cancer_mb(batch_size, size_test_set, use_own_batch_per_sample):
    return LogisticRegression_minibatch("breast_cancer", batch_size, size_test_set, use_own_batch_per_sample)

def make_german_credit_mb(batch_size, size_test_set, use_own_batch_per_sample):
    return LogisticRegression_minibatch("german_credit", batch_size, size_test_set, use_own_batch_per_sample)
