import numpy as np
import tensorflow as tf


class RegressionFunc:
    """ Base class for least-square regression

    Parameters:
        bias_entry: int
            index of the weight that corresponds to the constant offset (will not get regularized)
    """
    def __init__(self, bias_entry: int = None):
        self._bias_entry = bias_entry
        self._params = None
        self.o_std = None

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        if self._params is None:
            raise AssertionError("Model not trained yet")
        return self._feature_fn(inputs) @ self._params

    def _feature_fn(self, num_samples: int, x: tf.Tensor) -> tf.Tensor:
        """ Computes the design matrix for the given samples.

        Parameters:
            num_samples: int
                Number of samples contained in *x* (yes, we could get this from *x*)

            x: tf.Tensor
                the data / samples
        """
        raise NotImplementedError

    def fit(self, regularizer: float, num_samples: int, inputs: tf.Tensor,
            outputs: tf.Tensor, weights: tf.Tensor = None) -> tf.Tensor:
        """
        Compute the coefficients of the linear model.

        Parameters:
            regularizer: float
                ridge coefficient

            num_samples: int
                number of samples (could be obtained from *inputs*)

            inputs: tf.Tensor
                the data / samples

            outputs: tf.Tensor
                the targets / dependent variables

            weights: tf.Tensor or None
                (importance) weights for weighted least-squares

        Returns:
            tf.Tensor: the learned parameters of the linear model
        """
        if len(tf.shape(outputs)) > 1:
            outputs = tf.squeeze(outputs)
        features = self._feature_fn(num_samples, x=inputs)

        if weights is not None:
            if len(weights.shape) == 1:
                weights = tf.expand_dims(weights, 1)
            weighted_features = tf.transpose(weights * features)
        else:
            weighted_features = tf.transpose(features)
        # regression
        reg_mat = tf.eye(self.num_features) * regularizer
        #
        if self._bias_entry is not None:
            bias_index = tf.range(len(reg_mat))[self._bias_entry]
            reg_mat = tf.tensor_scatter_nd_update(reg_mat, [[bias_index, bias_index]], [0])
        params = tf.squeeze(tf.linalg.solve(weighted_features @ features + reg_mat,
                                            weighted_features @ tf.expand_dims(outputs, 1)))
        return params


class LinFunc(RegressionFunc):
    """ This class can be used to learn a function that is linear in the inputs.

    Parameters:
        reg_fact: float
            coefficient for ridge regularization
    """
    def __init__(self, reg_fact: float):
        super().__init__(reg_fact, -1)

    def _feature_fn(self, num_samples: int, dim: int, x: tf.Tensor):
        return tf.concat([x, tf.ones([x.shape[0], 1], dtype=x.dtype)], 1)


class QuadFunc(RegressionFunc):
    """ This class can be used to learn a function that is quadratic in the inputs
    (or linear in the quadratic features).
    The approximation takes the form: :math:`x^T R x + x^T r + r_0`

    Parameters:
        dim: int
            the dimension of x
    """

    def __init__(self, dim: int):
        super().__init__(bias_entry=-1)
        self.dim = dim
        self.quad_term = None
        self.lin_term = None
        self.const_term = None
        self.num_quad_features = int(tf.floor(0.5 * (self.dim + 1) * self.dim))
        self.num_features = self.num_quad_features + self.dim + 1
        self.triu_idx = tf.constant(tf.transpose(np.stack(np.where(np.triu(np.ones([dim, dim], np.bool))))))

    def _feature_fn(self, num_samples: int, x: tf.Tensor) -> tf.Tensor:
        linear_features = x
        constant_feature = tf.ones((len(x), 1))

        # quad features
        quad_features = tf.zeros((num_samples, 0))
        for i in range(self.dim):
            quad_features = tf.concat((quad_features, tf.expand_dims(x[:, i], axis=1) * x[:, i:]), axis=1)

        # stack quadratic features, linear features and constant features
        features = tf.concat((quad_features, linear_features, constant_feature), axis=1)
        return features

    def fit_quadratic(self, regularizer: float, num_samples: int, inputs: tf.Tensor, outputs: tf.Tensor,
                      weights: tf.Tensor = None, sample_mean: tf.Tensor = None, sample_chol_cov: tf.Tensor = None)\
            -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Fits the quadratic model.

        Parameters:
            regularizer: float
                Coefficient for ridge regression

            num_samples: int
                Number of input samples (we could get this from *inputs*)

            inputs: tf.Tensor
                A two-dimensional tensor containing the inputs x.

            outputs: tf.Tensor
                A one-dimensional tensor containing the targets / dependant variables.

            weights: tf.Tensor
                (importance) weights used for weighted least-squares

            sample_mean: tf.Tensor
                Mean of the Gaussian distribution that sampled the input (used for whitening)

            sample_chol_cov: tf.Tensor
                Cholesky matrix of the Gaussian distribution that sampled the input (used for whitening)

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor):

            **quad_term** - the matrix :math:`R`

            **lin_term** - the vector :math:`r`

            **const_term** - the scalar bias

        """
        whitening = True
        if sample_mean is None:
            assert sample_chol_cov is None
        if sample_chol_cov is None:
            assert sample_mean is None

        # whithening
        if whitening and sample_mean is not None and sample_chol_cov is not None:
            inv_samples_chol_cov = tf.linalg.inv(sample_chol_cov)
            inputs = (inputs - sample_mean) @ tf.transpose(inv_samples_chol_cov)

        params = super().fit(regularizer, num_samples, inputs, outputs, weights)

        qt = tf.scatter_nd(self.triu_idx, params[:- (self.dim + 1)], [self.dim, self.dim])

        quad_term = - qt - tf.transpose(qt)
        lin_term = params[-(self.dim + 1):-1]
        const_term = params[-1]

        # unwhitening:
        if whitening and sample_mean is not None and sample_chol_cov is not None:
            quad_term = tf.transpose(inv_samples_chol_cov) @ quad_term @ inv_samples_chol_cov
            t1 = tf.linalg.matvec(tf.transpose(inv_samples_chol_cov), lin_term)
            t2 = tf.linalg.matvec(quad_term, sample_mean)
            lin_term = t1 + t2
            const_term += tf.reduce_sum(sample_mean * (-0.5 * t2 - t1))

        return quad_term, lin_term, const_term
