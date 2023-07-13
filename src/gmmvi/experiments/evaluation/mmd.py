import tensorflow as tf
import tensorflow_probability as tfp

class MMD:
    """This class can be used for computing the Maximum Mean Discrepancy :cite:p:`Gretton2012`.
    The MMD can be used to compute the discrepancy between a model sample and a groundtruth sample.

    Note that instantiating this object can be quite slow, but computing the MMD using
    :py:meth:`compute_MMD<gmmvi.experiments.evaluation.mmd.MMD.compute_MMD>` should be fast.

    Parameters:
        groundtruth: tf.Tensor
            The groundtruth sample of shape number_of_samples x dimension

        alpha: tf.float32
            A factor for scaling the diagonal bandwidth matrix (which is automatically chosen based on the groundtruth
            sample using the Median trick :cite:p:`Gretton2012`).
    """

    def __init__(self, groundtruth, alpha):
        self.groundtruth = tf.cast(tf.convert_to_tensor(groundtruth), tf.float32)
        self.num_groundtruth = len(groundtruth)
        self.sigma = self.compute_sigma()
        self.set_alpha(alpha)

    @tf.function
    def compute_sigma(self, max_points_for_median=1000):
        max_points = tf.cast(tf.math.minimum(max_points_for_median, len(self.groundtruth)), tf.float32)
        distances = tf.TensorArray(tf.float32, size=int(0.5 * (max_points * max_points + max_points)))
        index = 0
        for i in range(int(max_points)):
            for j in range(i, int(max_points)):
                distances = distances.write(index, tf.math.square(self.groundtruth[i] - self.groundtruth[j]))
                index += 1
        sigma = tf.linalg.diag(tfp.stats.percentile(distances.stack(), 50, axis=0))
        return sigma

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.float32)])
    def compute_ustat(self, sample, alpha):
        kernel = tf.linalg.inv(alpha * self.sigma)
        ustat2 = 0.
        for i in tf.range(tf.shape(sample)[0]):
            diff = sample[i] - sample
            ustat2 += tf.reduce_sum(tf.exp(-tf.reduce_sum(diff @ kernel * diff, axis=1)))
        return ustat2

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.float32)])
    def kernel_mix(self, sample, alpha):
        kernel = tf.linalg.inv(alpha * self.sigma)
        ustat3 = 0.
        for i in range(self.num_groundtruth):
            diff = self.groundtruth[i] - sample
            ustat3 += tf.reduce_sum(tf.exp(-tf.reduce_sum(diff @ kernel * diff, axis=1)))
        return ustat3

    def set_alpha(self, alpha):
        self._alpha = alpha
        self.ustat1 = self.compute_ustat(self.groundtruth, alpha)

    def compute_MMD(self, model_sample):
        """ Compute the MMD between the model_sample and the groundtruth data that was provided when instantiating this
        object.

        Parameters:
            model_sample: tf.Tensor
                The sample from the model of shape number_of_samples x dimension

        Returns:
            float: The MMD between model sample and groundtruth sample
        """
        num_1 = self.num_groundtruth
        num_2 = len(model_sample)
        MMD = self.ustat1/(num_1**2) \
              + self.compute_ustat(model_sample, self._alpha)/(num_2**2) \
              - 2*self.kernel_mix(model_sample, self._alpha)/(num_1*num_2)
        return MMD
