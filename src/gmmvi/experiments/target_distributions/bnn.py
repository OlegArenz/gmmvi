import numpy as np
import os
from math import pi

import tensorflow as tf
import tensorflow_datasets as tfds

from gmmvi.experiments.target_distributions.lnpdf import LNPDF
from gmmvi.models.gmm_wrapper import GmmWrapper


def create_MNIST_splits(num_seeds):
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    for seed in range(num_seeds):
        tf.keras.utils.set_random_seed(seed)
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  #  return ds_train, ds_test, 784, 10


def create_WINE_splits(num_seeds):
    my_path = os.path.dirname(os.path.realpath(__file__))

    for i in range(num_seeds):
        tf.keras.utils.set_random_seed(i)
        DATASET_SIZE = 4898
        train_size = int(DATASET_SIZE * 0.60)
        test_size = int(DATASET_SIZE * 0.20)
        vali_size = DATASET_SIZE - train_size - test_size
        dataset = tfds.load(name="wine_quality", as_supervised=True, split="train").shuffle(DATASET_SIZE)
        features, labels = dataset.batch(DATASET_SIZE).get_single_element()
        feature_mat = tf.transpose(tf.stack([tf.reshape(tf.cast(a, tf.float32), [-1]) for a in features.values()]))
        feature_mean = tf.reduce_mean(feature_mat, axis=0)
        feature_mat -= feature_mean
        feature_std = tf.math.reduce_std(feature_mat, axis=0)
        feature_mat /= feature_std
        features_train = feature_mat[:train_size]
        features_test = feature_mat[train_size:train_size + test_size]
        features_vali = feature_mat[train_size + test_size:]
        labels_train = labels[:train_size]
        labels_test = labels[train_size:train_size + test_size]
        labels_vali = labels[train_size + test_size:]
        os.makedirs(os.path.join(my_path, "datasets", "wine"), exist_ok=True)
        np.savez(os.path.join(my_path, "datasets", "wine", f"wine_seed_{i}.npz"),
                 features_train=features_train, features_vali=features_vali, features_test=features_test,
                 labels_train=labels_train, labels_vali=labels_vali, labels_test=labels_test)

class BNN_LNPDF(LNPDF):
    """This class is used for implementing the target distribution given by the posterior for a Bayesian Neural Network.

    Parameters:
        likelihood_scaling: float
            a coefficient that can be used to scale the effect of the likelihood

        dataset_seed: int
            The dataset_seed is used for reproducible train/test-splits

        prior_std: float
            The standard deviation of the (zero-mean) prior over the network weights

        batch_size: int
            size of the minibatches

        hidden_units: list[int]
            The length of the list defines the number of hidden layers, the entries define their width

        loss: a tf.Keras.losses
            The loss function used for computing the log-likelihood

        activations: a list of Tensorflow activation functions
            activations for each hidden layer and the output layer


    """
    def __init__(self, likelihood_scaling, dataset_seed, prior_std, batch_size, hidden_units, loss, activations):
        super(BNN_LNPDF, self).__init__(use_log_density_and_grad=True)
        self.dataset_seed = dataset_seed
        self.likelihood_scaling = tf.Variable(likelihood_scaling, dtype=tf.float32)
        self.hidden_units = hidden_units
        self.activations = activations
        self.batch_size = batch_size
        ds_train, ds_test, ds_vali, self.input_dim, self.output_dim = self.prepare_data()
        self.train_size = len(ds_train)
        self.ds_train = ds_train.cache()\
            .repeat()\
            .shuffle(len(ds_train), reshuffle_each_iteration=True)\
            .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, drop_remainder=True)\
            .prefetch(tf.data.AUTOTUNE)

        self.test_size = len(ds_test)
        self.ds_test = ds_test.cache()\
                .shuffle(len(ds_test), reshuffle_each_iteration=False)\
                .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, drop_remainder=False)\
                .prefetch(tf.data.AUTOTUNE)

        self.vali_size = len(ds_vali)
        self.ds_vali = ds_vali.cache()\
                .shuffle(len(ds_vali), reshuffle_each_iteration=False)\
                .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, drop_remainder=False)\
                .prefetch(tf.data.AUTOTUNE)

        self.loss = loss #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        last_layer_width = self.input_dim
        self.layer_size = []
        self.layer_shape = []
        for width in self.hidden_units:
            self.layer_shape.append([last_layer_width, width]) # weights
            self.layer_size.append(last_layer_width * width) # weights
            self.layer_shape.append([width]) # bias
            self.layer_size.append(width) # bias
            last_layer_width = width

        self.layer_shape.append([last_layer_width, self.output_dim])
        self.layer_size.append(last_layer_width * self.output_dim)
        self.layer_shape.append([self.output_dim])
        self.layer_size.append(self.output_dim)

        self._num_dimensions = tf.reduce_sum(self.layer_size)

        self._prior_std = prior_std * tf.ones(self._num_dimensions)  # zero-mean prior is implicitly assumed
        self.prior_lnpdf_constant_part = - 0.5 * tf.cast(self._num_dimensions, dtype=tf.float32) * tf.math.log(2 * pi) \
                                         - tf.reduce_sum(tf.math.log(self._prior_std))

        self.model, self.metric = self.create_model()

    def prepare_data(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def get_num_dimensions(self):
        return self._num_dimensions

    @property
    def prior_std(self):
        return self._prior_std

    def forward_from_weight_vector(self, input, x):
        output = tf.reshape(input, [-1, self.input_dim])
        start = 0
        i = 0
        j = 0
        while i < len(self.layer_shape):
            W = tf.reshape(x[start:start+self.layer_size[i]], self.layer_shape[i])
            start += self.layer_size[i]
            i += 1

            b = tf.reshape(x[start:start+self.layer_size[i]], self.layer_shape[i])
            start += self.layer_size[i]
            i += 1
            output = self.activations[j](output @ W + b)
            j += 1
        return output

#    def set_weights(self, weights_as_vector):
#        start = 0
#        for i in range(len(self.layer_size)):
#            flat_layer = weights_as_vector[start:start + self.layer_size[i]]
#            start += self.layer_size[i]
#            self.model.trainable_variables[i].assign(tf.reshape(flat_layer, self.layer_shape[i]))

    def log_likelihood(self, x):
        lls = tf.TensorArray(size=tf.shape(x)[0], dtype=tf.float32)
        i = 0
        for features, labels in self.ds_train.take(tf.cast(tf.shape(x)[0], dtype=tf.int64)):
            output = self.forward_from_weight_vector(features, (x[i]))
            ll = - self.train_size * self.loss(labels, output)
            lls = lls.write(i, ll)
            i+= 1
        return lls.stack()

    def log_likelihood_and_grad(self, x):
        lls = tf.TensorArray(size=tf.shape(x)[0], dtype=tf.float32)
        ll_grads = tf.TensorArray(size=tf.shape(x)[0], dtype=tf.float32)
        i = 0
        for features, labels in self.ds_train.take(tf.cast(tf.shape(x)[0], dtype=tf.int64)):
            this_x = x[i]
            with tf.GradientTape() as tape:
                tape.watch(this_x)
                output = self.forward_from_weight_vector(features, this_x)
                ll = - self.train_size * self.loss(labels, output)
                lls = lls.write(i, ll)
                ll_grads = ll_grads.write(i, tape.gradient(ll, this_x))
            i+= 1
        return lls.stack(), ll_grads.stack()

    def log_likelihood_old(self, x):
        lls = tf.TensorArray(size=tf.shape(x)[0], dtype=tf.float32)
        for i in range(tf.shape(x)[0]):
            features, labels = self.ds_train.take(1).get_single_element()
            output = self.forward_from_weight_vector(features, (x[i]))
            ll = - self.train_size * self.loss(labels, output)
            lls = lls.write(i, ll)
        return lls.stack()

#    def weight_vector_to_network_parameters(self, x):
#        start = 0
#        weights_and_biases = []
#        for i in range(len(self.layer_shape)):
#            weights_and_biases.append(tf.reshape(x[start:start+self.layer_size[0]], self.layer_shape[0]))
#            start += self.layer_size[i]
#        return weights_and_biases

    def log_density(self, x):
        log_posterior = self.likelihood_scaling * (self.log_likelihood(x) + self.log_prior(x, ignore_constant=True))
        return log_posterior

    def log_density_and_grad(self, x: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        ll, ll_grad = self.log_likelihood_and_grad(x)
        log_prior, log_prior_grad = self.log_prior_and_grad(x, ignore_constant=True)
        log_posterior = self.likelihood_scaling * (ll + log_prior)
        log_posterior_grad = self.likelihood_scaling * (ll_grad + log_prior_grad)
        return log_posterior, log_posterior_grad

    def log_prior(self, x, ignore_constant=False):
        if ignore_constant:
            return - 0.5 * tf.reduce_sum(tf.square(x / self.prior_std), axis=1)
        else:
            return self.prior_lnpdf_constant_part - 0.5 * tf.reduce_sum(tf.square(x / self.prior_std), axis=1)

    def log_prior_and_grad(self, x, ignore_constant=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            log_prior = self.log_prior(x, ignore_constant=ignore_constant)
        log_prior_grad = tape.gradient(log_prior, x)
        return log_prior, log_prior_grad

    @tf.function
    def bayesian_inference_test_loss(self, x):
        features, labels = self.ds_test.take(1).get_single_element()
        output = self.forward_from_weight_vector(features, (x[0]))
        for weights in x:
            output += self.forward_from_weight_vector(features, weights)

        test_loss = self.loss(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
        self.accuracy.reset_state()
        self.accuracy.update_state(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
        return test_loss, self.accuracy.result()

  #  @tf.function
    def average_loss(self, x, dataset):
        all_losses = []
        if dataset == "test":
            ds = self.ds_test
        elif dataset == "vali":
            ds = self.ds_vali
        test_loss = 0.
        for params in x:
            this_test_loss = 0.
            num_batches = 0
            for features, labels in ds:
                output = self.forward_from_weight_vector(features, params)
                this_loss = self.loss(labels, output)
                this_test_loss += this_loss
                all_losses.append(this_loss)
                num_batches += 1
            test_loss += this_test_loss / num_batches
        return test_loss / tf.cast(tf.shape(x)[0], tf.float32), tf.stack(all_losses)

    @tf.function
    def avg_bayesian_inference_test_loss(self, x, num_batches):
        test_loss = 0.
        test_metric = 0.
        for _ in tf.range(num_batches):
            features, labels = self.ds_test.take(1).get_single_element()
            output = self.forward_from_weight_vector(features, (x[0]))
            for i in tf.range(1, tf.shape(x)[0]):
                output += self.forward_from_weight_vector(features, (x[i]))
            test_loss += self.loss(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
            self.metric.reset_state()
            self.metric.update_state(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
            test_metric += self.metric.result()
        return test_loss / num_batches, test_metric / num_batches


    @tf.function
    def avg_bayesian_inference_loss(self, x, dataset):
        if dataset == "train":
            ds = self.ds_train
        elif dataset == "test":
            ds = self.ds_test
        elif dataset == "vali":
            ds = self.ds_vali
        loss = 0.
        metric = 0.
        num_batches = 0.
        for features, labels in ds:
      #      features, labels = ds.take(1).get_single_element()
            output = self.forward_from_weight_vector(features, (x[0]))
            for i in tf.range(1, tf.shape(x)[0]):
                output += self.forward_from_weight_vector(features, (x[i]))
            loss += self.loss(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
            self.metric.reset_state()
            self.metric.update_state(labels, output / tf.cast(tf.shape(x)[0], tf.float32))
            metric += self.metric.result()
            num_batches += 1.
        return loss / num_batches, metric / num_batches

class BNN_MNIST(BNN_LNPDF):
    def __init__(self, likelihood_scaling, prior_std, batch_size):
        super(BNN_MNIST, self).__init__(likelihood_scaling=likelihood_scaling, prior_std=prior_std,
                                        batch_size=batch_size, dataset_seed=-1,
                                      hidden_units = [128],
                                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      activations = [tf.nn.relu, tf.keras.activations.linear])

    def prepare_data(self):
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label


        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        return ds_train, ds_test.take(5000), ds_test.skip(5000), 784, 10

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=self.loss,
            metrics=[accuracy],
        )
        return model, accuracy

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ This method computes four task-specific metric:

        1. bi_test_loss: Expected loss on the test set when using Bayesian inference

        2. bi_test_accuracy: Expected accuracy on the test set when using Bayesian inference

        3. bi_vali_loss: Expected loss on the validation set when using Bayesian inference

        4. bi_vali_accuracy: Expected accuracy on the validation set when using Bayesian inference

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing the four task-specific metrics
        """

        expensive_metrics = dict()
        bi_test_loss, bi_test_accuracy = self.avg_bayesian_inference_loss(samples, "test")
        expensive_metrics.update({"bi_test_loss": bi_test_loss, "bi_test_accuracy": bi_test_accuracy})
        bi_vali_loss, bi_vali_accuracy = self.avg_bayesian_inference_loss(samples, "vali")
        expensive_metrics.update({"bi_vali_loss": bi_vali_loss, "bi_vali_accuracy": bi_vali_accuracy})
        return expensive_metrics

def make_MNIST_target(likelihood_scaling, prior_std, batch_size):
    return BNN_MNIST(likelihood_scaling=likelihood_scaling, prior_std=prior_std, batch_size=batch_size)

class BNN_WINE(BNN_LNPDF):
    def __init__(self, dataset_seed, likelihood_scaling, prior_std, batch_size):
        super(BNN_WINE, self).__init__(dataset_seed=dataset_seed, likelihood_scaling=likelihood_scaling,
                                       prior_std=prior_std, batch_size=batch_size, hidden_units=[8, 8],
                                       loss=tf.keras.losses.MeanSquaredError(),
                                       activations=[tf.math.sigmoid, tf.math.sigmoid, tf.keras.activations.linear])

    def prepare_data(self):
        dataset_seed = self.dataset_seed % 10
        print(f"using dataset seed: {dataset_seed}")
        my_path = os.path.dirname(os.path.realpath(__file__))
        dataset = np.load(os.path.join(my_path, "datasets", "wine", f"wine_seed_{dataset_seed}.npz"))
        ds_train = tf.data.Dataset.from_tensor_slices((dataset["features_train"], dataset["labels_train"]))
        ds_test = tf.data.Dataset.from_tensor_slices((dataset["features_test"], dataset["labels_test"]))
        ds_vali = tf.data.Dataset.from_tensor_slices((dataset["features_vali"], dataset["labels_vali"]))
        return ds_train, ds_test, ds_vali, 11, 1

    def create_model(self):
        inputs = tf.keras.Input(shape=[11], dtype=tf.float32)
        features = inputs
        for units in self.hidden_units:
            features = tf.keras.layers.Dense(units, activation="sigmoid")(features)
        outputs = tf.keras.layers.Dense(units=1)(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        rmse = tf.keras.metrics.RootMeanSquaredError()
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=lambda y_true, y_pred: self.loss(y_true, y_pred),
            metrics=[rmse],
        )
        return model, rmse

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ This method computes four task-specific metric:

        1. bi_test_loss: Expected loss on the test set when using Bayesian inference

        2. bi_test_accuracy: Expected accuracy on the test set when using Bayesian inference

        3. bi_vali_loss: Expected loss on the validation set when using Bayesian inference

        4. bi_vali_accuracy: Expected accuracy on the validation set when using Bayesian inference


        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing the four task-specific metrics
        """
        expensive_metrics = dict()
        bi_test_loss, bi_test_accuracy = self.avg_bayesian_inference_loss(samples, "test")
        expensive_metrics.update({"bi_test_loss": bi_test_loss, "bi_test_accuracy": bi_test_accuracy})
        bi_vali_loss, bi_vali_accuracy = self.avg_bayesian_inference_loss(samples, "vali")
        expensive_metrics.update({"bi_vali_loss": bi_vali_loss, "bi_vali_rmse": bi_vali_accuracy})
        return expensive_metrics

def make_WINE_target(likelihood_scaling, dataset_seed, prior_std, batch_size):
    return BNN_WINE(likelihood_scaling=likelihood_scaling, dataset_seed=dataset_seed,
                    prior_std=prior_std, batch_size=batch_size)

if __name__ == "__main__":
    ds0_test_losses = []
    for seed in range(10):
        tf.keras.utils.set_random_seed(seed)
      #  test = make_MNIST_target(1., 1., 128)
        test = make_WINE_target(likelihood_scaling=1., dataset_seed=seed, prior_std=1., batch_size=128)
        best_vali_loss = tf.float32.max
        test_losses = []
        for i in range(2000):
            test.model.fit(
                test.ds_train,
                epochs=1,
                verbose=0,
                steps_per_epoch=test.train_size // test.batch_size,
                validation_data=test.ds_vali,
            )
            params = tf.concat([tf.reshape(x, [-1]) for x in test.model.trainable_variables], axis=0)
            vali_loss = test.average_loss(tf.reshape(params, [1,-1]), "vali")[0]
            if vali_loss < best_vali_loss:
                test_loss = test.average_loss(tf.reshape(params, [1, -1]), "test")[0]
             #   if test_loss < 0.41:
             #       print("debug")
                test_losses.append(test_loss)
                print(f"iter: {i}, new best params, test_loss: {test_loss}, vali_loss: {vali_loss}")
                best_params = params
                best_vali_loss = vali_loss
        print(f"run {seed}, test_loss {tf.stack(test_losses)[-1]}")
        ds0_test_losses.append(tf.stack(test_losses))

    params = tf.concat([tf.reshape(x, [-1]) for x in test.model.trainable_variables], axis=0)
    print("done")
