import logging
import os
from time import time

import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import numpy as np
import matplotlib

from gmmvi.experiments.evaluation.mmd import MMD
from gmmvi.optimization.gmmvi import GMMVI
from gmmvi.experiments.setup_experiment import init_experiment

matplotlib.use("Agg")

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

class GmmviRunner:
    """ This class runs :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>`, but also evaluates learning metrics and
    provides logging functionality.

    Parameters:
        config: dict
            A dictionary containing the hyperparameters and environment specifications.

        log_metrics_interval: int
            metrics that take non-negligible overhead are evaluated ever *log_metrics_interval* iterations.
    """
    def __init__(self, config, log_metrics_interval):
        if "seed" not in config.keys():
            config["seed"] = config["start_seed"]

        tf.keras.utils.set_random_seed(config["seed"])
        self.wall_times = []
        self.config = config
        self.log_metrics_interval = log_metrics_interval
        target_distribution, initial_model = init_experiment(self.config)
        self.gmmvi = GMMVI.build_from_config(self.config, target_distribution, initial_model)

        if "mmd_evaluation_config" in config.keys():
            print("building graph for MMD evaluation... (this may take a while)")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            samples = np.load(os.path.join(dir_path, config['mmd_evaluation_config']['sample_dir']))
            self.mmd = MMD(samples, config['mmd_evaluation_config']["alpha"])
            # Ensure that the TF graph for compute_MMD() is built during initialization:
            mmd = self.mmd.compute_MMD(self.gmmvi.model.sample(2000)[0])
            print("done")
        else:
            self.mmd = None

        if "dump_gmm_path" not in self.config:
            self.dump_gmms = False
        else:
            self.dump_gmms = True
            self.dump_gmm_path = os.path.join(self.config["dump_gmm_path"], str(time()))
            os.makedirs(self.dump_gmm_path, exist_ok=True)

    @staticmethod
    def build_from_config(config: dict):
        """Create a :py:class:`GMMVI<gmmvi.gmmvi_runner.GmmviRunner>` instance from a configuration dictionary.

        This static method provides a convenient way to create a :py:class:`GMMVI<gmmvi.gmmvi_runner.GmmviRunner>`
        instance, based on a dictionary containing the types and parameters of the
        :py:mod:`GMMVI modules<gmmvi.optimization.gmmvi_modules>`.

        Parameters:
            config: dict
                The dictionary should contain for each :py:mod:`GMMVI module<gmmvi.optimization.gmmvi_modules>`
                an entry of the form XXX_type (a string) and XXX_config (a dict) for specifying the type of each module,
                and the module-specific hyperparameters.
                For example, the dictionary could contain sample_selector_type={"component-based"} and
                sample_selector_config={"desired_samples_per_component": 100, "ratio_reused_samples_to_desired": 2.}.
                Refer to the example yml-configs, or to the individual GMMVI module for the expected parameters, and
                type-strings.
        """
        return GmmviRunner(config=config, **config['gmmvi_runner_config'])

    @tf.function
    def get_samples_and_entropy(self, num_samples):
        """ Draws *num_samples* from the model and uses them to estimate the model's entropy.

        Parameters:
            num_samples: int
                Number of samples to be drawn

        Returns:
            tuple(tf.Tensor, float):

            **test_samples** - The drawn samples

            **entropy** - MC estimate of the entropy
        """
        test_samples = self.gmmvi.model.sample(num_samples)[0]
        entropy = -tf.reduce_mean(self.gmmvi.model.log_density(test_samples))
        return test_samples, entropy

    def get_cheap_metrics(self):
        """ Returns a dictionary of 'cheap' metrics, e.g. the current number of components, that we can obtain
        after every iteration without adding computational overhead.

        Returns:
            dict: A dictionary containing cheap metrics.
        """
        cheap_metrics = dict()
        num_samples = self.gmmvi.sample_db.num_samples_written.numpy()
        cheap_metrics.update({"num_samples": num_samples,
                              "num_components": self.gmmvi.model.num_components,
                              "max_weight": tf.reduce_max(self.gmmvi.model.weights),
                              "num_db_samples": tf.shape(self.gmmvi.sample_db.samples)[0],
                              "num_db_components": tf.shape(self.gmmvi.sample_db.means)[0],
                              })
        return cheap_metrics

    def get_expensive_metrics(self):
        """
        Computes 'expensive' metrics, such as plots, test-set evaluations, etc.
        Some of these metrics can be task-specific (see
        :py:meth:`LNPDF.expensive_metrics()<gmmvi.experiments.target_distributions.lnpdf.LNPDF.expensive_metrics>`).

        Returns:
            dict: A dictionary containing expensive metrics.

        """
        expensive_metrics = dict()

        test_samples, entropy = self.get_samples_and_entropy(2000)
        mean_reward = tf.reduce_mean(self.gmmvi.sample_selector.target_uld(test_samples))
        elbo = mean_reward + self.gmmvi.temperature * entropy
        expensive_metrics.update({"-elbo": -elbo, "entropy": entropy, "target_density": mean_reward,
                                  "algo_time": np.sum(self.wall_times)})

        expensive_metrics.update(self.gmmvi.sample_selector.target_distribution.expensive_metrics(self.gmmvi.model,
                                                                                                  test_samples))

        if self.mmd is not None:
            mmd = self.mmd.compute_MMD(test_samples)
            expensive_metrics.update({"MMD:": mmd})

        return expensive_metrics

    def iterate_and_log(self, n: int) -> dict:
        """ Perform one learning iteration and computes and logs metrics.

        Parameters:
            n: int
                The current iteration

        Returns:
            dict: A dictionary containing metrics and plots that we want to log.
        """
        output_dict = {}

        ts1 = time()
        self.gmmvi.train_iter()
        ts2 = time()
        wall_time = ts2 - ts1
        output_dict.update({"walltime": wall_time})
        self.wall_times.append(ts2 - ts1)

        output_dict.update(self.get_cheap_metrics())

        # get metrics
        if n % self.log_metrics_interval == 0:
            eval_dict = self.get_expensive_metrics()
            print("Checkpoint {:3d} | FEVALS: {:10d} | avg. sample logpdf: {:05.05f} | ELBO: {:05.05f}".format(
                n, output_dict["num_samples"], eval_dict["target_density"], -eval_dict["-elbo"]))
            print(f"{self.gmmvi.model.num_components} components\n")
            output_dict.update(eval_dict)

        return output_dict

    def log_to_disk(self, n: int):
        """
        Saves the model parameters to the hard drive

        Parameters:
            n: int
                The current iteration
        """
        if self.dump_gmms:
            if n < 100 or n % 50 == 0:
                np.savez(self.dump_gmm_path + '/gmm_dump_' + str("%01d" % n) + '.npz',
                         weights=np.exp(self.gmmvi.model.log_weights.numpy()), means=self.gmmvi.model.means.numpy(),
                         covs=self.gmmvi.model.covs.numpy(), timestamps=time(),
                         fevals=self.gmmvi.sample_db.num_samples_written.numpy())

    def finalize(self):
        """
        Can be called after learning. Saves the final model parameters to the hard drive.
        """
        if self.dump_gmms:
            np.savez(self.dump_gmm_path + '/final_gmm_dump.npz',
                     weights=self.gmmvi.model.weights.numpy(), means=self.gmmvi.model.means.numpy(),
                     covs=self.gmmvi.model.covs.numpy(), timestamps=time(),
                     fevals=self.gmmvi.sample_db.num_samples_written.numpy())
