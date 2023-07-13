import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gmmvi.experiments.evaluation.visualize_planar_robot import visualize_mixture
from gmmvi.experiments.target_distributions.lnpdf import LNPDF
from gmmvi.models.gmm_wrapper import GmmWrapper

tfd = tfp.distributions

import matplotlib.pyplot as plt

class PlanarRobot(LNPDF):
    """This class reimplements the "PlanarRobot" experiments used by :cite:t:`Arenz2020`.

    Parameters:
        num_links: int
            The number of links of the robot
        num_goals: int
            The number of goal positions, must be either 1 or 4
        prior_std: float
            The standard deviation of the (zero-mean) prior on the joint angles. The first value is ignored, as the
            first link always has a standard deviation of 1.
        likelihood_std: float
            The standard deviation used for penalizing the distance in X-Y between the robot endeffector and the goal
            position.
    """

    def __init__(self, num_links, num_goals, prior_std=2e-1, likelihood_std=1e-2):
        super(PlanarRobot, self).__init__(use_log_density_and_grad=False)
        self._num_dimensions = num_links
        prior_stds = prior_std * np.ones(num_links)
        prior_stds[0] = 1.
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(num_links), scale_diag=prior_stds.astype(np.float32))
        self.link_lengths = np.ones(self._num_dimensions)
        self._num_goals = num_goals
        if num_goals == 1:
            self.goals = tf.constant([[7., 0.]], dtype=tf.float32)
        elif num_goals == 4:
            self.goals = tf.constant([[7., 0.], [-7., 0.], [0., 7.], [0., -7.]], dtype=tf.float32)
        else:
            raise ValueError

        self.goal_Gaussians = []
        for goal in self.goals:
            self.goal_Gaussians.append(tfd.MultivariateNormalDiag(loc=goal, scale_diag=[likelihood_std, likelihood_std]))

    def likelihood(self, pos: tf.Tensor) -> tf.Tensor:
        likelihoods = tf.TensorArray(size=self._num_goals, dtype=tf.float32)
        for i in range(self._num_goals):
            likelihoods = likelihoods.write(i, self.goal_Gaussians[i].log_prob(pos))
        return tf.reduce_max(likelihoods.stack(), axis=0)

    def get_num_dimensions(self):
        return self._num_dimensions

    def forward_kinematics(self, theta):
        y = tf.zeros(tf.shape(theta)[0])
        x = tf.zeros(tf.shape(theta)[0])
        for i in range(0, self._num_dimensions):
            y += self.link_lengths[i] * tf.math.sin(tf.reduce_sum(theta[:, :i + 1], axis=1))
            x += self.link_lengths[i] * tf.math.cos(tf.reduce_sum(theta[:, :i + 1], axis=1))
        return tf.stack((x, y), axis=1)

    def log_density(self, theta):
        return self.prior.log_prob(theta) + self.likelihood(self.forward_kinematics(theta))

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        """ This method computes two task-specific metrics:

        1. The number of detected modes: This is course heuristic to count the different configurations used for reaching
           each of the goal positions (potentially misleading!)

        2. Plots of the mean configurations of the learned model

        Parameters:
            model: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
                The learned model that we want to evaluate for this target distribution.

            samples: tf.Tensor
                Samples that have been drawn from the model, which can be used for evaluations.

        Returns:
            dict: a dictionary containing two items (the number of detected modes, and a figure showing the mean
            configurations).
        """
        all_comp_colors = np.repeat("k", model.num_components)
        colors = ["b", "g", "r", "c", "m", "y"]
        next_color_idx = 0
        expensive_metrics = dict()

        for goal in self.goals:
            end_eff_error = tf.norm(
                self.forward_kinematics(model.means) - goal,
                axis=1)
            close_to_goal = (end_eff_error < 0.05)
            good_components = self.log_density(model.means) > -7.
            this_goal_indices = tf.reshape(tf.where(tf.reduce_all([close_to_goal, good_components], axis=0)), [-1])
            if tf.size(this_goal_indices) == 0:
                num_detected_modes = 0
            else:
                this_goal_first_links = tf.gather(model.means[:, 0], this_goal_indices)
                try:
                    first_angles_sorted_ind = tf.argsort(this_goal_first_links)
                except:
                    print("debug")
                first_angles_sorted = tf.gather(this_goal_first_links, first_angles_sorted_ind)
                gaps = first_angles_sorted[1:] - first_angles_sorted[:-1]
                comp_colors = np.repeat(colors[next_color_idx], tf.size(first_angles_sorted_ind))
                num_detected_modes = 1
                clusters = []
                splits = tf.reshape(tf.where(gaps > 0.4), [-1])
                start = 0
                for i in range(len(splits)):
                    clusters.append(np.mean(first_angles_sorted[start:splits[i] + 1]))
                    start = splits[i] + 1
                    num_detected_modes += 1
                    next_color_idx += 1
                    next_color_idx = next_color_idx % len(colors)
                    next_color = colors[next_color_idx]
                    comp_colors[splits[i]:] = next_color
                all_comp_colors[this_goal_indices.numpy()[first_angles_sorted_ind.numpy()]] = comp_colors
                clusters.append(np.mean(first_angles_sorted[start:]))
                print(f"clusters for goal at position [{goal[0]}, {goal[1]}]: {clusters}")

            expensive_metrics.update({f"num_detected_modes_{goal}": num_detected_modes})
        mixture_fig = plt.figure(1)
        visualize_mixture(model.weights.numpy(), model.means.numpy()) #, comp_colors=all_comp_colors)
        mixture_fig.tight_layout()
        expensive_metrics.update(
            {"mixture_plot": mixture_fig})
        return expensive_metrics

def make_single_goal():
    return PlanarRobot(10, 1)

def make_four_goal():
    return PlanarRobot(10, 4)
