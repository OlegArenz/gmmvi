import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from gmmvi.models.gmm_wrapper import GmmWrapper

tfd = tfp.distributions
from collections import OrderedDict
from tf_robot_learning import kinematic as tk
from tf_robot_learning import distributions as rlds
from gmmvi.experiments.target_distributions.lnpdf import LNPDF
from pathlib import Path
import os
import numpy as np

class Talos(LNPDF):
    """
    The target distribution for the humanoid inverse kinematik problem. The unnormalized target distribution is given
    by a mixture of experts, that penalize the task space error of the left endeffector, joint-limit violations,
    unstable configurations, etc.

    Parameters:
        context: list[float]
            A list of three floats, specifying the desired x, y, z coordinates of the left endeffector.
    """

    def __init__(self, context):
        super().__init__(use_log_density_and_grad=True, safe_for_tf_graph=False)
        self.context = context
        root_path = Path(__file__).parent.resolve()
        self.urdf_path = os.path.join(root_path, "datasets/talos_reduced.urdf")
        self.chain_names = ['r_gripper', 'l_gripper', 'r_foot', 'l_foot']
        self.left_foot_target = [-0.02, 0.09, -0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]
        self.right_foot_target = [-0.02, -0.09, -0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]
        self.chain = self.create_chain()
        # tf.print(self.chain.mean_pose)
        # tf.print(self.chain.actuated_joint_names)
        self.mean_joints = tf.constant(self.chain.mean_pose + [0., 0., 1.08, 0., 0., 0.], dtype=tf.float32)
        self._num_dimensions = 34
        _ = self.log_density_and_grad(tf.random.normal((10, self._num_dimensions)))
        self.log_density_and_grad = tf.function(self.log_density_and_grad)

    def log_density_and_grad(self, samples):
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(samples)
            target = self.log_density(samples)
        gradient = tape.gradient(target, samples)
        return target, gradient

    def create_chain(self):  # cannot be created just once, needs to be recreated in each tf graph
        urdf = tk.urdf_from_file(self.urdf_path)
        tips = OrderedDict({
            'r_gripper': 'gripper_right_base_link',
            'l_gripper': 'gripper_left_base_link',
            'r_foot': 'right_sole_link',
            'l_foot': 'left_sole_link',
        })

        # get chains of arms and leg from base
        chain = tk.ChainDict({
            name: tk.kdl_chain_from_urdf_model(urdf, 'base_link', tip=tip)
            for name, tip in tips.items()
        })
        _ = chain.actuated_joint_names
        return chain

    def get_left_gripper(self, x, chain):  # l_gripper
        _q, _p, _m = param_to_joint_pos(x, chain)
        return chain.xs(_q, floating_base=(_p, _m), name='l_gripper')[:, -1]

    def get_right_foot(self, x, chain):  # r_foot
        _q, _p, _m = param_to_joint_pos(x, chain)
        return chain.xs(_q, floating_base=(_p, _m), name='r_foot')[:, -1]

    def get_left_foot(self, x, chain):  # l_foot
        _q, _p, _m = param_to_joint_pos(x, chain)
        return chain.xs(_q, floating_base=(_p, _m), name='l_foot')[:, -1]

    def get_joint_angles(self, x, chain):  # joint angles
        return x[..., :chain.nb_joint]

    def get_com_feet_diff(self, x, chain):  # relative position between center of feet and projection of com
        # basic implementation, needs to be replaced with more complete differentiable formula
        _q, _p, _m = param_to_joint_pos(x, chain)
        # get center of mass projection on the ground
        com_xy = chain.xs(_q, floating_base=(_p, _m), get_links=True)[-1][:, :2]

        center_feet = chain.xs(_q, floating_base=(_p, _m), name='l_foot')[:, -1, :2]

        return com_xy - center_feet

    def get_joint_prior(self, x, chain):
        joint_limits = tf.constant(chain.joint_limits, dtype=tf.float32)
        joint_limits_std = 0.05
        joint_limits_temp = 1.

        joint_limits_exp = rlds.SoftUniformNormalCdf(
            low=joint_limits[:, 0],
            high=joint_limits[:, 1],
            std=joint_limits_std,
            temp=joint_limits_temp,
            reduce_axis=-1
        )
        return joint_limits_exp.log_prob(self.get_joint_angles(x, chain))

    def get_left_gripper_reward(self, x, context, chain):
        pos = context
        stds = tf.ones_like(pos) * 0.02
        target_dist = tfp.distributions.MultivariateNormalDiag(pos, stds)
        return target_dist.log_prob(self.get_left_gripper(x, chain)[:, :3])

    def get_left_foot_reward(self, x, context, chain):
        std = [0.02] * 3 + [0.1] * 9
        target_dist = tfp.distributions.MultivariateNormalDiag(self.left_foot_target, std)
        return target_dist.log_prob(self.get_left_foot(x, chain))

    def get_right_foot_reward(self, x, context, chain):
        std = [0.02] * 3 + [0.1] * 9
        target_dist = tfp.distributions.MultivariateNormalDiag(self.right_foot_target, std)
        return target_dist.log_prob(self.get_right_foot(x, chain))

    def get_com_reward(self, x, chain):
        com_limits = 0.14
        com_limits_std = 0.01
        com_limits_temp = 1.

        com_limits_exp = rlds.SoftUniformNormalCdf(
            low=-com_limits,
            high=com_limits,
            std=com_limits_std,
            temp=com_limits_temp,
            reduce_axis=-1
        )

        return com_limits_exp.log_prob(self.get_com_feet_diff(x, chain))

    def _reward(self, sample, context):
        chain = self.chain #self.create_chain()
        rew = self.get_joint_prior(sample, chain)
        rew += self.get_com_reward(sample, chain)
        rew += self.get_right_foot_reward(sample, context, chain)
        rew += self.get_left_foot_reward(sample, context, chain)
        rew += self.get_left_gripper_reward(sample, context, chain)
        return rew

    def get_num_dimensions(self):
        return self._num_dimensions

    def log_density(self, thetas):
        return self._reward(thetas, self.context)

    def expensive_metrics(self, model: GmmWrapper, samples: tf.Tensor) -> dict:
        expensive_metrics = dict()
        fig = self.plot(self.context, samples[:100])
        expensive_metrics.update(
            {"plot": fig})
        return expensive_metrics

    def plot(self, context, samples):
        chain = self.chain #create_chain()
        plt.ioff()
        plt.figure(1)
        plt.clf()
        _, samples_position, samples_rotation = param_to_joint_pos(samples, chain)
        links, _, coms = chain.xs(samples, floating_base=(samples_position, samples_rotation), get_links=True)
        fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(15, 10))

        for i in range(2):
            dim = [i, 2]
            chain.plot(
                links, feed_dict={}, ax=ax[i],
                dim=dim, alpha=0.2, color='k'
            )
            ax[i].plot(coms[:, dim[0]], coms[:, dim[1]], 'yx')

            ax[i].plot(self.left_foot_target[dim[0]], self.left_foot_target[dim[1]], ls=' ', marker='s',
                       label="left_foot_target", color="red")
            ax[i].plot(self.right_foot_target[dim[0]], self.right_foot_target[dim[1]], ls=' ', marker='s',
                       label="right_foot_target", color="green")
            ax[i].plot(context[dim[0]], context[dim[1]], ls=' ', marker="o", markersize=6,
                       label="left_gripper_target", color="orange")
            ax[i].legend()
        return fig


def param_to_joint_pos(x, chain):
    return (x[..., :chain.nb_joint],  # 28 joint angle
            x[..., chain.nb_joint:chain.nb_joint + 3],  # 3 position of floating base
            tk.rotation.rpy(x[..., -3:]))  # 3 orientation of floating base euler to rotation matrix


class TalosLeftGripperTargetPdf:

    def __init__(self):
        self.radius_prior = tfd.Uniform(low=0.2, high=1)
        self.xy_ang_prior = tfd.Uniform(low=-np.pi/2, high=np.pi/2)
        self.z_ang_prior = tfd.Uniform(low=float(np.deg2rad(10)), high=float(np.rad2deg(170)))

    def sample(self, n):
        xy_ang = self.xy_ang_prior.sample(n)
        rad = self.radius_prior.sample(n)
        z_ang = self.z_ang_prior.sample(n)

        x = rad * tf.sin(z_ang) * tf.cos(xy_ang)
        y = rad * tf.sin(z_ang) * tf.sin(xy_ang)
        z = rad * tf.cos(z_ang)
        return tf.stack([x, y, z], axis=-1)


def make_talos_target(context):
    """
    planar robot experiment with one goal from vips, here with conditioning on the goal position.

    Parameters:
        context: list[float]
            A list of three floats, specifying the desired x, y, z coordinates of the left endeffector.
    """
    target = Talos(context)
    goal_prior = TalosLeftGripperTargetPdf()

    return target

if __name__ == "__main__":
    target = make_talos_target([-0.02, 0.09, -0.0])
    target.log_density(tf.random.normal((2, target._num_dimensions)))
    print("done")


