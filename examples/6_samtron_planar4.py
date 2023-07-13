import os
import matplotlib.pyplot as plt

from gmmvi.gmmvi_runner import GmmviRunner
from gmmvi.configs import update_config, get_default_experiment_config, get_default_algorithm_config
from time import time

def create_mixture_plot(seed, codename, figure_path):
    os.makedirs(figure_path, exist_ok=True)

    # This script runs the PlanarRobot4 experiment with the SAMTRON- or SEPYFUX-design-choices, using
    # the respective hyperparameters that were used in the paper. This script was used for creating
    # the plots for comparing the learned robot configurations.

    environment_config = get_default_experiment_config("planar_robot_4")
    environment_config = update_config(environment_config, {"start_seed": seed})
    if codename.lower() == "samtron":
        algorithm_config = get_default_algorithm_config("SAMTRON")
        used_hyperparameters = {
            "num_component_adapter_config": {"del_iters": 10, "add_iters": 1},
            "component_stepsize_adapter_config": {"initial_stepsize": 0.1, "min_stepsize": 0.001, "max_stepsize": 1.},
            "sample_selector_config": {"desired_samples_per_component": 100, "ratio_reused_samples_to_desired": 0.},
            "weight_stepsize_adapter_config": {"initial_stepsize": 5},
            "model_initialization": {"num_initial_components": 100},
            "gmmvi_runner_config": {"log_metrics_interval": 10}
        }
    elif codename.lower() == "sepyfux":
        algorithm_config = get_default_algorithm_config("SEPYFUX")
        used_hyperparameters = {
            "component_stepsize_adapter_config": {"initial_stepsize": 0.001},
            "sample_selector_config": {"desired_samples_per_component": 300, "ratio_reused_samples_to_desired": 0.},
            "weight_stepsize_adapter_config": {"initial_stepsize": 0.0001},
            "model_initialization": {"num_initial_components": 100},
            "gmmvi_runner_config": {"log_metrics_interval": 100}
        }
    else:
        raise ValueError(f"unknown codename: {codename}")

    algorithm_config = update_config(algorithm_config, used_hyperparameters)
    config = update_config(environment_config, algorithm_config)

    # Create the GmmviRunner and start optimizing, save the marginals as pdf.
    gmmvi_runner = GmmviRunner.build_from_config(config=config)
    timelimit = time() + 1800 # half an hour
    n=0
    while time() < timelimit:
        metrics = gmmvi_runner.iterate_and_log(n)
        if 'mixture_plot' in metrics:
            plt.figure(metrics['mixture_plot'])
            plt.savefig(os.path.join(figure_path, f"mixture_plot_{codename}_{n}.pdf"))
        n += 1

if __name__ == "__main__":
    create_mixture_plot(0, "samtron", "figures")
