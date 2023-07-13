import os
import matplotlib.pyplot as plt

from gmmvi.gmmvi_runner import GmmviRunner
from gmmvi.configs import update_config, get_default_experiment_config, get_default_algorithm_config

def create_marginal_plot(seed, figure_path):
    os.makedirs(figure_path, exist_ok=True)

    # This script runs the 20D Mixture of Student-T experiment ("stm20") with the SAMTRON-design-choices, using
    # the same hyperparameters that were used in the paper. This script was used for creating
    # the plots for comparing the marginals of the learned model and the target distribution.
    algorithm_config = get_default_algorithm_config("SAMTRON")
    environment_config = get_default_experiment_config("stm20")
    environment_config = update_config(environment_config, {"start_seed": seed})
    used_hyperparameters = {
        "num_component_adapter_config": {"del_iters": 100, "add_iters": 60},
        "component_stepsize_adapter_config": {"initial_stepsize": 0.1, "min_stepsize": 0.001, "max_stepsize": 1.},
        "sample_selector_config": {"desired_samples_per_component": 200, "ratio_reused_samples_to_desired": 0.},
        "weight_stepsize_adapter_config": {"initial_stepsize": 1},
        "model_initialization": {"num_initial_components": 45},
        "gmmvi_runner_config": {"log_metrics_interval": 100}
    }

    algorithm_config = update_config(algorithm_config, used_hyperparameters)
    config = update_config(environment_config, algorithm_config)

    # Create the GmmviRunner and start optimizing, save the marginals as pdf.
    gmmvi_runner = GmmviRunner.build_from_config(config=config)
    for n in range(1501):
        metrics = gmmvi_runner.iterate_and_log(n)
        if 'marginals' in metrics:
            plt.figure(metrics['marginals'])
            plt.savefig(os.path.join(figure_path, f"marginals_{n}.pdf"))

if __name__ == "__main__":
    create_marginal_plot(0, "figures")
