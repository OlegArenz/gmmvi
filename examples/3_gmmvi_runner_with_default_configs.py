from gmmvi.gmmvi_runner import GmmviRunner
import gmmvi.configs

# In this example, we will create the config for a GmmviRunner using default configs
# for a given Codename (we weill use SAMYROX) and an and an environment name
# (we will use GMM20).
# Let's first get the default config for SAMYROX
algorithm_config = gmmvi.configs.get_default_algorithm_config("SAMYROX")

# Internally, this loaded the yaml files in gmmvi/configs/module_configs corresponding
# to the chosen design choices and stored them in a single dict "algorithm_config".
# Note that these default values were chosen independently for every design choice,
# and, thus, may not always be sensible. For example, the initial_stepsize defined in
# gmmvi/configs/module_configs/component_stepsize_adaptation/improvement_based.yml
# (Codeletter "R") is suitable if the stepsize is treated as a trust-region
# (Codeletter "T"), but not if it directly corresponds to the stepsize
# (Codeletter "I" or "Y")! Hence, we will overwrite the stepsize to something more
# suitable for SAMYROX:
better_stepsize_config = {
   'initial_stepsize': 0.0001,
   'min_stepsize': 0.0001,
   'max_stepsize': 0.001
}
algorithm_config = gmmvi.configs.update_config(algorithm_config, better_stepsize_config)

# We will use a target distribution that was shipped with the framework, namely "gmm20":
environment_config = gmmvi.configs.get_default_experiment_config("gmm20")

# The last call searched configs/experiment_configs for a corresponding yml-file and found
# gmm20.yml and stored the config in the dictionary "environment_config". We now just need
# to merge both config files:
config = gmmvi.configs.update_config(algorithm_config, environment_config)

# Create the GmmviRunner and start optimizing.
gmmvi_runner = GmmviRunner.build_from_config(config=config)
for n in range(1500):
    gmmvi_runner.iterate_and_log(n)
