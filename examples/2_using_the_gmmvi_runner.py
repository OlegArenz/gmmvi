import os
from gmmvi.gmmvi_runner import GmmviRunner
from gmmvi.configs import load_yaml

my_path = os.path.dirname(os.path.realpath(__file__))
config = load_yaml(os.path.join(my_path, "example_config.yml"))
gmmvi_runner = GmmviRunner.build_from_config(config)

for n in range(10001):
    gmmvi_runner.iterate_and_log(n)

