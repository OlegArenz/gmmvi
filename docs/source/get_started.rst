Get Started
===========

.. _installation:

Installation
------------

To install GMMVI (optionally) create a virtual environment and run 

.. code-block:: console

   (.venv) $ pip install .

.. _usage:

Usage
----------------
For performing the optimization, you can directly instantiate a :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>`
and run :py:meth:`GMMVI.train_iter()<gmmvi.optimization.gmmvi.GMMVI.train_iter>` in a loop, or, for adding basic logging
capability and easier integration, for example with WandB, you can instantiate a
:py:class:`GmmviRunner<gmmvi.gmmvi_runner.GmmviRunner>` and run
:py:meth:`GmmviRunner.iterate_and_log(n)<gmmvi.gmmvi_runner.GmmviRunner.iterate_and_log>` in a loop.

Directly Using GMMVI
~~~~~~~~~~~~~~~~~~~~
Before instantiating the :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>`, we need to create several other
objects, namely:

1. A :py:class:`wrapped model<gmmvi.models.gmm_wrapper.GmmWrapper>` which stores the parameters of the GMM, as well as
   component-specific meta-information (reward histories, learning-rates, etc.)

2. A :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>` for storing samples.

3. One object for each of the seven :ref:`Design Choices`.

Fortunately, each of these classes and also :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>` itself, have a
static method called build_from_config(), which allows to create
the object from a common config dictionary (which can be created from a YAML file). Using a common dictionary is
recommended, to ensure that the parameters passed to the different constructors are consistent (e.g. the sample
dimensions needs to be the same).

It is easiest to directly use :py:meth:`GMMVI.build_from_config<gmmvi.optimization.gmmvi.GMMVI.build_from_config>`,
which will automatically construct most of the required objects. However, you still need to pass

1. the dictionary containing the hyperparameters,

2. the :py:class:`target distribution<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`,

3. and the :py:class:`initial model<gmmvi.models.gmm_wrapper.GmmWrapper>`.

The following example script directly uses :py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>` using the hyperparameters
from the following YAML file: :gitlink:`examples/example_config.yml`.

.. literalinclude:: /../../examples/1_directly_using_gmmvi.py

The script can be found under :gitlink:`examples/1_directly_using_gmmvi.py`.


Using the GmmviRunner
~~~~~~~~~~~~~~~~~~~~~
The :py:class:`GmmviRunner<gmmvi.gmmvi_runner.GmmviRunner>` wraps around
:py:class:`GMMVI<gmmvi.optimization.gmmvi.GMMVI>` to add logging capabilities.
Furthermore, the :py:class:`GmmviRunner<gmmvi.gmmvi_runner.GmmviRunner>` takes care
of initializing the model and the target distribution (when using one of the provided
target distributions). Hence, we only need to provide the config to create it,
as shown by the following script:

.. literalinclude:: /../../examples/2_using_the_gmmvi_runner.py

The script can be found under :gitlink:`examples/2_using_the_gmmvi_runner.py`.


Using the GmmviRunner with Default Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can also directly create a default config based on the 7-letter Codeword
to specify the design choices, thereby, not requiring an external YAML file:

.. literalinclude:: /../../examples/3_gmmvi_runner_with_default_configs.py

The script can be found under :gitlink:`examples/3_gmmvi_runner_with_default_configs.py`.

Using the GmmviRunner with Custom Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can still use the :py:class:`GmmviRunner<gmmvi.gmmvi_runner.GmmviRunner>`
with custom environments, but we need to store the
:py:class:`target distribution object<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
in the config:

.. literalinclude:: /../../examples/4_gmmvi_runner_with_custom_environments.py

The script can be found under :gitlink:`examples/4_gmmvi_runner_with_custom_environments.py`.


.. toctree::
   :maxdepth: 2
   :glob: