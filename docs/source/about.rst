About
=====

GMMVI (Gaussian Mixture Model Variational Inference) is a framework for learning GMMS for variational inference,
that was released along with the article :cite:p:`Arenz2023`.

Formally, we aim to optimize a GMM
:math:`q(\mathbf{x})` with Gaussian components :math:`q(\mathbf{x}|o)` and weights :math:`q(o)`,

.. math::
    q(\mathbf{x}) = \sum_o q(o) q(\mathbf{x}|o),

with respect to the optimization problem,

.. math::
    \max_{q(\mathbf{x})} \mathbb{E}_{q(\mathbf{x})} \left[ r(\mathbf{x}) \right] + \text{H}(q(\mathbf{x})),

where :math:`\text{H}(q(\mathbf{x}))` denotes the mixture's entropy and :math:`r(\mathbf{x})` assigns a reward to the
sample :math:`\mathbf{x}`. If :math:`r(\mathbf{x})` corresponds to the energy of a Gibbs-Boltzmann distribution
:math:`p(\mathbf{x}) \propto \exp(r(\mathbf{x}))`, the learned GMM will approximate the target distribution
:math:`p(\mathbf{x})` by minimizing the reverse Kullback-Leibler divergence
:math:`\text{KL}\left(q(\mathbf{x})||p(\mathbf{x})\right)`.

The optimization is performed with respect to the weights, means and covariance matrices, and if desired the number of
components. The framework is build on Tensorflow 2, however, the reward function can also be implemented using different
libraries, such as PyTorch.

Method
------
The optimization is performed iteratively, where at every iteration an independent natural gradient descent step is
performed to the categorical distribution over weights, :math:`q(o)`, and to each individual component
:math:`q(\mathbf{x}|o)`. This procedure was concurrently proposed by :cite:t:`Arenz2018` and :cite:t:`Lin2019`.
However, both approaches differ quite significantly in several design choices (e.g. how the natural gradients are
estimated) and derived the procedure from different perspectives with different theoretical guarantees, and therefore
the equivalence of both approaches was initially not understand. This framework is published along with the article
:cite:p:`Arenz2023` that first established the close connection between both approaches, and was used to systematically evaluate the
effects of the different design choices. For reproducing these experiments, please refer to our
`reproducibility package <https://github.com/OlegArenz/gmmvi-reproducibility>`_.

.. _design choices:

Design Choices
--------------
We distinguish design choices for seven different :ref:`modules <gmmvi-modules>` corresponding to the abstract classes,
where for each design choice, there are two to three option implemented as concrete classes.

1. :ref:`NgEstimator<ng-estimator>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.ng_estimator.NgEstimator
        gmmvi.optimization.gmmvi_modules.ng_estimator.SteinNgEstimator
        gmmvi.optimization.gmmvi_modules.ng_estimator.MoreNgEstimator
        :parts: -4
2. :ref:`ComponentAdaptation<component-adaptation>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.component_adaptation.ComponentAdaptation
        gmmvi.optimization.gmmvi_modules.component_adaptation.FixedComponentAdaptation
        gmmvi.optimization.gmmvi_modules.component_adaptation.VipsComponentAdaptation
        :parts: -4
3. :ref:`SampleSelector<sample-selector>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.sample_selector.SampleSelector
        gmmvi.optimization.gmmvi_modules.sample_selector.VipsSampleSelector
        gmmvi.optimization.gmmvi_modules.sample_selector.LinSampleSelector
        :parts: -4
4. :ref:`NgBasedComponentUpdater<ng-based-component-updater>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.ng_based_component_updater.NgBasedComponentUpdater
        gmmvi.optimization.gmmvi_modules.ng_based_component_updater.DirectNgBasedComponentUpdater
        gmmvi.optimization.gmmvi_modules.ng_based_component_updater.NgBasedComponentUpdaterIblr
        gmmvi.optimization.gmmvi_modules.ng_based_component_updater.KLConstrainedNgBasedComponentUpdater
        :parts: -4
5. :ref:`ComponentStepsizeAdaptation<component-stepsize-adaptation>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation.ComponentStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation.FixedComponentStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation.DecayingComponentStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation.ImprovementBasedComponentStepsizeAdaptation
        :parts: -4
6. :ref:`WeightUpdater<weight-updater>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.weight_updater.WeightUpdater
        gmmvi.optimization.gmmvi_modules.weight_updater.TrustRegionBasedWeightUpdater
        gmmvi.optimization.gmmvi_modules.weight_updater.DirectWeightUpdater
        :parts: -4
7. :ref:`WeightStepsizeAdaptation<weight-stepsize-adaptation>`
    .. inheritance-diagram::
        gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation.WeightStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation.FixedWeightStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation.DecayingWeightStepsizeAdaptation
        gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation.ImprovementBasedWeightStepsizeAdaptation
        :parts: -4

Naming Convention
-----------------
Depending on which option is used for each design choice, there are currently 432 different instantiation
supported by GMMVI. When referring to a specific instantiation, we use 7-letter codewords, where the
presence of a letter implies, that the corresponding option was chosen.
The mapping from letter to option is given in the following table:


====================================  ==================================  ====================================  =====================================
          Module                                                                   Options
------------------------------------  ---------------------------------------------------------------------------------------------------------------
:ref:`ng-estimator`                   :ref:`MORE (Z)<more>`               :ref:`Stein (S)<stein>`
:ref:`component-adaptation`           :ref:`Fixed (E)<fixed-adaptation>`  :ref:`VIPS (A)<vips-adaptation>`
:ref:`sample-selector`                :ref:`Lin (P)<lin-selection>`       :ref:`VIPS (M)<vips-selection>`
:ref:`ng-based-component-updater`     :ref:`Direct (I)<direct-comp>`      :ref:`iBLR (Y)<iblr>`                 :ref:`Trust-Region (T)<kl-comp>`
:ref:`component-stepsize-adaptation`  :ref:`Fixed (F)<fixed-comp>`        :ref:`Decaying (D)<decaying-comp>`    :ref:`Adaptive (R)<adaptive-comp>`
:ref:`weight-updater`                 :ref:`Direct (U)<direct-weight>`    :ref:`Trust-Region (O)<kl-weight>`
:ref:`weight-stepsize-adaptation`     :ref:`Fixed (X)<fixed-weight>`      :ref:`Decaying (G)<decaying-weight>`  :ref:`Adaptive (N)<adaptive-weight>`
====================================  ==================================  ====================================  =====================================

Using this naming convention, ZAMTRUX refers to VIPS :cite:p:`Arenz2020`, and SEPIFUX refers to the method by
:cite:t:`Lin2019`. The recommended setting, however, is SAMTRON.

.. toctree::
   :maxdepth: 2
   :glob:

