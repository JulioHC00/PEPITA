.. PEPITA documentation master file, created by
   sphinx-quickstart on Tue Feb 14 15:18:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Prediction of Exoplanet Precisions using Information in Transit Analysis
========================================================================

Introduction
************

PEPITA is a Python package that allows making predictions for the precision of exoplanet parameters using transit light-curves. Behind scenes, it makes use of the Information Analysis techniques to predict the best precision you will get by fitting a light-curve without actually needing to perform the fit.

Motivation
**********

Being able to predict the precision of parameters without needing to perform fits to data allows a more efficient planning of observations or re-observations. For example, if you find that an exoplanet of your interest which has been observed with a cadence of 1800s cadence will get an improved measurement of its radius ratio of 30% if reobserved with 120s cadence, you know that re-observations will be worth it. Or you may find out that the improvement is 1% and re-observing is not worth it.

For more details about the motivation and results using this package see the `associated paper <https://doi.org/10.1093/mnras/stad408>`_

Documentation index
*******************

.. toctree::
   :maxdepth: 2

   pepita
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
