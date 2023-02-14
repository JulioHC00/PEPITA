# Prediction of Exoplanet Precisions using Information in Transit Analysis (PEPITA)


[![Documentation Status](https://readthedocs.org/projects/pepita/badge/?version=latest)](https://pepita.readthedocs.io/en/latest/?badge=latest)
## Introduction

PEPITA is a Python package that allows making predictions for the precision of exoplanet parameters using transit light-curves. Behind scenes, it makes use of the Information Analysis techniques to predict the best precision you will get by fitting a light-curve without actually needing to perform the fit.
### Motivation

Being able to predict the precision of parameters without needing to perform fits to data allows a more efficient planning of observations or re-observations. For example, if you find that an exoplanet of your interest which has been observed with a cadence of 1800s cadence will get an improved measurement of its radius ratio of 30% if reobserved with 120s cadence, you know that re-observations will be worth it. Or you may find out that the improvement is 1% and re-observing is not worth it.

For more details about the motivation and results using this package see the [associated paper](https://doi.org/10.1093/mnras/stad408)

## Get started

1. Install the package using

```bash
pip install pepita
```
2. Read the [docs](https://pepita.readthedocs.io/en/latest/) and follow through the example notebooks.

## Data and others

Some notebooks are provided either as examples of how to use our information analysis class to make predictions of parameter precisions or to showcase how some of the analyses used in the paper were performed. These can be found under the data and notebooks directories
