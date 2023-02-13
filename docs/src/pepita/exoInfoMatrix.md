# Exoinfomatrix

[Pepita Index](../../README.md#pepita-index) /
`src` /
[Pepita](./index.md#pepita) /
Exoinfomatrix

> Auto-generated documentation for [src.pepita.exoInfoMatrix](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py) module.

- [Exoinfomatrix](#exoinfomatrix)
  - [exoInfoMatrix](#exoinfomatrix)
    - [exoInfoMatrix().erase_priors](#exoinfomatrix()erase_priors)
    - [exoInfoMatrix().eval_cov](#exoinfomatrix()eval_cov)
    - [exoInfoMatrix().eval_deriv_matrix](#exoinfomatrix()eval_deriv_matrix)
    - [exoInfoMatrix().eval_fisher](#exoinfomatrix()eval_fisher)
    - [exoInfoMatrix().eval_point](#exoinfomatrix()eval_point)
    - [exoInfoMatrix().get_approx_transit_duration](#exoinfomatrix()get_approx_transit_duration)
    - [exoInfoMatrix().get_in_transit](#exoinfomatrix()get_in_transit)
    - [exoInfoMatrix().plot_derivs](#exoinfomatrix()plot_derivs)
    - [exoInfoMatrix().setExptime](#exoinfomatrix()setexptime)
    - [exoInfoMatrix().set_data](#exoinfomatrix()set_data)
    - [exoInfoMatrix().set_priors](#exoinfomatrix()set_priors)
    - [exoInfoMatrix().set_t](#exoinfomatrix()set_t)

## exoInfoMatrix

[Show source in exoInfoMatrix.py:7](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L7)

Information Analysis Matrix class.

To initalize, use the exposure time (cadence) of the lightcurve in seconds. For example:

info_matrix = exoInfoMatrix(20) # 20 second cadence

Then define the parameters of the fiducial model

info_matrix.set_data(
        time_array,
        period_val,
        t0_val,
        ror_val,
        b_val,
        u1_val,
        u2_val,
        m_star_val,
        r_star_val,
        )

To calculate the covariance matrix, passing the error of the measurements into it

cov_matrix = info_matrix.eval_cov(sigma)

#### Signature

```python
class exoInfoMatrix:
    def __init__(self, exptime, oversample=100):
        ...
```

### exoInfoMatrix().erase_priors

[Show source in exoInfoMatrix.py:364](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L364)

Used to erase all priors

#### Signature

```python
def erase_priors(self):
    ...
```

### exoInfoMatrix().eval_cov

[Show source in exoInfoMatrix.py:416](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L416)

Used to evaluate the covariance matrix

Parameters
----------

sigma: float or array
    Error in the flux measurements. Either a single float value to be used for all points or an array of size len(time_array) with individual errors for each timestamps

Returns
-------
array matrix

#### Signature

```python
def eval_cov(self, sigma=None):
    ...
```

### exoInfoMatrix().eval_deriv_matrix

[Show source in exoInfoMatrix.py:227](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L227)

Evaluates the matrix of derivatives for all timestamps in time_array

Returns
-------
array matrix

#### Signature

```python
def eval_deriv_matrix(self):
    ...
```

### exoInfoMatrix().eval_fisher

[Show source in exoInfoMatrix.py:373](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L373)

Used to evaluate the information matrix

Parameters
----------

sigma: float or array
    Error in the flux measurements. Either a single float value to be used for all points or an array of size len(time_array) with individual errors for each timestamps

Returns
-------
array matrix

#### Signature

```python
def eval_fisher(self, sigma):
    ...
```

### exoInfoMatrix().eval_point

[Show source in exoInfoMatrix.py:190](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L190)

Evaluates the derivatives at a given point

Parameters
----------

tval: float
    Time at which to evaluate the derivatives, in same units as time_array

Raises
------
ValueError
    If values of the fiducial model have not been set.

Returns
-------
array of derivatives

#### Signature

```python
def eval_point(self, tval):
    ...
```

### exoInfoMatrix().get_approx_transit_duration

[Show source in exoInfoMatrix.py:469](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L469)

Approximate the duration of the transit. Not efficient and used only for testing purposes

#### Signature

```python
def get_approx_transit_duration(self, n_points=10000):
    ...
```

### exoInfoMatrix().get_in_transit

[Show source in exoInfoMatrix.py:445](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L445)

Get the number of data points which are in-transit

Returns
-------
int

#### Signature

```python
def get_in_transit(self):
    ...
```

### exoInfoMatrix().plot_derivs

[Show source in exoInfoMatrix.py:249](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L249)

Plots the derivatives of the transit model.

Parameters
----------

fig_ax: (figure, axes)
    If derivatives should be plotted in given figure and axes. Note that this is meant to be used for plotting different model derivatives on top of eachother and so (figure, axis) should be the output of calling plot_derivs() in other model.

Returns
-------
(figure, axes)

#### Signature

```python
def plot_derivs(self, fig_ax=None):
    ...
```

### exoInfoMatrix().setExptime

[Show source in exoInfoMatrix.py:304](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L304)

Changes the exposure time (cadence) of the model. Will force redefinition of parameters

Parameters
----------

exptime: float
    The new cadence in seconds.

#### Signature

```python
def setExptime(self, exptime):
    ...
```

### exoInfoMatrix().set_data

[Show source in exoInfoMatrix.py:74](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L74)

Set the data for the fiducial model. The fiducial model is used to evaluate the derivatives and can be a first approximation of the planet parameters from the maximum likelihood set of parameters or from previous fits.

Parameters
----------

time_array: array[float]
    Timestamps of the lightcurve. In days.
period_val: float
    Period of the planet in days
t0_val: float
    Reference time for the middle of one of the planet transits. In days.
ror_val: float
    Radius ratio of the planet and star
b_val: float
    Impact parameter of the planet
u1_val: float
    First limb-darkening parameter for a quadratic limb-darkening model
u2_val: float
    Second limb-darkening parameter for a quadratic limb-darkening model
m_star_val: float
    Mass of the star in solar masses
r_star_val: float
    Radius of star in solar radii

#### Signature

```python
def set_data(
    self,
    time_array,
    period_val,
    t0_val,
    ror_val,
    b_val,
    u1_val,
    u2_val,
    m_star_val,
    r_star_val,
):
    ...
```

### exoInfoMatrix().set_priors

[Show source in exoInfoMatrix.py:323](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L323)

Used to define priors for the parameters

Parameters
----------

period_prior: float
    Prior for planet period in days
t0_prior: float
    Prior for t0 in days
ror_prior: float
    Prior for planet ratio
b_prior: float
    Prior for impact parameter
u1_prior: float
    Prior for first quadratic limb darkening parameter
u2_prior: float
    Prior for second quadratic limb darkening parameter
m_star_prior: float
    Prior for mass of the star
r_star_prior: float
    Prior for radius of the star

#### Signature

```python
def set_priors(
    self,
    period_prior=np.nan,
    t0_prior=np.nan,
    ror_prior=np.nan,
    b_prior=np.nan,
    u1_prior=np.nan,
    u2_prior=np.nan,
    m_star_prior=np.nan,
    r_star_prior=np.nan,
):
    ...
```

### exoInfoMatrix().set_t

[Show source in exoInfoMatrix.py:518](https://github.com/JulioHC00/PEPITA/blob/main/src/pepita/exoInfoMatrix.py#L518)

Change the time array of the model

Parameters
----------

time_array: array
    New timestamps for the data

#### Signature

```python
def set_t(self, time_array):
    ...
```