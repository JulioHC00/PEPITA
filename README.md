# exoInfoMatrix

Files relating to the paper (MISSING)

## Main files

The main results are contained in the following files

### mcmc_fit_results.csv

Contains the results of the transit fits. The colum meanings are

```{style="max_height:100px"}
- hostname: name of the host star
- host_mass: mass of the host star (in solar masses)
- host_radius: radius of the host star (in solar radii)
- host_radius_sd: standard deviation used in the prior of the host star radius
- exptime: cadence or exposure time of the fit [s]
- sectors: sector of the fitted lightcurve
- model_key: not used
- seeds: seeds used for each of the chains in the MCMC
- n_tune: number of tuning steps in the MCMC
- n_ draw: number of draws in the MCMC
- n_cores: number of cores used
- target_accept: target accept parameter of the MCMC
- n_chains: number of chains in the MCMC
- transit_points: number of points in the lightcurve that are inside a transit
```
There are columns which refer to different characteristics of the posterior distributions. The variables are

```{style="max_height:100px"}
- mean: mean flux of the lightcurve outside the transit
- t0: reference time of a transit [days]
- period: period of the transit [days]
- log_ror: logarithm of the radius ratio
- ror: radius ratio
- log_sigma_lc, log_rho_gp, log_sigma_gp: parameters of the Gaussian process
- u_star1, u_star2: quadratic limb-darkening parameters
- m_star: mass of the host star
- r_star: radius of the host star
- r_pl: radius of the planet (in solar radii)
- b: impact parameter
```
Each of these variables will have a number of columns with extra descriptions in their names corresponding to:

```{style="max_height:100px"}
- _median: median of the posterior
- _mean: mean of the posterior
- _sd: standard deviation of the posterior
- _16p: 16 percentile of the posterior
- _84p: 84 percentile of the posterior
- _skewness: skewness of the posterior
- _rhat: rhat value of the posterior
```
