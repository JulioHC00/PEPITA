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
Each of these variables will have a number of columns with extra descriptions ({variable} can be any of the above) in their names corresponding to:

```{style="max_height:100px"}
- {variable}_median: median of the posterior
- {variable}_mean: mean of the posterior
- {variable}_sd: standard deviation of the posterior
- {variable}_16p: 16 percentile of the posterior
- {variable}_84p: 84 percentile of the posterior
- {variable}_skewness: skewness of the posterior
- {variable}_rhat: rhat value of the posterior
```

### information_analysis_results.csv

Contains the same information as mcmc_fit_results.csv with the addition to the following columns for all variables representing predictions of the information analysis made using our numerical method

```{style="max_height:100px"}
-{variable}_fisher_sd: standard deviation of the posterior as predicted using the information analysis
- {variable1}_{variable2}_cov: covariance of two variables (variable1 and variable2 being any of the variables defined above)
- transit_dur: duration of the transit [days]
```

### information_analysis_results_no_limb_darkening.csv

Same as information_analysis_results.csv but with predictions made using the analytical implementation of the information analysis presented by M. Price (2014).

### multisector_results.csv

Same as mcmc_fit_results but for multisector fits. Includes predictions of the precisions calculated using our information analysis method.

### formatted_toi_predictions.csv

Formatted table with all TESS objects of interest predictions.

```
- TOI: TESS object of interest
- 20s_improv: improvement in radius ratio precision with reobservations with 20s cadence
- 120s_improv: improvement in radius ratio precision with reobservations with 120s cadence
- 600s_improv: improvement in radius ratio precision with reobservations with 600s cadence
```
