# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as c
import lightkurve as lk

# %%

nea = pd.read_csv("PS_2022.12.01_03.43.54.csv", header=293)
fits = pd.read_csv("../../mcmc_fit_results.csv")
fisher = pd.read_csv("../../information_analysis_results.csv")

hostnames = set(fits["hostname"])

nea = nea[list(map(lambda x: x in hostnames, nea["hostname"]))]

fits["nea_depth"] = [nea[nea["hostname"] == hostname]["pl_trandep"].to_numpy()[0] / 100 for hostname in fits["hostname"]]

# %%

m_star = fits["m_star_median"].to_numpy() * c.M_sun
r_star = fits["r_star_median"].to_numpy() * c.R_sun
period = fits["period_median"].to_numpy() * u.day
ror = fits["ror_median"].to_numpy() * u.dimensionless_unscaled
cadences = fits["exptime"].to_numpy() * u.s
b = fits["b_median"].to_numpy() * u.dimensionless_unscaled

# Calculate the quantities we need. We use units until we save them into the fits dataframe

a = (((c.G * m_star * (period) ** 2) / (4 * np.pi ** 2)) ** (1 / 3) / r_star).decompose() # Semi-major axis

tau_0 = period / (a * 2 * np.pi) # Time for planet to move 1 stellar radii projected on sky

tau = 2 * tau_0 * ror / np.sqrt(1 - b ** 2)

delta = ror ** 2

Gamma = 1 / cadences # Sampling rate

T = 2 * tau_0 * np.sqrt(1 - b ** 2) # FWHM duration of transit

# %%

# To get the total duration of the observation in one sector and get the mean value of the errors in the measurements we search for the lightcurves

Ttot = []
sigma = []

for idx, row in fits.reset_index().iterrows():
    print(f"{idx} out of {len(fits)}")

    host = row["hostname"]
    exptime = row["exptime"]
    sector = row["sectors"][1:-1]

    search = lk.search_lightcurve(host, sector=sector, exptime=exptime, mission="TESS")
    search = search[["SPOC" in author for author in search.author]]

    lc = search.download_all().stitch().remove_nans().remove_outliers(sigma_lower=float('inf'))

    sig = np.array(lc.flux_err.value)
    t = np.array(lc.time.value)

    Ttot.append((max(t) - min(t)) * u.day)
    sigma.append(np.mean(sig))

# %%

# Convert all quantities to common units and remove units

Ttot = np.array([val.value for val in Ttot]) * u.day
sigma = np.array(sigma)

# %%

# Now we calculate constants defined in M. Price's paper

b1 = (6 * cadences ** 2 - 3 * cadences * Ttot + tau * Ttot) / cadences ** 2
b10 = (-tau ** 4 + 24 * T * cadences ** 2 * (tau - 3 * cadences) + 60 * cadences ** 4 + 52 * tau * cadences ** 3 - 44 * tau ** 2 * cadences ** 2 + 11 * tau ** 3 * cadences) / cadences ** 4
b2 = (tau * T + 3 * cadences * (cadences - T)) / cadences ** 2
b4 = (6 * T **2 - 6 * T * Ttot + cadences * (5 * Ttot - 4 * cadences)) / cadences ** 2
b6 = (12 * b4 * cadences ** 3 + 4 * tau * (-6 * T ** 2 + 6 * T * Ttot + cadences * (13 * Ttot - 30 * cadences))) / cadences ** 3
b7 = (b6 * cadences ** 5 + 4 * tau ** 2 * cadences ** 2 * ( 12 * cadences - 11 * Ttot) + tau ** 3 * cadences * (11 * Ttot - 6 * cadences) - tau ** 4 * Ttot) / cadences ** 5

B9 = (24 * b1 + b10 * delta ** 2 - 48 * b2 * delta)

# And the actual standard deviation
variance = sigma**2 / Gamma * (B9 / (4 * b7 * cadences * ror ** 2))
sd = np.sqrt(variance)

# %%

fits["ror_price_sd"] = sd.decompose().value

fits.to_csv("results_with_price_predictions.csv", index=False)
