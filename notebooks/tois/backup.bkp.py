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

## Generate predictions for the radius ratio precision of TESS Objects of Interest (TOIs)

# %% [markdown]

### Load required packages

# %%

import pandas as pd
import lightkurve as lk
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.constants import G
from IPython.display import display
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
from exoInfoMatrixTOI import exoInfoMatrix
import ldtk.filters as filters
import exoplanet as xo
import torch.multiprocessing as mp

# %% [markdown]

### Select only planet candidates

# %%

# Read TOIs table from the Nasa Exoplanet Archive (NEA)
nea_tois = pd.read_csv("nea_tois.csv", header=90)

nea_tois = nea_tois[nea_tois["tfopwg_disp"] == "PC"] # Only want planet candidates
nea_tois = nea_tois[nea_tois["pl_pnum"] == 1] # And only with a single planet

print(f"Initial number of planet candidates is {len(nea_tois)}\n")

# Only keep if there are values for stellar logg and stellar radius which we will need later on
nea_tois.dropna(axis=0, subset=["st_logg", "st_rad", "st_teff"], inplace=True)

# We also need errors, wherever two values (lower and upper boundaries) for the error are reported or only one is given, we will keep the largest
nea_tois["st_rad_err"] = np.nanmax(nea_tois[["st_raderr1", "st_raderr2"]], axis=-1) 
nea_tois["st_logg_err"] = np.nanmax(nea_tois[["st_loggerr1", "st_loggerr2"]], axis=-1) 
nea_tois["st_teff_err"] = np.nanmax(nea_tois[["st_tefferr1", "st_tefferr2"]], axis=-1) 

# And since we also need the errors later on, only keep columns with errors included
nea_tois.dropna(axis=0, subset=["st_rad_err", "st_logg_err", "st_teff_err"], inplace=True)

print(f"{len(nea_tois)} planet candidates with values for logg, R_star and T_eff along with errors\n")

# Reset indices
nea_tois.reset_index(inplace=True, drop=True)

# %% [markdown]

##### Out of these, we want only those which were observed __only__ with 1800s cadence.

# %%

# We will need to search for the available lightcurves for each of the candidates. Then, if they are only observed with 1800s we add them to a new dataframe

# This can take long
for i, row in nea_tois.iterrows():
    print(f"\n{i} out of {len(nea_tois) - 1}")

    TID = f"TIC {row['tid']}"

    # Results of the lightcurve search
    search = lk.search_lightcurve(TID, mission="TESS")

    # If there were no matches, a KeyError will be raised
    try:
        exptimes = set(search.exptime.value)
    except KeyError:
        # We add a note letting us know this PC wasn't found
        nea_tois.at[i, "notes"] = "NOT FOUND"
        print("NOT FOUND")
        continue

    # We check if the only cadence is 1800s. If it's not we do not flag this candidate as accepted. Otherwise we do
    if not exptimes.issubset(set([1800])):
        nea_tois.at[i, "notes"] = "NOT ONLY 1800s"
        print("NOT ONLY 1800s")
    else:
        nea_tois.at[i, "accepted"] = True
        print("ACCEPTED")

# %%

# Now we save the PCs with observations only in 1800s
nea_tois[nea_tois["accepted"] == True].to_csv("tois_with_only_1800s.csv", index=False)

# %%

# And read these results into a new dataframe
only1800 = pd.read_csv("tois_with_only_1800s.csv")

# %% [markdown]

#### Add columns with some values needed

# %%
# We use astropy units to not have to deal with conversion between units.


logg = only1800["st_logg"] # Log(g)
g = 10**logg.to_numpy() * u.cm * u.s ** (-2) # g in cm/s^2
R = only1800["st_rad"].to_numpy() * c.R_sun # Stellar radius in solar radii
P = only1800["pl_orbper"].to_numpy() * u.day # Period in days
T = only1800["pl_trandurh"].to_numpy() * u.hour # Transit duration in hours

# We store the values only, no units
only1800["st_rho"] = (3/(4 * np.pi * c.G) * g / R).to(u.g * (u.cm)**(-3)).value # Stellar density

only1800["b"] = np.sqrt(1 - ((np.pi * g) / (4 * P * R)) ** (2/3) * T ** 2).value # Impact parameter

# %% [markdown]

#### To make stimates, we need fiducial values for the limb-darkening parameters. We obtain approximate values using 'PyLDTk'.

### THIS CAN TAKE LONG AND WILL DOWNLOAD FILES

# %%

filt = filters.create_tess() # Create TESS filters profiles

copy = only1800.copy() # Copy df to iterate through rows safely

# Iterate through all rows
for i, row in copy.iterrows():
    print(f"Row {i} out of {len(copy) - 1}")

    # Read effective temperature and logg values
    teff = row["st_teff"]
    teff_err = row["st_logg_err"]

    logg = row["st_logg"]
    logg_err = row["st_logg_err"]

    # Just to be sure, we check there are no NaN values
    names = np.array(["teff", "teff_err", "logg", "logg_err"])
    anynan = np.isnan(np.array([teff, teff_err, logg, logg_err]))

    if anynan.any():
        print(f"{row['tid']} has NaN value in {names[anynan]}")

    # Create profiles. Because we have no z value from the table we use 0.25 with error 0.125
    sc = LDPSetCreator(teff=(teff, teff_err), logg=(logg, logg_err), z=(0.25, 0.125), filters=[filt])

    ps = sc.create_profiles(nsamples=1000)

    # Do a mcmc to get the values, if it can't converge print message
    try:
        qc, qe = ps.coeffs_qd(do_mc=True)
    except np.linalg.LinAlgError:
        print(f"Row {i} ({row['tid']}) did not converge")
        only1800.at[i, "u_star1"] = None
        only1800.at[i, "u_star2"] = None
        only1800.at[i, "u_star1_sd"] = None
        only1800.at[i, "u_star2_sd"] = None
        continue

    # Check no NaN values in results
    if np.isnan([qc,qe]).any():
        print(f"Row {i} ({row['tid']}) calculated values are nan somewhere")

    only1800.at[i, "u_star1"] = qc[0][0]
    only1800.at[i, "u_star2"] = qc[0][1]
    only1800.at[i, "u_star1_sd"] = qe[0][0]
    only1800.at[i, "u_star2_sd"] = qe[0][1]

# %%

# And we save only those that did converge

only1800[np.invert(np.isnan(only1800["u_star1"].to_numpy()))].to_csv("tois_with_only_1800s_limbdark.csv", index=False)

limbdarkened = pd.read_csv("tois_with_only_1800s_limbdark.csv")

# %% [markdown]

#### Now that we have limb-darkening values we can get an approximate value for the radius ratio

# %%

# We calculate for each row
for i, row in limbdarkened.copy().iterrows():
    # We create a limb-darkened star from the exoplanet package
    star = xo.LimbDarkLightCurve(row["u_star1"], row["u_star2"])

    # And use the 'get_ror_from_approx_transit_depth' utility to obtain an approximate value for the radius ratio
    ror = star.get_ror_from_approx_transit_depth(row["pl_trandep"]*1e-6, row["b"]).eval()

    limbdarkened.at[i, "ror"] = ror

# %%

# We can now save this as our final dataframe

limbdarkened = limbdarkened[np.invert(np.isnan(limbdarkened["ror"]))]

print(f"{len(limbdarkened)} final planet candidates to be passed onto prediction calculation")

limbdarkened.to_csv("final_dataframe.csv", index=False)

# %% [markdown]

### Now we make the actual predictions

# %%

# To make it faster, we will parallelize the calculations

# CHANGE THIS TO THE NUMBER OF CORES YOU WISH TO USE
NCORES = 12

# Read the final input table
table = pd.read_csv("final_dataframe.csv")

# Now we split them into NCORES tables
tables = np.array_split(table, NCORES)

# We will calculate predicted radius ratio for the following exposure times
calc_expt = {20, 120, 600, 1800}

indices = np.arange(0, NCORES, 1) # Just to keep track of how each thread is doing

# %% [markdown]

#### This function will calculate predictions for each of the tables. We need to include it in a function so as to be able to do multiprocessing

# %%

def calculate_prediction(args):
    df, index = args

    copy = df.copy()
    copy.reset_index(inplace=True, drop=True)

    # Loop through all rows
    for idx, row in copy.iterrows():
        print(f"THREAD {index}: {idx+1} out of {len(copy)}\n")

        # Read the hostname
        host = f"TIC {row['tid']}"

        ref_exptime = 1800 # Our reference exposure time is 1800s, to download a reference lightcurve later one

        # Search the lightcurve
        search = lk.search_lightcurve(host, mission="TESS", exptime=ref_exptime)

        # We give priority to SPOC lightcurves, then QLP and then CDIPS. No reason beyond keeping lightcurves as homogeneous as possible.
        if len(search[["SPOC" in author for author in search.author]]) != 0:
            search = search[["SPOC" in author for author in search.author]]
        elif len(search[["QLP" in author for author in search.author]]) != 0:
            search = search[["QLP" in author for author in search.author]]
        elif len(search[["CDIPS" in author for author in search.author]]):
            search = search[["CDIPS" in author for author in search.author]]

        # Download the lightcurve
        try:
            lc = search[-1].download_all().stitch().remove_nans().remove_outliers(sigma_lower=float('inf'))
        except lk.LightkurveError:
            print(f"{host} lightcurve can't be downloaded ({search.author})")


        # Set the reference mean error of measurements as the mean error for the measurements in the 1800s lightcurve
        ref_sigma = np.mean(np.array(lc.flux_err.value))

        # And the reference timestamps array is also obtained from the lightcurve
        ref_t = np.array(lc.time.value)

        # We also keep track of these values
        copy.at[idx, "ref_exptime"] = ref_exptime
        copy.at[idx, "ref_sigma"] = ref_sigma

        # Now we make the actual predictions for each exposure time
        for exptime in calc_expt:
            # New array of timestamps with points spaced by one exposure time and with a total observation time equal to one sector
            t = np.arange(min(ref_t), max(ref_t), exptime / (3600 * 24))

            # Calculate the new mean error for this exposure time
            sigma = ref_sigma * np.sqrt(ref_exptime)/np.sqrt(exptime)

            # Initialize the information matrix. Oversample of ~100 should be fine but can also do 1000, it will just take longer
            infomatrix = exoInfoMatrix(exptime, oversample=100)

            # This is just to make sure there are no nan values
            anynan = np.isnan(np.array([
                row["pl_orbper"],
                row["pl_tranmid"],
                row["ror"],
                row["b"],
                row["u_star1"],
                row["u_star2"],
                row["st_rho"],
                row["st_rad"]]))

            names = np.array(["pl_orbper", "pl_tranmid", "ror", "b", "u_star1", "u_star2", "st_rho", "st_rad"])

            if np.isnan(t).any():
                print(f"{host} has NaN values for t")
                continue
            if anynan.any():
                print(f"{host} has NaN values for {names[anynan]}")
                continue

            # If there are no NaNs then we set the data
            infomatrix.set_data(
                time_array = t,
                period_val = row["pl_orbper"],
                t0_val     = row["pl_tranmid"],
                ror_val    = row["ror"],
                b_val      = row["b"],
                u1_val     = row["u_star1"],
                u2_val     = row["u_star2"],
                rho_star_val = row["st_rho"],
                r_star_val = row["st_rad"],
            )

            # Then we set the priors. We do not use a prior on stellar density
            infomatrix.set_priors(
                period_prior = np.nanmax(np.abs(row[["pl_orbpererr1", "pl_orbpererr2"]])),
                t0_prior = np.nanmax(np.abs(row[["pl_tranmiderr1", "pl_tranmiderr2"]])),
                r_star_prior = np.nanmax(np.abs(row[["st_raderr1", "st_raderr2"]])),
                b_prior = 1/np.sqrt(12),
                u1_prior = 0.4713,
                u2_prior = 0.4084,
            )

            # And we calculate the information matrix
            try:
                matrix = infomatrix.eval_cov(sigma = np.mean(sigma))
            except ValueError:
                print(f"{host} inversion of matrix failed")
                continue


            # Now we loop through the rows and columns of the matrix to extract the values
            for i, value1 in enumerate(["period", "t0", "ror", "b", "u_star1", "u_star2", "rho_star", "r_star"]):
                for j, value2 in enumerate(["period", "t0", "ror", "b", "u_star1", "u_star2", "rho_star", "r_star"]):

                    # Diagonal gives the standard deviation or predicted precision
                    if value1 == value2:
                        std = np.sqrt(np.abs(matrix[i,j]))
                        col = f"{value1}_{exptime}_sd"
                        copy.at[idx, col] = std

    return copy

# %%

# Now we parallelize the calculation and execute it
# May have problems downloading lightcurves authored by DIAMANTE

arguments = [(df, index) for df, index in zip(tables, indices)]

p = mp.Pool(NCORES)

result = list(p.imap(calculate_prediction, arguments))

p.close()
p.join()

final_df = pd.DataFrame()


for df in result:
    final_df = pd.concat([final_df, df])

final_df.to_csv("tois_with_predictions.csv", index=False)

# %% [markdown]

## Now we calculate the actual improvements in precision by using the predicted precisions and make it into a nice table

# %%

final_df = pd.read_csv("tois_with_predictions.csv")

improvements = pd.DataFrame(columns=['toi', 'tid', '20_improv', '120_improv', 'ror_sd_20', 'ror_sd_120', 'ror_sd_1800'])

improvements['toi'] = final_df['toi']
improvements['tid'] = final_df['tid']
improvements['ror_sd_20'] = final_df['ror_20_sd']
improvements['ror_sd_120'] = final_df['ror_120_sd']
improvements['ror_sd_600'] = final_df['ror_600_sd']
improvements['ror_sd_1800'] = final_df['ror_1800_sd']

improvements['20_improv'] = (1 - improvements['ror_sd_20'] / improvements['ror_sd_1800']) * 100
improvements['120_improv'] = (1 - improvements['ror_sd_120'] / improvements['ror_sd_1800']) * 100
improvements['600_improv'] = (1 - improvements['ror_sd_600'] / improvements['ror_sd_1800']) * 100

improvements.sort_values(by=['20_improv'], ascending=False, inplace=True)

# And this is our nice table with all predictions
improvements

# %% [markdown]

#### Can also convert the table to a latex table

# %%

# We select the top 10

table = improvements.head(10)

table.drop(labels=['tid', 'ror_sd_20', 'ror_sd_120', 'ror_sd_600', 'ror_sd_1800'], inplace=True, axis=1)

table.rename(columns={
    'toi': 'TOI',
    '20_improv': '20s Improv. [%]',
    '120_improv': '120s Improv [%]',
    '600_improv': '600s Improv [%]'
}, inplace=True)

table.to_latex('improvements_table.tex', index=False, float_format="%.2f")

# Which is the table used in the paper
