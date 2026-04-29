
# The following code is used to infer the LIV parameters for individual GRBs. 
# The intrinsic time lags are fitted with Cubic Spline defined in Function "fit_int_spline".

import os
import pandas as pd
import dynesty
from dynesty.utils import resample_equal
import dynesty.plotting as dyplot
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1)

# LIV scenario selection
liv_type = 1  # 1 for linear LIV, 2 for quadratic LIV

# Constants
hbar = const.hbar
c = const.c
G = const.G
E_P = ((hbar * c**5 / G)**0.5).to(u.joule)
E_P_GeV = E_P.to(u.gigaelectronvolt)
Epl = E_P_GeV.value  # GeV

# Cosmology parameters
H0 = 67.36
OmegaM = 0.315
OmegaLambda = 1.0 - OmegaM

# LIV parameters
sn = -1  # -1 for sub-luminous, +1 for super-luminous

# LIV type and label setup
nth = liv_type
label = [r"$log_{10}\ (E_{\rm QG," + str(nth) + "})$"]
ndim = 1  # Number of sampling parameters

# Model functions
def flatuniverse(z):
    return (1.0 + z)**nth / ((1.0 + z)**3.0 * OmegaM + OmegaLambda)**0.5 

def k_factor_function(z, H0):
    cosmos_ = quad(flatuniverse, 0.0, z)[0]
    return 3.086E19 * cosmos_ / (2.0 * H0)

def fit_int_spline(x, t_int, s=None):
    t_err = data['t_err']
    spline_model = UnivariateSpline(x, t_int, w=1/t_err, s=s)
    tint_pred = spline_model(x)
    residuals = t_int - tint_pred
    return tint_pred, np.std(residuals), spline_model

def LIV_lag(logE_QG, E, E_1, z):
    k_factor = k_factor_function(z, H0)
    return sn * (1.0 + nth) * (E**nth - E_1**nth) * k_factor / (10**logE_QG * 1.0E6)**nth

def compute_chisq(logE_QG):
    E, t_obs, E_err, t_err, E_1, z = data['E'], data['t_obs'], data['E_err'], data['t_err'], data['E_1'], data['redshift']
    lag_liv = LIV_lag(logE_QG, E, E_1, z)
    lag_int = t_obs - lag_liv
    int_pred, sigma_spline, _ = fit_int_spline(E, lag_int)
    lag_fit = int_pred + lag_liv
    residuals = t_obs - lag_fit
    sigma_liv = sn * (1.0 + nth) * nth * (E**(nth - 1.0)) * k_factor_function(z, H0) / (10**logE_QG * 1.0E6)**nth * E_err
    sigma = np.sqrt(t_err**2 + sigma_liv**2)
    return np.sum((residuals / sigma)**2)

def prior_transform(utheta):
    return 20 * utheta if liv_type == 1 else 15 * utheta

def loglike(theta):
    logE_QG = theta[0]
    E, t_obs, E_err, t_err, E_1, z = data['E'], data['t_obs'], data['E_err'], data['t_err'], data['E_1'], data['redshift']
    try:
        lag_liv = LIV_lag(logE_QG, E, E_1, z)
        int_pred, sigma_int, _ = fit_int_spline(E, t_obs - lag_liv)
        t_fit = int_pred + lag_liv
        residuals = t_obs - t_fit
        sigma_liv = sn * (1.0 + nth) * nth * (E**(nth - 1.0)) * k_factor_function(z, H0) / (10**logE_QG * 1.0E6)**nth * E_err
        sigma = np.sqrt(t_err**2 + sigma_liv**2)
        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf

# Run for one event
def run_single_event(nthreads, event, event_parameters, directory, outdir, fn_lnevidence, labels):
    global data
    print(f"Processing {event[:-4]} ...")

    df = pd.read_csv(os.path.join(directory, event), names=['E', 'E_err', 't_obs', 't_err'], delim_whitespace=True)
    data = {
        'redshift': event_parameters[event[:-4]][2],
        'E_1': event_parameters[event[:-4]][0],
        'E': df['E'].values,
        'E_err': df['E_err'].values,
        't_obs': df['t_obs'].values,
        't_err': df['t_err'].values
    }

    # Save event info 
    # ---------------NOT THERE----------------------
    with open(os.path.join(outdir, "run_info_records.txt"), "a") as f:
        f.write(f"{event[:-4]},{data['redshift']:.4f},{data['E_1']:.4f}\n")

    # Run the dynamic nested sampler
    with dynesty.pool.Pool(nthreads, loglike, prior_transform) as pool:
        dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim=ndim, nlive=1500, bound='multi', sample='rwalk')
        dsampler.run_nested(n_effective=25000)
        print(f"{event[:-4]} finished.")
        dres = dsampler.results

    # Save log evidence
    fn_logz = os.path.join(outdir, fn_lnevidence)
    with open(fn_logz, "a") as f:
        f.write(f"{event[:-4]},{dres['logz'][-1]:.4f},{dres['logzerr'][-1]:.4f}\n")

    # Resample and save posterior samples
    weights = np.exp(dres['logwt'] - dres['logz'][-1])
    samples = resample_equal(dres.samples, weights)
    np.savetxt(os.path.join(outdir, f"{event[:-4]}_posterior_samples.txt"), samples[:, 0], fmt="%.4f")

    # Corner plot
    try:
        fig, _ = dyplot.cornerplot(dres, color='slateblue', show_titles=True, smooth=0.03, title_kwargs={'y': 1.04, 'fontsize': 16}, labels=labels)
        fig.savefig(os.path.join(outdir, f"{event[:-4]}_cornerplot.png"))
        plt.close(fig)
    except Exception as e:
        print("cornerplot error:", e)

    # Goodness of fit check
    median_vals = np.percentile(dres['samples'][:, 0], 50)
    logE_QG_best = median_vals
    lag_liv_best = LIV_lag(logE_QG_best, data['E'], data['E_1'], data['redshift'])
    lag_int_best, sigma_int_best, _ = fit_int_spline(data['E'], data['t_obs'] - lag_liv_best)
    lag_fit = lag_int_best + lag_liv_best
    residuals_fit = data['t_obs'] - lag_fit

    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1]})
    axs[0].errorbar(data['E'], data['t_obs'], data['t_err'], data['E_err'], fmt='o', c='k')
    axs[0].plot(data['E'], lag_fit)
    axs[0].plot(data['E'], lag_int_best, '--b', alpha=0.3)
    axs[0].plot(data['E'], lag_liv_best, '--g', alpha=0.3)
    axs[0].fill_between(data['E'], lag_int_best - sigma_int_best, lag_int_best + sigma_int_best, alpha=0.3)
    axs[1].errorbar(data['E'], residuals_fit, data['t_err'], data['E_err'], fmt='o', c='k')
    axs[1].axhline(0, color='r', linestyle='--')
    axs[1].set_xscale("log")
    axs[0].set_xscale("log")
    axs[1].set_xlabel(r"$E_{\rm obs}$(keV)")
    axs[0].set_ylabel("Time lag (s)")
    axs[1].set_ylabel("Residuals")
    fig.savefig(os.path.join(outdir, f"{event[:-4]}_best_fit_results.png"))
    plt.close(fig)

    # Posterior reduced chi-square
    chisqs = np.array([compute_chisq(E) / (len(data['E']) - ndim) for E in samples[:, 0]])
    low, high = np.percentile(chisqs, [2.5, 97.5])
    plt.figure(figsize=(4, 4))
    plt.hist(chisqs, bins=40, range=(low, high))
    plt.xlabel(r"$\chi^2/\mathrm{dof}$")
    plt.ylabel("Counts")
    plt.title("Posterior reduced chi-square")
    plt.savefig(os.path.join(outdir, f"{event[:-4]}_chisq_distribution.png"))
    plt.close()

def main():
    directory = "../Data/lag_data/lag_err_fermi_32grbs/"
    event_parameters = pd.read_csv("../Data/lag_data/GRBPARAM.csv")
    events_name_list = os.listdir(directory)
    nthreads = 1  # Number of threads for multiprocessing processes

    outdir = "../Data/posteriors_cubicspline_DU25/result_subL_n{0}_cubicspline".format(nth)
    fn_lnevidence = f"ln_evidence_subL_n{nth}.txt"
    labels = r"$\log(E_{QG," + str(nth) + "})$"
    
    os.makedirs(outdir, exist_ok=True)
    for event in events_name_list:
        try:
            run_single_event(nthreads, event, event_parameters, directory, outdir, fn_lnevidence, labels)
        except Exception as e:
            print(f"Run failed for {event}: {e}")
    
    print("All events finished.")

if __name__ == "__main__":
    main()
