
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import dynesty
from dynesty.utils import resample_equal
import dynesty.plotting as dyplot
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy import constants as const
from astropy import units as u

np.random.seed(1)

liv_type = 2
nth = liv_type
sn = -1
ndim = 6

H0 = 67.36
OmegaM = 0.315
OmegaLambda = 1 - OmegaM

Epl = ((const.hbar * const.c**5 / const.G)**0.5).to(u.GeV).value

data = {}


def flatuniverse(z):
    return (1 + z)**nth / np.sqrt((1 + z)**3 * OmegaM + OmegaLambda)

def k_factor(z):
    # SAME as authors (DO NOT change normalization)
    integral = quad(flatuniverse, 0, z)[0]
    return 3.086e19 * integral / (2.0 * H0)

def LIV_lag(logEQG, E, E1, z):
    EQG_keV = (10**logEQG) * 1e6  # GeV ? keV

    return sn * (1 + nth) * (E**nth - E1**nth) * k_factor(z) / (EQG_keV**nth)

# -------- SBPL intrinsic lag --------
def tau_int_sbpl(E, zeta, Eb, alpha1, mu, alpha2, z, E0):

    Eb = np.maximum(Eb, 1e-6)
    mu = np.clip(mu, 1e-3, 3.0)

    x = (E - E0) / Eb

    sign = np.sign(x)
    abs_x = np.abs(x) + 1e-12

    log_x = np.log(abs_x)
    y = np.clip(log_x / mu, -50, 50)
    term = 0.5 * (1.0 + np.exp(y))
    tau = zeta * (abs_x**alpha1) * (term ** ((alpha2 - alpha1)*mu))

    # restore sign for power-law part
    tau *= sign**alpha1

    # In paper, they multiply by (1+z) to convert to observer frame.
    tau *= (1 + z)

    if not np.all(np.isfinite(tau)):
        return np.full_like(E, np.nan)

    return tau

def prior_transform(u):

    if nth == 1:
        return np.array([
            20*u[0],                          # logEQG
            4*u[1],                               # zeta
            0.05 + (5000 - 0.05)*u[2],            # Eb (avoid 0)
            -3 + 13*u[3],                         # alpha1
            1e-3 + (3 - 1e-3)*u[4],               # mu (avoid 0)
            -10 + 13*u[5]                         # alpha2
        ])

    if nth == 2:
        return np.array([
            15*u[0],
            4*u[1],
            0.05 + (5000 - 0.05)*u[2],
            -3 + 13*u[3],
            1e-3 + (3 - 1e-3)*u[4],
            -10 + 13*u[5]
        ])

    raise ValueError("Invalid nth value")

def d_tau_liv_dE(logEQG, E, E1, z):
    EQG_keV = (10**logEQG) * 1e6
    K = k_factor(z)

    return sn * (1 + nth) * nth * (E**(nth - 1)) * K / (EQG_keV**nth)

def d_tau_int_dE(E, zeta, Eb, a1, mu, a2, z, E0, E_err):

    # use data-driven step (VERY important)
    dE = np.maximum(E_err, 1e-6)

    tau_plus = tau_int_sbpl(E + dE, zeta, Eb, a1, mu, a2, z, E0)
    tau_minus = tau_int_sbpl(E - dE, zeta, Eb, a1, mu, a2, z, E0)

    if np.any(~np.isfinite(tau_plus)) or np.any(~np.isfinite(tau_minus)):
        return None

    return (tau_plus - tau_minus) / (2 * dE)

def sigma_model(logEQG, zeta, Eb, a1, mu, a2, E, E1, z, E_err):

    d_liv = d_tau_liv_dE(logEQG, E, E1, z)
    d_int = d_tau_int_dE(E, zeta, Eb, a1, mu, a2, z, E1, E_err)

    if d_int is None:
        return None

    deriv_total = d_liv + d_int

    sigma_mod = np.abs(deriv_total) * E_err
    sigma_mod = np.maximum(sigma_mod, 1e-6)

    return sigma_mod

def loglike(theta):
    logEQG, zeta, Eb, a1, mu, a2 = theta

    if a1 <= a2:
        return -np.inf


    try:
        E = data['E']
        t_obs = data['t_obs']
        t_err = data['t_err']
        E1 = data['E_1']
        z = data['redshift']
        E_err = data['E_err']

        # --- model ---
        tau_liv = LIV_lag(logEQG, E, E1, z)
        tau_int = tau_int_sbpl(E, zeta, Eb, a1, mu, a2, z, E1)

        if tau_int is None:
            return -np.inf

        if np.any(~np.isfinite(tau_liv)) or np.any(~np.isfinite(tau_int)):
            return -np.inf

        tau_model = tau_liv + tau_int
        residuals = t_obs - tau_model

        # sigma_model = derivative-based error propagation
        sigma_liv = sigma_model(logEQG, zeta, Eb, a1, mu, a2, E, E1, z, E_err)
        # sigma_liv = sn * (1.0 + nth) * nth * (E**(nth - 1.0)) * k_factor(z) / (10**logEQG * 1.0E6)**nth * E_err

        if sigma_liv is None:
            return -np.inf

        # sigma2 = t_err**2 + sigma_liv**2
        sigma = np.sqrt(t_err**2 + sigma_liv**2)

        ll = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))

        return ll if np.isfinite(ll) else -np.inf

    except Exception as e:
        print("Error in loglike:", e)
        return -np.inf

def run_event(nthreads, event, event_params, data_dir, out_root):

    global data

    name = event.replace(".txt", "")
    print(f"\n?? Running {name}")

    if name not in event_params:
        print(f"?? Skipping {name}")
        return

    df = pd.read_csv(
        os.path.join(data_dir, event),
        names=['E','E_err','t_obs','t_err'],
        sep=r"\s+"
    )

    data = {
        'E': df['E'].values,
        'E_err': df['E_err'].values,
        't_obs': df['t_obs'].values,
        't_err': df['t_err'].values,
        'E_1': event_params[name][0],
        'redshift': event_params[name][2]
    }

    outdir = os.path.join(out_root, name)
    os.makedirs(outdir, exist_ok=True)

    # Run the dynamic nested sampler with multiprocessing
    with dynesty.pool.Pool(nthreads, loglike, prior_transform) as p:
        
        sampler = dynesty.DynamicNestedSampler(
            p.loglike,
            p.prior_transform,
            ndim=ndim,
            nlive=1500,          
            bound='multi',
            sample='rwalk'
        )

        sampler.run_nested(n_effective=25000)
        res = sampler.results

    # -------- Evidence --------
    logz = res['logz'][-1]
    logzerr = res['logzerr'][-1]

    with open(os.path.join(out_root, f"ln_evidence_subL_n{nth}.txt"), "a") as f:
        f.write(f"{name}\t{logz:.6f}\t{logzerr:.6f}\n")

    # -------- Posterior --------
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples = resample_equal(res.samples, weights)

    cols = ['logEQG','zeta','E_b','alpha_1','mu','alpha_2']
    df_samples = pd.DataFrame(samples, columns=cols)
    df_samples['weights'] = 1.0/len(df_samples)

    df_samples.to_csv(os.path.join(outdir, f"{nth}th-post_equal_weights.dat"), sep="\t", index=False, header=False)

    # -------- Corner --------
    fig, _ = dyplot.cornerplot(res, labels=cols)
    if nth == 1:
        fig.savefig(os.path.join(outdir, "1st-marg.jpg"))
    if nth == 2:
        fig.savefig(os.path.join(outdir, "2nd-marg.jpg"))

    plt.close(fig)
    print(f"? Done {name}")

# MAIN

def main():
    # nthreads = 30
    nthreads = 1
    data_dir = "../Data/lag_data/lag_err_fermi_32grbs/"
    param_file = "../Data/lag_data/GRBPARAM.csv"
    out_root = "../Data/posteriors_SBPL_L22"

    os.makedirs(out_root, exist_ok=True)

    # -------- Load CSV (TRANSPOSE) --------
    params_df = pd.read_csv(param_file)
    params_df.set_index(params_df.columns[0], inplace=True)

    event_params = {}

    for grb in params_df.columns:
        try:
            E1 = float(params_df.loc['E0', grb])
            z  = float(params_df.loc['redshift', grb])
            event_params[grb] = [E1, None, z]
        except:
            print(f"?? Skipping {grb}")

    print("\n?? Sample:")
    for k in list(event_params.keys())[:3]:
        print(k, event_params[k])

    # -------- Loop --------
    for event in sorted(os.listdir(data_dir)):
        if event.endswith(".txt"):
            try:
                run_event(nthreads, event, event_params, data_dir, out_root)
            except Exception as e:
                print(f"? Failed {event}: {e}")

    print("\n?? ALL DONE")


if __name__ == "__main__":
    main()