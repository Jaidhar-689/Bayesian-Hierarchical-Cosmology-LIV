import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
import traceback
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from bilby.hyper.likelihood import HyperparameterLikelihood

# Hyperparameters flags
hyperLIV_liu_n1 = 0  # Linear LIV model for Sample I
hyperLIV_liu_n2 = 0  # Quadratic LIV model for Sample I
hyperLIV_du_n1 = 0   # Linear LIV model for Sample II
hyperLIV_du_n2 = 1   # Quadratic LIV model for Sample II

# Population-level distribution options
gauss = 0     # Use Gaussian distribution if True
log_normal = 1  # Use log-normal distribution if True

# Load posterior samples from the files
def posterior_du(current_dir, column_names=None):
    """
    Load posterior data from sample files in the specified directory.
    """
    results_du = []
    if column_names is None:
        column_names = ['logEQG']
        
    if not os.path.exists(current_dir):
        print(f"Error: Directory '{current_dir}' does not exist")
        return results_du

    for filename in os.listdir(current_dir):
        if filename.endswith('_samples.txt'):
            file_path = os.path.join(current_dir, filename)
            try:
                df = pd.read_csv(file_path, sep=',', header=None, skiprows=1, names=column_names)
                if not df.empty:
                    results_du.append(df)
                    print(f"Processed file {filename}: read {len(df)} rows")
                else:
                    print(f"Warning: File {filename} is empty")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                traceback.print_exc()
    
    return results_du

def posterior_L22(current_dir, data_for_LIVorder, column_names=None):
    """
    Load posterior data from L22 format files in the specified directory.
    """
    if column_names is None:
        column_names = ['logEQG', 'zeta', 'E_b', 'alpha_1', 'mu', 'alpha_2', 'weights']
    
    results_liu = []
    
    if not os.path.exists(current_dir):
        print(f"Error: Directory '{current_dir}' does not exist")
        return results_liu

    folders = [item for item in os.listdir(current_dir) 
               if os.path.isdir(os.path.join(current_dir, item)) and not item.startswith('.')]
    result_file_list = [os.path.join(current_dir, name, data_for_LIVorder) for name in folders]
    
    sample_dir = [s for s in result_file_list if 'DS_Store' not in s and os.path.exists(s)]
    
    for dat_file in sample_dir:
        try:
            df = pd.read_csv(dat_file, sep='\s+', header=None)
            df.columns = column_names
            results_liu.append(df)
            print(f"Successfully loaded: {dat_file}")
        except Exception as e:
            print(f"Error loading file {dat_file}: {str(e)}")
    
    print(f"Total files loaded: {len(results_liu)}")
    return results_liu

def gaussian_prior(dataset, mu, sigma):
    """Gaussian distribution."""
    return np.exp(-(dataset["logEQG"] - mu) ** 2 / (2 * sigma**2)) / (2 * np.pi * sigma**2) ** 0.5

def log_normal_prior(dataset, mu, sigma):
    """Log-normal distribution."""
    x = dataset["logEQG"]
    prob = np.exp(-(np.log(x / mu))**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
    # return prob
    return np.maximum(prob, 1e-300)  # Avoid underflow by setting a minimum probability

# Load posterior samples based on hyperLIV flags
if hyperLIV_liu_n1:
    current_dir_L22 = '../Data/posteriors_SBPL_L22/'
    # current_dir_L22 = '/home/dss/HierarchicalLIV/Data/posteriors_SBPL_L22/'
    samples = posterior_L22(current_dir_L22, '1th-post_equal_weights.dat').copy()
if hyperLIV_liu_n2:
    current_dir_L22 = '../Data/posteriors_SBPL_L22/'
    # current_dir_L22 = '/home/dss/HierarchicalLIV/Data/posteriors_SBPL_L22/'
    samples = posterior_L22(current_dir_L22, '2th-post_equal_weights.dat').copy()
if hyperLIV_du_n1:
    current_dir_du = '../Data/posteriors_cubicspline_DU25/result_subL_n1_cubicspline/'
    samples = posterior_du(current_dir_du).copy()
if hyperLIV_du_n2:
    current_dir_du = '../Data/posteriors_cubicspline_DU25/result_subL_n2_cubicspline/'
    samples = posterior_du(current_dir_du).copy()

# Add prior to samples, for all event we set a uniform prior
for sample in samples:
    sample["prior"] = 1.0e-6

# Define priors
if hyperLIV_liu_n1 or hyperLIV_du_n1:
    hp_priors = dict(mu=Uniform(0, 20, "mu"), sigma=Uniform(0, 10, "sigma"))
if hyperLIV_liu_n2 or hyperLIV_du_n2:
    hp_priors = dict(mu=Uniform(0, 15, "mu"), sigma=Uniform(0, 10, "sigma"))

# Select the hyper-prior
hyper_prior = gaussian_prior if gauss else log_normal_prior
hyper_tag = "gauss" if gauss else "log"

# Set up likelihood and sampler
hp_likelihood = HyperparameterLikelihood(
    posteriors=samples,
    hyper_prior=hyper_prior,
    log_evidences=0,
    max_samples=5000,
)

# Set model tag based on hyperLIV flags
if hyperLIV_liu_n1:
    model_tag = "EQG1_Liu"
elif hyperLIV_liu_n2:
    model_tag = "EQG2_Liu"
elif hyperLIV_du_n1:
    model_tag = "cubicspline_subLn1"
else:
    model_tag = "cubicspline_subLn2"

outdir = f"./results_main/outdir_{hyper_tag}_{model_tag}"
check_directory_exists_and_if_not_mkdir(outdir)

label = f"hyperLIV_{hyper_tag}"
result = run_sampler(
    likelihood=hp_likelihood,
    priors=hp_priors,
    sampler="dynesty",
    nlive=1500,
    use_ratio=False,
    outdir=outdir,
    label=label,
    verbose=True,
    clean=True,
)

# result.plot_corner(save=True, priors=hp_priors)
# import matplotlib.pyplot as plt

result.plot_corner(priors=hp_priors)
# plt.savefig(f"{outdir}/corner.png")
# plt.close()
