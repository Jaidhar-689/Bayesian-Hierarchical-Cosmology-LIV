# Hierarchical Test of Lorentz Invariance with Gamma-Ray Burst Spectral-Lag Measurements

---

## Overview

**GRB_LIV_Hierarchical** contains the code and data required to reproduce the results presented in the paper ["Hierarchical Test of Lorentz Invariance with Gamma-Ray Burst Spectral-Lag Measurements"](https://arxiv.org/abs/2512.22875), authored by Shen-Shi Du, Yi Gong, Jun-Jie Wei, Zi-Ke Liu, Zhi-Qiang You, Yan-Zhi Meng, and Xing-Jiang Zhu.

This repository includes scripts for:
1. Conducting single-GRB Bayesian inference to constrain the quantum-gravity energy scale within a Taylor-expansion framework.
2. Performing hierarchical Bayesian inference on the quantum-gravity energy scale, utilizing posterior samples obtained from individual GRBs.

## Folder Contents

- **`Data`**: 
  - `Data/lag_data/lag_err_fermi_32grbs/`: Contains the spectral-lag data for 32 GRBs. Each file, named `GRBName.txt`, lists the energy bands used to calculate the time lags (with the lowest energy band as the reference), along with the 1$\sigma$ errors for both the energy bands and the time lags.
  - `Data/lag_data/GRBPARAM.csv`: Contains information about the energy bands used to extract the Fermi-GBM light curves for each GRB, as well as the redshift for each burst.
  - `Data/posteriors_cubicspline_DU25/`: Contains the posterior samples of $log_{10}(E_{\rm QG,n})$ derived from single-burst Bayesian analysis using a cubic spline to fit the source-intrinsic time lags in the linear (or quadratic) Lorentz Invariance violation (LIV) scenario.
  - `Data/posteriors_SBPL_L22/`: Contains results from single-burst Bayesian analysis using a smooth broken power-law (SBPL) function, published by [Liu et al.](https://doi.org/10.3847/1538-4357/ac81b9).
  - `Data/lnBayesFactor.txt`: Provides the e-based logarithmic Bayes factors comparing the Cubic Spline and SBPL models.

- **`Demos`**:
  - `Demos/single_burst_inference.py`: Script for performing single-burst Bayesian analysis.
  - `Demos/hierarchical_inference.py`: Script for conducting hierarchical Bayesian inference. The results can be found in `Demos/results_main`.
  - `Demos/plot_figures_and_analyse_results.ipynb`: Contains the code to generate Figures 1–5.
  - `Demos/AppendixB_analysis_and_results`: Provides the code and data to reproduce the analysis and plots in Appendix B of the paper.

- **`Figures`**: Contains all the figures used in the paper.

## Main Requirements

* [BILBY](https://github.com/bilby-dev/bilby)
* [dynesty](https://github.com/joshspeagle/dynesty/tree/master)
* [Scipy](https://github.com/scipy/scipy)
* [Numpy](https://github.com/numpy/numpy)

## License

This project is licensed under the MIT License.

## Citation Guide

To cite this work, please use the following reference: [DOI: 10.48550/arXiv.2512.22875](https://doi.org/10.48550/arXiv.2512.22875).

To use the spectral-lag data, please refer to the work by [Liu et al.](https://doi.org/10.3847/1538-4357/ac81b9).
