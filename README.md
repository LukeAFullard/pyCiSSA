## pyCiSSA

A Python package implementing Circulant Singular Spectrum Analysis (CiSSA) for time series decomposition, reconstruction, and significance testing. 
Please check out the original Matlab verion written by the creator of the CiSSA method - https://github.com/jbogalo/CiSSA

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Module Overview](#module-overview)

   * [Preprocessing](#preprocessing)
   * [Core CiSSA Algorithm](#core-cissa-algorithm)
   * [Postprocessing](#postprocessing)
5. [Examples](#examples)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

* **Gap Filling**: Robust handling of missing values before analysis .
* **CiSSA Core**: Circulant Singular Spectrum Analysis for extracting oscillatory components, trend, noise.
* **Time-Frequency Analysis**: Compute and visualize the instantaneous frequency and amplitude of reconstructed components.
* **Trend Extraction**: Automated extraction of the trend component using CiSSA.
* **Noise Removal**: Automated noise removal using CiSSA.
* **Monte Carlo Significance Testing**: Evaluate component significance with surrogate data tests.

---

## Installation

```bash
# Clone the repository and switch to the pycissa_v2 branch
git clone -b pycissa_v2 https://github.com/LukeAFullard/pyCiSSA.git
cd pyCiSSA

# Install dependencies via Poetry
poetry install
```

> **Note**: Python 3.8+ is required. All dependencies are managed via `pyproject.toml`.

---

## Quick Start

```python
import numpy as np
from pycissa import Cissa

# 1. Prepare equally spaced time array `t` and data array `x`
N = 500
t = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(N)

# 2. Initialize Cissa
#    The window length L critically influences frequency resolution and trend separation.
cissa = Cissa(t, x)

# 3. Run the full automated pipeline
#    auto_cissa: fixes censoring/nan, plots original, fits CiSSA, Monte Carlo test, grouping, frequency-time, trend, autocorrelation, periodogram citeturn1file3
cissa.auto_cissa(L=50, alpha=0.05, K_surrogates=5, surrogates='random_permutation')

# 4. Retrieve results and figures
#    - Numerical outputs in `cissa.results['cissa']`
#    - Matplotlib figures in `cissa.figures['cissa']`
print(cissa.figures['cissa'].keys())

# 5. Use standalone auto-functions if required
#    • auto_fix_censoring_nan: clean outliers & NaNs citeturn1file4
#    • auto_denoise: denoise signal and plot citeturn1file0
#    • auto_detrend: detrend signal and plot citeturn1file1
cissa.auto_fix_censoring_nan(L=50)
cissa.auto_denoise(L=50, plot_denoised=True)
cissa.auto_detrend(L=50, plot_result=True)
```

> **Note**: Always choose `L` (window length) between \~N/3 to N/2 as a starting point, then inspect the eigenvalue spectrum to fine-tune. The default behavior of auto-functions uses `L = floor(N/2)` if `L` is omitted. citeturn1file3

---

## Module Overview

This package exposes a single class, `Cissa`, which encapsulates the full CiSSA workflow:

* **Initialization**

  * `Cissa(t, x)`: Create an instance with time array `t` (1D, equally spaced) and data array `x` (same length).

* **Automated Pipelines**

  * `auto_fix_censoring_nan(L)`: Impute missing or censored values before analysis.
  * `auto_cissa(L, alpha, K_surrogates, surrogates)`: Run the complete pipeline—cleaning, decomposition, Monte Carlo testing, grouping, time-frequency analysis, trend analysis, and diagnostic plots.
  * `auto_denoise(L, plot_denoised)`: Perform denoising and plot the denoised series.
  * `auto_detrend(L, plot_result)`: Perform detrending and plot the trend vs. detrended signal.

* **Postprocessing Helpers**
  These methods are available on the `Cissa` instance after `fit` or `auto_cissa`:

  * `post_run_monte_carlo_analysis(alpha, K_surrogates, surrogates)`: Monte Carlo significance testing.
  * `post_group_components(grouping_type)`: Automatic grouping of oscillatory components.
  * `post_run_frequency_time_analysis()`: Instantaneous frequency and amplitude calculation.
  * `post_analyse_trend()`: Trend extraction and smoothing.
  * `plot_autocorrelation()`: Autocorrelation of residuals.
  * `post_periodogram_analysis()`: Periodogram of the original and reconstructed signals.

---

## API Reference

Since `Cissa` encapsulates all functionality, the public API comprises:

```python
from pycissa import Cissa, __version__
```

* **Cissa**
  Full-featured class for CiSSA analysis. See docstrings in `pycissa/processing/cissa/cissa.py` for complete parameter listings and return values.

* ****version****
  Package version string.

---

Explore the `examples/` directory for Jupyter notebooks.

---

## API Reference

Detailed API documentation is available in the `docs/` folder (coming soon) or via the docstrings in each module.

---

## Testing

Run unit tests with pytest:

```bash
pytest tests/
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and create a new branch.
2. Follow the existing code style (PEP8) and add tests.
3. Submit a pull request describing your changes.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
