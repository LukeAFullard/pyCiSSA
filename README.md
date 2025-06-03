# pycissa: Advanced Time Series Analysis

`pycissa` is a powerful Python library designed for advanced time series analysis. Its core methodology revolves around **Circulant Singular Spectrum Analysis (CISSA)**, a sophisticated technique for decomposing time series data.

## Purpose

The primary goal of `pycissa` is to provide tools for:

*   **Denoising:** Separating underlying signals from noise.
*   **Detrending:** Identifying and removing trends to analyze cyclical patterns or residuals.
*   **Seasonality Analysis:** Extracting and understanding seasonal components within the data.
*   **Significance Testing:** Employing Monte Carlo methods to assess the statistical significance of decomposed components.

`pycissa` is well-suited for researchers, data scientists, and analysts working with time series data across various domains, including economics, finance, environmental science, and engineering.

## Installation

You can install `pycissa` using pip:
```bash
pip install pycissa
```

## Core Concepts: Circulant Singular Spectrum Analysis (CISSA)

At the heart of `pycissa` lies **Circulant Singular Spectrum Analysis (CISSA)**. CISSA is an advanced technique used for decomposing a time series into its constituent parts. This decomposition typically allows for the identification and extraction of:

*   **Trend:** The underlying long-term direction of the series.
*   **Oscillatory Components:** Periodic or quasi-periodic patterns, such as seasonality or business cycles.
*   **Noise:** The random, irregular fluctuations remaining after accounting for trend and oscillations.

The "circulant" aspect refers to the specific way the trajectory matrix (a key component in SSA-based methods) is constructed, often providing advantages in computational efficiency and handling of time series endpoints.

By separating these components, `pycissa` enables a deeper understanding of the underlying dynamics of the time series, facilitating tasks like forecasting, anomaly detection, and signal extraction.

For a detailed understanding of the CISSA methodology as implemented and referenced in this library, users are encouraged to consult the primary academic paper:

*   Bógalo, J., Poncela, P., & Senra, E. (2021). "Circulant singular spectrum analysis: a new automated procedure for signal extraction". *Signal Processing, 179*, 107824. [https://doi.org/10.1016/j.sigpro.2020.107824](https://doi.org/10.1016/j.sigpro.2020.107824)

(Note: The original MATLAB version by the paper's authors can also be found at [https://github.com/jbogalo/CiSSA](https://github.com/jbogalo/CiSSA)).

## Basic Usage / Getting Started

To begin using `pycissa`, you'll primarily interact with the `Cissa` class. Here's how to get started:

1.  **Import the `Cissa` class:**

    ```python
    from pycissa import Cissa
    import numpy as np # For creating sample data
    import pandas as pd # For creating sample data if using pandas Series/DataFrames
    ```

2.  **Prepare your time series data:**
    Your time series data should consist of two main components:
    *   **Time array (`t`):** A sequence of time points (e.g., dates, numerical indices). This should be a 1D NumPy array or a pandas Series.
    *   **Value array (`x`):** The corresponding values of the time series at each time point. This should also be a 1D NumPy array or a pandas Series.

    *Ensure your data is clean (e.g., handle NaNs or censored data if necessary, though `pycissa` also provides tools for this - see Key Features).*

3.  **Create a `Cissa` object:**
    Instantiate the `Cissa` class by passing your time and value arrays.

    ```python
    # Example with NumPy arrays
    # t = np.array([...]) # Your time points
    # x = np.array([...]) # Your data values
    # cissa_object = Cissa(t=t, x=x)

    # Example with pandas Series (assuming 'date_column' and 'value_column' in a DataFrame 'df')
    # df = pd.read_csv('your_data.csv', parse_dates=['date_column']) # Or your data loading method
    # t_series = df['date_column']
    # x_series = df['value_column']
    # cissa_object = Cissa(t=t_series, x=x_series)
    ```

Once you have your `cissa_object`, you can then call its various methods to perform analysis, such as the automated pipelines detailed below.

## Automated Pipelines

`pycissa` provides several high-level "automated pipeline" methods within the `Cissa` class. These functions streamline common time series analysis workflows by combining multiple preprocessing, fitting, and postprocessing steps. The main input for these is typically the **window length (`L`)**, which is crucial for SSA-based methods. Many other parameters for finer control can be passed as keyword arguments (`**kwargs`).

### 1. `auto_cissa()`

*   **Purpose:** Performs a comprehensive automated CISSA procedure. This is often the main go-to function for a full analysis.
*   **Signature:** `cissa_object.auto_cissa(L=None, **kwargs)`
    *   `L` (int, optional): The CISSA window length. If `None`, it defaults to half the time series length.
    *   `**kwargs`: Keyword arguments to customize various underlying steps (e.g., Monte Carlo parameters, grouping criteria, plotting options).
*   **Key Steps Performed:**
    1.  **Data Cleaning (`auto_fix_censoring_nan`):** Automatically handles censored data and NaN values.
    2.  **Plot Original Series:** Displays the input time series.
    3.  **CISSA Fitting (`fit`):** Applies the core CISSA algorithm.
    4.  **Monte Carlo Significance (`post_run_monte_carlo_analysis`):** Tests the statistical significance of decomposed components.
    5.  **Component Grouping (`post_group_components`):** Groups components into trend, periodic, and noise.
    6.  **Frequency-Time Analysis (`post_run_frequency_time_analysis`):** Generates frequency-time plots.
    7.  **Trend Analysis (`post_analyse_trend`):** Analyzes and plots the extracted trend.
    8.  **Autocorrelation (`plot_autocorrelation`):** (Optional) Plots ACF/PACF for the original series and residuals.
    9.  **Periodogram Analysis (`post_periodogram_analysis`):** (Optional) Analyzes fractal scaling and Hurst exponent.
*   **Example:**
    ```python
    # Assuming cissa_object is already created
    # L_window = 60 # Example: window length of 60 data points (e.g., 5 years for monthly data)
    # cissa_object.auto_cissa(L=L_window, K_surrogates=5, alpha=0.05)
    ```
*   **Key Outputs:**
    *   Populates `cissa_object.results` with detailed analysis results.
    *   Populates `cissa_object.figures` with generated plots.
    *   Stores reconstructed components: `cissa_object.x_trend`, `cissa_object.x_periodic`, `cissa_object.x_noise`.

### 2. `auto_denoise()`

*   **Purpose:** Automatically denoises a time series by separating the signal (trend + periodic components) from the noise.
*   **Signature:** `cissa_object.auto_denoise(L=None, plot_denoised=True, **kwargs)`
    *   `L` (int, optional): CISSA window length. Defaults to half series length if `None`.
    *   `plot_denoised` (bool, optional): If `True` (default), plots the original vs. denoised signal.
    *   `**kwargs`: Additional arguments for underlying steps.
*   **Key Steps Performed:**
    1.  **Data Cleaning (`auto_fix_censoring_nan`).**
    2.  **CISSA Fitting (`fit`).**
    3.  **Monte Carlo Significance (if `grouping_type` in `kwargs` is 'monte_carlo').**
    4.  **Component Grouping (`post_group_components`)** to identify signal and noise.
*   **Example:**
    ```python
    # cissa_object.auto_denoise(L=L_window)
    ```
*   **Key Outputs:**
    *   `cissa_object.x_denoised`: The denoised time series (trend + periodic).
    *   `cissa_object.x_trend`, `cissa_object.x_periodic`, `cissa_object.x_noise`.
    *   Figure of denoised signal (if `plot_denoised=True`).

### 3. `auto_detrend()`

*   **Purpose:** Automatically detrends a time series by identifying and separating the trend component.
*   **Signature:** `cissa_object.auto_detrend(L=None, plot_result=True, **kwargs)`
    *   `L` (int, optional): CISSA window length. Defaults to half series length if `None`.
    *   `plot_result` (bool, optional): If `True` (default), plots the original series, trend, and detrended series.
    *   `**kwargs`: Additional arguments.
*   **Key Steps Performed:**
    1.  **Data Cleaning (`auto_fix_censoring_nan`).**
    2.  **CISSA Fitting (`fit`).**
    3.  **Component Grouping (manual style):** Separates the trend component (typically the first component or group) from the rest (detrended signal).
    4.  **Trend Analysis (`post_analyse_trend`):** Further analyzes the extracted trend.
*   **Example:**
    ```python
    # cissa_object.auto_detrend(L=L_window, trend_type='rolling_OLS', window=12)
    ```
*   **Key Outputs:**
    *   `cissa_object.x_trend`: The extracted trend.
    *   `cissa_object.x_detrended`: The detrended time series (original - trend).
    *   Figure of detrended signal (if `plot_result=True`).

### 4. `auto_cissa_classic()`

*   **Purpose:** Implements an automated CISSA procedure that is more faithful to an original MATLAB version of CISSA (by J. Bógalo et al.). This method uses a specific manual grouping strategy.
*   **Signature:** `cissa_object.auto_cissa_classic(I, L=None, **kwargs)`
    *   `I` (int, float, or dict): Grouping criteria, similar to the MATLAB version's input.
        *   *Positive integer:* Number of data points per year (for automatic trend, business cycle, seasonality grouping).
        *   *Dictionary:* User-defined groups of component indices.
        *   *Float (0 to 1):* Cumulative PSD share for component selection.
        *   *Float (-1 to 0):* PSD percentile threshold for component selection.
    *   `L` (int, optional): CISSA window length. Defaults to half series length if `None`.
    *   `**kwargs`: Additional arguments (e.g., `season_length`, `cycle_length` if `I` is an integer).
*   **Key Steps Performed:**
    1.  **Data Cleaning (`auto_fix_censoring_nan`).**
    2.  **Plot Original Series.**
    3.  **CISSA Fitting (`fit`).**
    4.  **Manual Component Grouping (`post_group_manual`):** Groups components based on the `I` parameter.
*   **Example:**
    ```python
    # Assuming monthly data, so I=12 for yearly seasonality
    # cissa_object.auto_cissa_classic(I=12, L=L_window)
    ```
*   **Key Outputs:**
    *   Reconstructed components based on grouping `I` (e.g., `cissa_object.x_trend`, `cissa_object.x_seasonality`, `cissa_object.x_long_term_cycle`, `cissa_object.x_noise` if `I` is an integer).
    *   Results stored in `cissa_object.results['cissa']['manual']`.

### Helper Pipeline: `auto_fix_censoring_nan()`

*   **Purpose:** A utility pipeline specifically for data cleaning. It's called by the other `auto_` methods but can also be used standalone if only preprocessing is needed before a custom analysis.
*   **Signature:** `cissa_object.auto_fix_censoring_nan(L, **kwargs)`
    *   `L` (int): CISSA window length (required by `pre_fill_gaps` if NaNs are present).
    *   `**kwargs`: Arguments for `pre_fix_censored_data` and `pre_fill_gaps`.
*   **Key Steps Performed:**
    1.  Calls `pre_fix_censored_data` if censored data is detected.
    2.  Calls `pre_fill_gaps` if NaN data is detected.
*   **Example:**
    ```python
    # cissa_object.auto_fix_censoring_nan(L=L_window)
    ```
*   **Key Outputs:**
    *   Modifies `cissa_object.x` and `cissa_object.t` in place.
    *   Updates `cissa_object.censored` and `cissa_object.isnan` flags.

These automated pipelines provide convenient entry points to the rich functionality of `pycissa`. For more granular control, users can call the individual `pre_`, `fit`, and `post_` methods directly.

## Key Features & Functionality Overview

Beyond the automated pipelines, the `Cissa` class offers a rich set of methods for more granular control over your time series analysis workflow. These can be broadly categorized as follows:

### Preprocessing (`pre_*` methods)

These methods help in preparing your data before applying CISSA:

*   **`cissa_object.restore_original_data()`**: Restores the raw time series data (`t_raw`, `x_raw`) if it was modified by preprocessing steps.
*   **`cissa_object.pre_fill_gaps(L, ...)`**: An iterative algorithm to fill in missing (NaN) values or replace outliers. It uses CISSA reconstructions and offers various component selection strategies (Monte Carlo, dropping smallest N, etc.) and error estimation.
*   **`cissa_object.pre_fix_censored_data(...)`**: Handles censored data points (e.g., values like `<1` or `>10`) by converting them into numerical representations based on specified rules (raw value, multiplier, constant).
*   **`cissa_object.pre_fix_missing_samples(...)`**: (Work in Progress) Aims to identify and fill in missing time steps to ensure the series is approximately evenly spaced, which is important for CISSA.

### Fitting (`fit` method)

This is the core method that applies the Circulant Singular Spectrum Analysis:

*   **`cissa_object.fit(L, extension_type='AR_LR', ...)`**: Performs the main CISSA decomposition of the time series `x` using a window length `L`.
    *   `extension_type`: Specifies how to handle the endpoints of the time series (e.g., 'AR_LR' for autoregressive extension, 'Mirror').
    *   Outputs include the elementary reconstructed components (`Z`) and the power spectral density (`psd`) of the circulant matrix.

### Postprocessing (`post_*` methods)

After fitting CISSA, these methods allow for detailed analysis and interpretation of the components:

*   **`cissa_object.post_run_frequency_time_analysis(data_per_period, ...)`**: Generates frequency-time and period-time matrices and plots, helping to visualize how the spectral content of components evolves over time.
*   **`cissa_object.post_analyse_trend(trend_type='rolling_OLS', ...)`**: Calculates and visualizes the trend component, offering methods like linear regression or rolling Ordinary Least Squares (OLS).
*   **`cissa_object.post_run_monte_carlo_analysis(alpha=0.05, K_surrogates=1, surrogates='random_permutation', ...)`**: Performs Monte Carlo significance testing on the extracted CISSA components. It compares the power spectral density (PSD) of each component against those from surrogate data (e.g., random permutations, AR(1) surrogates) to determine statistical significance.
*   **`cissa_object.post_group_manual(I, ...)`**: Allows for manual grouping of reconstructed components based on user-defined criteria (e.g., by number of data points per year for seasonality, by specific frequency indices, or by PSD share).
*   **`cissa_object.post_group_components(grouping_type='monte_carlo', ...)`**: Automatically groups CISSA components into 'trend', 'periodic', and 'noise' categories based on strategies like Monte Carlo significance, smallest PSD proportion, or dropping the N smallest components. It can also perform statistical tests (e.g., Ljung-Box, normality) on the resulting noise component.
*   **`cissa_object.post_periodogram_analysis(...)`**: Analyzes the fractal scaling properties of the time series (or its components) using periodogram methods (e.g., Lomb-Scargle for unevenly spaced data resulting from component selection). It can also calculate the Hurst exponent.

### Plotting

Several methods are dedicated to or include plotting capabilities:

*   **`cissa_object.plot_original_time_series()`**: Plots the (potentially preprocessed) time series.
*   **`cissa_object.plot_autocorrelation(...)`**: Generates plots of the time series along with its Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). Can also be used for residuals.
*   **`cissa_object.plot_seasonal_boxplots(...)`**: Creates monthly or yearly boxplots to visualize seasonal patterns, with options to split data by a date or remove the trend first.
*   Many of the `auto_*` and `post_*` methods also generate and store relevant figures in `cissa_object.figures`.

### Prediction

*   **`cissa_object.predict()`**: (Future Implementation) Intended for time series forecasting, currently not implemented.

This overview highlights the main individual methods available. Users can combine these flexibly to create custom analysis workflows tailored to their specific needs. Refer to the method docstrings and example notebooks for detailed parameter options and usage.

## Examples

The `examples/` directory in this repository contains several Jupyter notebooks demonstrating the usage of `pycissa` for different tasks. These notebooks provide practical code and visualizations to help you understand how to apply the library to your own time series data.

Key examples include:

*   **`examples/cissa/auto_cissa_examples/Auto-Cissa.ipynb`**:
    *   A great starting point that showcases the use of the main `auto_cissa()` automated pipeline for a comprehensive analysis.
    *   Demonstrates loading data, running the pipeline, and inspecting various outputs like component plots and Monte Carlo significance tests.

*   **`examples/cissa/auto_cissa_examples/`**: This sub-directory contains further examples focusing on automated features:
    *   `Automatic denoising - Monte Carlo Surrogates.ipynb` (and variations for different window lengths `L`): Illustrates the `auto_denoise` capability and how Monte Carlo surrogates are used for significance testing of components.
    *   `Automatic denoising - number, proportion of components.ipynb`: Shows alternative component grouping strategies for denoising.
    *   `Automatic trend removal.ipynb`: Demonstrates the `auto_detrend` functionality.

*   **`examples/cissa/gap_filling/`**: This sub-directory provides notebooks on using `pycissa` for imputing missing data:
    *   `Gap filling - Monte Carlo.ipynb`: Shows gap filling with Monte Carlo based component selection.
    *   `Gap filling - Effect of window length L.ipynb`: Explores the impact of window length on imputation quality.
    *   `Gap filling - effect of convergence threshold.ipynb`: Examines how convergence criteria affect the iterative gap-filling process.

*   **`examples/cissa/validation_tests/`**:
    *   `Matlab_vs_Python_Comparison_auto_cissa_classic.ipynb`: Provides a comparison with an original MATLAB implementation, useful for validation and understanding the `auto_cissa_classic` method.

We encourage you to explore these notebooks to see `pycissa` in action and adapt the code snippets for your specific use cases.

## Output and Results

When you use the methods of the `Cissa` class, the results, intermediate calculations, and generated figures are stored as attributes of your `cissa_object`.

### Key Data Outputs:

*   **`cissa_object.results` (dict):**
    *   This is a primary dictionary where most numerical results and parameters from various analysis steps are stored.
    *   It's often structured, for example, results from the core `fit` method might be under `cissa_object.results['cissa']`, Monte Carlo results under `cissa_object.results['cissa']['components'][component_key]['monte_carlo']`, etc.
    *   Explore this dictionary to find detailed outputs like Power Spectral Density (PSD) values, component shares, statistical test outcomes, and model parameters.

*   **Reconstructed Time Series Components (NumPy arrays):**
    After running pipelines like `auto_cissa`, `auto_denoise`, or `auto_detrend`, or after specific grouping methods, you can directly access the key reconstructed time series:
    *   `cissa_object.x_original`: The (potentially preprocessed) input series used for the last fit.
    *   `cissa_object.x_trend`: The extracted trend component.
    *   `cissa_object.x_periodic`: The sum of significant periodic (oscillatory) components.
    *   `cissa_object.x_noise`: The residual noise component.
    *   `cissa_object.x_denoised`: The signal after noise removal (typically `x_trend + x_periodic`).
    *   `cissa_object.x_detrended`: The signal after trend removal (typically `x_periodic + x_noise`).
    *   If using `auto_cissa_classic` with `I` as an integer, components like `cissa_object.x_seasonality` and `cissa_object.x_long_term_cycle` might also be populated.
    *   Individual reconstructed components from the CISSA decomposition are available in `cissa_object.results['cissa']['components'][component_key]['reconstructed_data']`.

*   **`cissa_object.t` (NumPy array):**
    *   The time array corresponding to the `x_*` components. This might be modified from your original `t_raw` if preprocessing steps like `pre_fix_missing_samples` were applied.

### Figures:

*   **`cissa_object.figures` (dict):**
    *   This dictionary stores `matplotlib` figure objects generated by various plotting methods (e.g., `plot_original_time_series`, `auto_cissa`, `post_analyse_trend`).
    *   Figures are typically nested, e.g., `cissa_object.figures['cissa']['figure_split_components']`.
    *   You can display these figures in a Jupyter environment by simply calling the figure object in a cell or save them using `fig.savefig('filename.png')`.

    ```python
    # Example: Displaying a generated figure in a Jupyter Notebook
    # fig_trend = cissa_object.figures.get('cissa', {}).get('figure_trend')
    # if fig_trend:
    #     fig_trend
    ```

### Information Text:

*   **`cissa_object.information_text` (str):**
    *   Contains a summary of key findings and operations performed, such as the number of censored points fixed, gap-filling RMSE, component variance shares, and Monte Carlo significance results. This can be useful for a quick textual overview.

By inspecting these attributes, you can access all the detailed outputs from your `pycissa` analyses for further investigation, reporting, or integration into other workflows.

## Citing / References

The Circulant Singular Spectrum Analysis (CISSA) methodology and related techniques implemented or referenced in this library are based on academic research. If you use `pycissa` in your work, please consider citing the relevant publications:

1.  **Primary CISSA Methodology:**
    *   Bógalo, J., Poncela, P., & Senra, E. (2021). "Circulant singular spectrum analysis: a new automated procedure for signal extraction". *Signal Processing, 179*, 107824.
        *   DOI: [https://doi.org/10.1016/j.sigpro.2020.107824](https://doi.org/10.1016/j.sigpro.2020.107824)

2.  **Application & Gap Filling Reference (Seasonality in COVID-19):**
    *   Bógalo, J., Llada, M., Poncela, P., & Senra, E. (2022). "Seasonality in COVID-19 times". *Economics Letters, 211*, 110206.
        *   DOI: [https://doi.org/10.1016/j.econlet.2021.110206](https://doi.org/10.1016/j.econlet.2021.110206)
        *(This paper is referenced in the context of gap-filling techniques within pycissa).*

### Original MATLAB Code

The authors of the primary CISSA paper also provide a MATLAB implementation, which can be found at:
*   [https://github.com/jbogalo/CiSSA](https://github.com/jbogalo/CiSSA)

Please refer to these sources for a deeper theoretical understanding of the methods.

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
