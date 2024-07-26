import numpy as np

###############################################################################
###############################################################################
###############################################################################
def calculate_autocorrelation_function(x:                np.ndarray,
                                       adjusted:         bool = False, 
                                       nlags:            int|None = None, 
                                       qstat:            bool = False, 
                                       fft:              bool = True, 
                                       alpha:            float = 0.05, 
                                       bartlett_confint: bool = True, 
                                       missing:          str = 'none'):
    '''
    Wrapper of statsmodels acf to calculate autocorrelation function.
    [1]
    Parzen, E., 1963. On spectral analysis with missing observations and amplitude modulation. Sankhya: The Indian Journal of Statistics, Series A, pp.383-392.
    
    [2]
    Brockwell and Davis, 1987. Time Series Theory and Methods
    
    [3]
    Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: The time series data.
    adjusted : bool, optional
        DESCRIPTION: If True, then denominators for autocovariance are n-k, otherwise n. The default is False.
    nlags : int|None, optional
        DESCRIPTION: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs - 1). The returned value includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,). The default is None.
    qstat : bool, optional
        DESCRIPTION: If True, returns the Ljung-Box q statistic for each autocorrelation coefficient. See q_stat for more information. The default is False.
    fft : bool, optional
        DESCRIPTION: If True, computes the ACF via FFT. The default is True.
    alpha : float, optional
        DESCRIPTION: If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to Bartlett”s formula. The default is 0.05.
    bartlett_confint : bool, optional
        DESCRIPTION: Confidence intervals for ACF values are generally placed at 2 standard errors around r_k. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2 in [3]. For the ACF of raw data, the standard error at a lag k is found as if the right model was an MA(k-1). This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an MA of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett’s formula. For more details on Bartlett formula result, see section 7.2 in [2]. The default is True.
    missing : str, optional
        DESCRIPTION: A string in [“none”, “raise”, “conservative”, “drop”] specifying how the NaNs are to be treated. “none” performs no checks. “raise” raises an exception if NaN values are found. “drop” removes the missing observations and then estimates the autocovariances treating the non-missing as contiguous. “conservative” computes the autocovariance using nan-ops so that nans are removed when computing the mean and cross-products that are used to estimate the autocovariance. When using “conservative”, n is set to the number of non-missing observations. The default is 'none'.

    Returns
    -------
    acf : np.ndarray
        DESCRIPTION: The autocorrelation function for lags 0, 1, …, nlags. Shape (nlags+1,).
    confidence_interval : np.ndarray
        DESCRIPTION: Confidence intervals for the ACF at lags 0, 1, …, nlags. Shape (nlags + 1, 2). Returned if alpha is not None.
    q_stat : np.ndarray
        DESCRIPTION: The Ljung-Box Q-Statistic for lags 1, 2, …, nlags (excludes lag zero). Returned if q_stat is True.
    p_values : np.ndarray
        DESCRIPTION: The p-values associated with the Q-statistics for lags 1, 2, …, nlags (excludes lag zero). Returned if q_stat is True.

    '''
    from statsmodels.tsa.stattools import acf
    acf,confidence_interval,q_stat,p_values = acf(x=x, 
                                                  adjusted=adjusted, 
                                                  nlags=nlags, 
                                                  qstat=qstat, 
                                                  fft=fft, 
                                                  alpha=alpha, 
                                                  bartlett_confint=bartlett_confint, 
                                                  missing=missing)
    return acf,confidence_interval,q_stat,p_values

###############################################################################
###############################################################################
###############################################################################
def calculate_partial_autocorrelation_function(x:      np.ndarray, 
                                               nlags:  int|None = None, 
                                               method: str = 'ywadjusted', 
                                               alpha:  float = 0.05):
    '''
    Wrapper of statsmodels acf to calculate partial autocorrelation function.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Observations of time series for which pacf is calculated.
    nlags : int|None, optional
        DESCRIPTION: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,). The default is None.
    method : str, optional
        DESCRIPTION: Specifies which method for the calculations to use.
                    “yw” or “ywadjusted” : Yule-Walker with sample-size adjustment in denominator for acovf. Default.
                    “ywm” or “ywmle” : Yule-Walker without adjustment.
                    “ols” : regression of time series on lags of it and on constant.
                    “ols-inefficient” : regression of time series on lags using a single common sample to estimate all pacf coefficients.
                    “ols-adjusted” : regression of time series on lags with a bias adjustment.
                    “ld” or “ldadjusted” : Levinson-Durbin recursion with bias correction.
                    “ldb” or “ldbiased” : Levinson-Durbin recursion without bias correction.
                    “burg” : Burg”s partial autocorrelation estimator.
                    The default is 'ywadjusted'.
    alpha : float, optional
        DESCRIPTION. If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to 1/sqrt(len(x)). The default is 0.05.

    Returns
    -------
    pact : np.ndarray
        DESCRIPTION: The partial autocorrelations for lags 0, 1, …, nlags. Shape (nlags+1,).
    confidence_interval : np.ndarray
        DESCRIPTION: Confidence intervals for the PACF at lags 0, 1, …, nlags. Shape (nlags + 1, 2). Returned if alpha is not None.

    '''
    from statsmodels.tsa.stattools import pacf
    pact,confidence_interval = pacf(x=x, 
                                    nlags=nlags,
                                    method=method,
                                    alpha=alpha)
    return pact,confidence_interval