import numpy as np
import warnings


def initial_data_checks(t: np.ndarray, x: np.ndarray):
    '''
    Data checks to ensure t,x are numpy arrays of the correct shape.
    Will try to convert to the correct shape if they are not

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of input times.
    x : np.ndarray
        DESCRIPTION: Array of input data.

    Raises
    ------
    ValueError
        DESCRIPTION: Exception raised if the input variables are either not arrays or can't be converted to an array, or if the array shape is incorrect.'

    Returns
    -------
    t : np.ndarray
        DESCRIPTION: Array of (possible reshaped) input times.
    x : np.ndarray
        DESCRIPTION: Array of (possible reshaped) input data.
    '''
    ######################################
    #check x is a numpy array
    if not type(x) is np.ndarray:
        try: 
            x = np.array(x)
            x = x.reshape(len(x),)
        except: raise ValueError(f'Input "x" is not a numpy array, nor can be converted to one.')
    myshape = x.shape
    if not len(myshape) == 1:
        try: 
            x = x.reshape(len(x),)
        except:
            raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({myshape})')
            
    ######################################        
    #check t is a numpy array
    if not type(t) is np.ndarray:
        try: 
            t = np.array(t)
            t = t.reshape(len(t),)
        except: raise ValueError(f'Input "t" is not a numpy array, nor can be converted to one.')
    myshape = t.shape
    
    if not len(myshape) == 1:
        try: 
            t = t.reshape(len(t),)
        except:
            raise ValueError(f'Input "t" should be a column vector (i.e. only contain a single column). The size of t is ({myshape})')
    return t,x


class Cissa:
    '''
    Circulant Singular Spectrum Analysis: Data must be equally spaced!
    '''
    def __init__(self, t: np.ndarray, x: np.ndarray):
        #----------------------------------------------------------------------
        # perform initial checks to ensure input variables are numpy arrays of the correct shape.
        t,x = initial_data_checks(t,x)
        self.x_raw = x #array of corresponding data
        self.t_raw = t #array of corresponding data
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #perform check for censored data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data
        self.censored = detect_censored_data(x)
        if self.censored: warnings.warn("WARNING: Censored data detected. Please run pre_fix_censored_data before fitting.")
        #----------------------------------------------------------------------    
        
        #----------------------------------------------------------------------
        #perform check for nan data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(x)
        if self.isnan: warnings.warn("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        self.t = t #array of times
        self.x = x #array of corresponding data
        
        self.figures = {}  #make a space for future figures
        self.figures.update({'cissa':{}})
        #----------------------------------------------------------------------
    def restore_original_data(self):
        '''
        Method to restore original data (x,t) = (x_raw,t_raw)
        '''
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data,detect_nan_data
        self.x = self.x_raw
        self.t = self.t_raw
        self.censored = detect_censored_data(self.x)  #if we restore the data we must check if the restored data is censored again...
        self.isnan = detect_nan_data(self.x)
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------    
    def fit(self,
            L: int,
            extension_type: str = 'AR_LR',
            multi_thread_run: bool = True,
            generate_toeplitz_matrix: bool = False):
        '''
        Function to fit CiSSA to a timeseries.
        -------------------------------------------------------------------------
        References:
        [1] Bógalo, J., Poncela, P., & Senra, E. (2021). 
            "Circulant singular spectrum analysis: a new automated procedure for signal extraction". 
              Signal Processing, 179, 107824.
            https://doi.org/10.1016/j.sigpro.2020.107824.
        -------------------------------------------------------------------------  

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION: Input array
        L : int
            DESCRIPTION: CiSSA window length.
        extension_type : str
            DESCRIPTION: extension type for left and right ends of the time series. The default is AR_LR 
        multi_thread_run : bool, optional
            DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.
        generate_toeplitz_matrix : bool, optional
            DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False. 

        Returns
        -------
        Z : np.ndarray
            DESCRIPTION: Output CiSSA results.
        psd : np.ndarray
            DESCRIPTION: estimation of the the circulant matrix power spectral density

        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------
        
        #run cissa
        from pycissa.processing.matrix_operations.matrix_operations import run_cissa
        self.Z, self.psd = run_cissa(self.x,
                                      L,
                                      extension_type=extension_type,
                                      multi_thread_run=multi_thread_run,
                                      generate_toeplitz_matrix=generate_toeplitz_matrix)
        
        #generate initial results dictionary
        from pycissa.utilities.generate_cissa_result_dictionary import generate_results_dictionary
        self.results = generate_results_dictionary(self.Z,self.psd,L)

        results = self.results
        results.get('cissa').setdefault('model parameters', {})
        results.get('cissa').setdefault('noise component tests', {})
        results.get('cissa').get('model parameters').update({
            'extension_type'   : extension_type, 
            'L'                : L,
            'multi_thread_run' : multi_thread_run,
            })
        self.results = results
        
        
        #save settings
        self.extension_type = extension_type
        self.L = L
        self.multi_thread_run = multi_thread_run
        
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def predict(self):
        print("FUTURE PREDICTION NOT YET IMPLEMENTED")
        #TO DO, maybe using AutoTS or MAPIE?
        return self
        
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def pre_fill_gaps(self,                     
                  L:                          int,
                  convergence:                list = ['value', 1],
                  extension_type:             str  = 'AR_LR',
                  multi_thread_run:           bool = True,
                  initial_guess:              list = ['previous', 1],
                  outliers:                   list = ['nan_only',None],
                  estimate_error:             bool  = True,
                  test_number:                int = 10,
                  test_repeats:               int = 1,
                  z_value:                    float = 1.96,
                  component_selection_method: str = 'monte_carlo_significant_components',
                  eigenvalue_proportion:      float = 0.95,
                  number_of_groups_to_drop:   int = 1,
                  data_per_unit_period:       int = 1,
                  use_cissa_overlap:          bool = False,
                  drop_points_from:           str = 'Left',
                  max_iter:                   int = 50,
                  verbose:                    bool = False,
                  alpha:                      float = 0.05,
                  **kwargs,
                  ):
        '''
        Function to fill in gaps (NaN values) and/or replace outliers in a timeseries via imputation.
        This is achieved by replacing gaps/outliers with an initial guess and then iteratively running the CiSSA (or overlap-CiSSA) method, keeping some (but not all) of the reconstructed series in each step of the algorithm, until convergence is achieved. 
        Optionally, we evaluate the accuracy of the imputation by testing known points.
        
        -------------------------------------------------------------------------
        References:
        [1] Bógalo, J., Poncela, P., & Senra, E. (2021). 
            "Circulant singular spectrum analysis: a new automated procedure for signal extraction". 
              Signal Processing, 179, 107824.
            https://doi.org/10.1016/j.sigpro.2020.107824.
        [2] Bógalo, J., Llada, M., Poncela, P., & Senra, E. (2022). 
            "Seasonality in COVID-19 times". 
              Economics Letters, 211, 110206.
              https://doi.org/10.1016/j.econlet.2021.110206
        -------------------------------------------------------------------------

        Parameters
        ----------
        t : np.ndarray
            DESCRIPTION: Array of input times.
        x : np.ndarray
            DESCRIPTION: Input time series array which possibly has
        L : int
            DESCRIPTION: CiSSA window length.
        convergence : list, optional
            DESCRIPTION. How to define the convergence of the outlier fitting method.
                         Current options are:
                             1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                             2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                             3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence.
                        The default is ['value', 1].
        extension_type : str, optional
            DESCRIPTION. extension type for left and right ends of the time series. The default is 'AR_LR'.
        multi_thread_run : bool, optional
            DESCRIPTION. Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.. The default is True.
        initial_guess : list, optional
            DESCRIPTION. How to choose the initial guess for the missing data/outliers.
                         Current options are:
                             1) ['max', ''] -- Initial guess is the maximum of the time series (ignoring outliers)
                             2) ['median', ''] -- Initial guess is the median of the time series
                             3) ['value', numeric] -- Initial guess is the provided numeric value, the second entry in the list.
                             4) ['previous', numeric] -- Initial guess is the previous value of the time series (ignoring any outliers) multiplied by the numeric value, the second entry in the input list.
                        The default is ['previous', 1].
        outliers : list, optional
            DESCRIPTION. How to find outliers/missing values and the threshold value. 
                         Current options are:
                             0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                             1) ['<',threshold]  -- classifies all values below the threshold as outliers
                             2) ['>',threshold]  -- classifies all values above the threshold as outliers
                             3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                             4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.
                        The default is ['nan_only',None].
        estimate_error : bool, optional
            DESCRIPTION. Flag which determines if we will be estimating the error in the gap filling or not. The default is True.
        test_number : int  
            DESCRIPTION: Number of known points to remove in each iteration to help validate the error (the larger the longer the code will take to run but more accurate our error estimate). The default is 10.
        test_repeats : int
            DESCRIPTION: Number of times to repeat the gap filling process to estimate error (the larger the longer the code will take to run but more accurate our error estimate). The default is 1.    
        z_value : float, optional
            DESCRIPTION: z-value for confidence interval (= 1.96 for a 95% confidence interval, for example)           
        component_selection_method : str, optional
            DESCRIPTION. Method for choosing the way we drop components from the reconstruction. Current options are 'drop_smallest_n', 'drop_smallest_proportion', 'monte_carlo_significant_components'. The default is 'monte_carlo_significant_components'.
        eigenvalue_proportion : float, optional
            DESCRIPTION. only used if component_selection_method == 'drop_smallest_proportion'.
                         if between 0 and 1, the cumulative proportion psd to keep, or if between -1 and 0, a psd proportion threshold to keep a component.
                         The default is 0.95.
        number_of_groups_to_drop : int, optional
            DESCRIPTION. only used if component_selection_method == 'drop_smallest_n'.
                         Number of components to drop from the reconstruction.
                         The default is 1.
        data_per_unit_period : int, optional
            DESCRIPTION. How many data points per season period. If season is annual, season_length is number of data points in a year.
                         The default is 1.
        use_cissa_overlap : bool, optional
            DESCRIPTION. Whether we use ordinary CiSSA (True) or overlap-Cissa (False). The default is False. The default is False.
        drop_points_from : str, optional
            DESCRIPTION. Only used if use_cissa_overlap == True. If the time series does not divide the overlap exactly, which side to drop data from. The default is 'Left'. 
        max_iter : int, optional
            DESCRIPTION. Maximum number of iterations to check for convergence. The default is 50.
        verbose : bool, optional
            DESCRIPTION. Whether to print some info to the console or not. The default is False.
        alpha : float, optional
            DESCRIPTION. Only used if component_selection_method == 'monte_carlo_significant_components'. Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
        **kwargs : named monte carlo input parameters
        
        Returns  
        -------
        x_ca : np.ndarray
            DESCRIPTION: Array with gaps (possibly) filled.
        error_estimates : np.ndarray|None
            DESCRIPTION: Array of errors associated with the test points
        error_estimates_percentage : np.ndarray|None
            DESCRIPTION: Array of percentage errors associated with the test points.
        error_rmse : float|None
            DESCRIPTION: Root mean squared error of test points.
        error_rmse_percentage : float|None
            DESCRIPTION: Percentage root mean squared error of test points.
        original_points : np.ndarray|None
            DESCRIPTION: Array of the original time series points
        imputed_points : np.ndarray|None
            DESCRIPTION: Array of the imputed time series points.
        fig_errors : matplotlit.figure|None
            DESCRIPTION: Figure plotting the error metrics (accuracy of the gap filling imputation)
        fig_time_series : matplotlit.figure|None
            DESCRIPTION: Figure plotting the time series with imputed values. 

        '''
        from pycissa.preprocessing.gap_fill.gap_filling import fill_timeseries_gaps
        from pycissa.postprocessing.monte_carlo.montecarlo import prepare_monte_carlo_kwargs
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
        K_surrogates, surrogates, seed, sided_test, remove_trend,trend_always_significant, A_small_shuffle, generate_toeplitz_matrix = prepare_monte_carlo_kwargs(kwargs)
        x_ca,error_estimates,error_estimates_percentage,error_rmse,error_rmse_percentage,original_points,imputed_points, fig_errors,fig_time_series = fill_timeseries_gaps(
                                self.t,                     
                                self.x,
                                 L,
                                 convergence=convergence,
                                 extension_type=extension_type,
                                 multi_thread_run=multi_thread_run,
                                 initial_guess=initial_guess,
                                 outliers=outliers,
                                 estimate_error=estimate_error,
                                 z_value=z_value,
                                 component_selection_method=component_selection_method,
                                 eigenvalue_proportion=eigenvalue_proportion,
                                 number_of_groups_to_drop=number_of_groups_to_drop,
                                 data_per_unit_period=data_per_unit_period,
                                 use_cissa_overlap=use_cissa_overlap,
                                 drop_points_from=drop_points_from,
                                 max_iter=max_iter,
                                 test_number=test_number,
                                 test_repeats=test_repeats,
                                 verbose=verbose,
                                 alpha=alpha,
                                 K_surrogates=K_surrogates,
                                 surrogates=surrogates,
                                 seed=seed,   
                                 sided_test=sided_test,
                                 remove_trend=remove_trend,
                                 trend_always_significant=trend_always_significant,
                                 A_small_shuffle=A_small_shuffle,
                                 generate_toeplitz_matrix=generate_toeplitz_matrix,)
        
        self.x = x_ca.reshape(len(x_ca),)
        self.gap_fill_error_estimates            = error_estimates
        self.gap_fill_error_estimates_percentage = error_estimates_percentage
        self.gap_fill_error_rmse                 = error_rmse
        self.gap_fill_error_rmse_percentage      = error_rmse_percentage
        self.gap_fill_original_points            = original_points 
        self.gap_fill_imputed_points             = imputed_points, 
        # self.figure_gap_fill_error               = fig_errors,
        # self.figure_gap_fill                     = fig_time_series
        self.figures.get('cissa').update({'figure_gap_fill_error':fig_errors,
                            'figure_gap_fill'      :fig_time_series,
                            })
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)

        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------        
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------        
    def pre_fix_censored_data(self,
                             replace_type:        str = 'raw',
                             lower_multiplier:    float = 0.5,
                             upper_multiplier:    float = 1.1, 
                             default_value_lower: float = 0.,
                             default_value_upper: float = 0.,
                             hicensor_lower:      bool = False,
                             hicensor_upper:      bool = False,
                             ):
        '''
        Function to find and replace upper and lower censored data in an input array x.
        There are three different types of replacemet for censored values:
            
        replacement_type = 'raw'      -> Here, censored data is simply replaced by the numeric value. e.g. '<1' becomes 1
        replacement_type = 'multiple' -> Here, censored data is replaced by a multiple multipled the censored data numeric value. For example, '<1' becomes 1* lower_multiplier, while '>1' becomes 1*upper_multiplier.
        replacement_type = 'constant' -> Here, censored data is replaced by a constant value as defined by the input variables default_value_lower and default_value_upper.
        
        Additionally, the function has the option to apply hicensoring to censored data. This is an option where all of the lower censored values are replaced with the largest lower censored value after processing (e.g. after applying a multiplier). Similarly, all of the upper censored values can be replaced with the smallest upper censored value after processing.
        This is useful for data that has multiple levels of censoring to help avoid bias and reduce the potential for trends apprearing only because of a changing censoring level.
        See, for example, Helsel, D. R. (1990). Less than obvious-statistical treatment of data below the detection limit. Environmental science & technology, 24(12), 1766-1774.
        
        
        NOTE 1: Any entries that are not numeric nor censored are converted to np.nan
        NOTE 2: In the future we would like to impute values for censored data based on assumed empirical distribution. See for example, https://cran.r-project.org/web/packages/NADA2/index.html

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION: array of input data.
        replacement_type : str, optional
            DESCRIPTION: Type of replacememt if a censored value is found. Allowed values are 'raw', 'multiple', or 'constant'. The default is 'raw'.
        lower_multiplier : float, optional
            DESCRIPTION. Only used if replacememt_type == 'multiple'. This is the multiplier to apply to a lower censored data point. For example, a point '<1' will become '1*lower_multiplier'. The default is 0.5.
        upper_multiplier : float, optional
            DESCRIPTION. Only used if replacememt_type == 'multiple'. This is the multiplier to apply to a upper censored data point. For example, a point '>1' will become '1*upper_multiplier'. The default is 1.1.
        default_value_lower : float, optional
            DESCRIPTION. Only used if replacement_type == 'constant'. The numeric value to replace any left (lower) censored data. For example, '<1' becomes 'default_value_lower'. The default is 0.-
        default_value_upper : float, optional
            DESCRIPTION. Only used if replacement_type == 'constant'. The numeric value to replace any right (upper) censored data. For example, '<1' becomes 'default_value_upper'. The default is 0.
        hicensor_lower : bool, optional
            DESCRIPTION. Whether lower censored data should be replaced with the largest (replaced) censored value. The default is False.
        hicensor_upper : bool, optional
            DESCRIPTION. Whether upper censored data should be replaced with the smallest (replaced) censored value. The default is False.
            

        Returns
        -------
        x_uncensored : np.ndarray
            DESCRIPTION: array of now uncensored data
        x_censoring : np.ndarray
            DESCRIPTION: array of locations where any censoring was found. Value of the array = None if no censoring is found at a given array position.

        '''
        if self.censored:
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_censored_data, detect_nan_data, detect_censored_data
            self.x,self.censoring = _fix_censored_data(self.x,
                                     replacement_type = replace_type,
                                     lower_multiplier = lower_multiplier,
                                     upper_multiplier = upper_multiplier, 
                                     default_value_lower = default_value_lower,
                                     default_value_upper = default_value_upper,
                                     hicensor_lower = hicensor_lower,
                                     hicensor_upper = hicensor_upper,)
            self.isnan = detect_nan_data(self.x)
            self.censored = detect_censored_data(self.x)
            
        else: warnings.warn("WARNING: No censored data detected. Returning unchanged data.")    
        
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------    
    
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    from datetime import datetime
    def pre_fix_missing_samples(
            self,
            version:              str = 'date', 
            start_date:           str|datetime = 'min',
            date_settings:        dict = {'input_dateformat'  :'',
                                          'years'             :0, 
                                          'months'            :1, 
                                          'days'              :0, 
                                          'hours'             :0,
                                          'minutes'           :0,
                                          'seconds'           :0,
                                          'year_delta'        :0, 
                                          'month_delta'       :0, 
                                          'day_delta'         :14, 
                                          'hour_delta'        :0,
                                          'minute_delta'      :0,
                                          'second_delta'      :0,
                                          },
            numeric_time_settings: dict = {'t_step'    :1.,
                                           'wiggleroom':0.99
                                            },
            missing_value:      int = np.nan
            ):
        '''
        Function that finds and corrects missing values in the time series.
        Missing dates result in adding a default value "missing_value" into the input data.
        
        **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

        Parameters
        ----------
        self : Cissa object
            DESCRIPTION: Cissa object
        version : str, optional
            DESCRIPTION: String describing the type of time data. One of 'date' or 'numeric'. The default is 'date'.
        start_date : str|datetime    
            DESCRIPTION: Only used if version = 'date'. If start_date = 'min' then the minimum date is used, otherwise the given datetime is taken as the first required time. The default is 'min'.
        date_settings : dict, optional
            DESCRIPTION: Dictionary of date settings as defined below:
                                {
                                years : int, optional
                                    DESCRIPTION: (ideal) number of years between each timestep in input array t. The default is 0.
                                months : int, optional
                                    DESCRIPTION: (ideal) number of months between each timestep in input array t. The default is 1.
                                days : int, optional
                                    DESCRIPTION: (ideal) number of days between each timestep in input array t. The default is 0.
                                hours : int, optional
                                    DESCRIPTION: (ideal) number of hours between each timestep in input array t. The default is 0.
                                minutes : int, optional
                                    DESCRIPTION: (ideal) number of minutes between each timestep in input array t. The default is 0.
                                seconds : int, optional
                                    DESCRIPTION: (ideal) number of seconds between each timestep in input array t. The default is 0.
                                input_dateformat : str, optional
                                    DESCRIPTION: Datetime string format. The default is '%Y'. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
                                year_delta : int, optional
                                    DESCRIPTION: Integer years to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                month_delta : int, optional
                                    DESCRIPTION: Integer months to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                day_delta : int, optional
                                    DESCRIPTION: Integer days to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                hour_delta : int, optional
                                    DESCRIPTION: Integer hours to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                minute_delta : int, optional
                                    DESCRIPTION: Integer minutes to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                second_delta : int, optional
                                    DESCRIPTION: Integer seconds to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.    
                                    }
        numeric_time_settings : dict, optional                                
            Dictionary of date settings as defined below:
                               {
                               t_step : int|float, optional
                                   DESCRIPTION: numeric value of the time step. The default is 1.   
                               wiggleroom : int|float, optional
                                   DESCRIPTION: Numeric value for the 'wiggle room' associated with a tolerance tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a time step of 2 and the wiggle room is 0.2, then a series of times 2,4,6,7.9,10,... would be OK, but 2,4,6,7.7,10 would not and would correct the time value to 2,4,6,8,10. The default is 0.99.        
                                   }
        missing_value : int, optional
            DESCRIPTION: The value which is entered when a missing value is found. The default is np.nan.

        Returns
        -------
        final_t : np.ndarray
            DESCRIPTION: array of corrected time values (i.e. missing values are added)
        final_x : np.ndarray
            DESCRIPTION: array of corrected data values (i.e. missing values are added)
        x_missing : np.ndarray
            DESCRIPTION: array of values indicating whether a value is added or not. If not, None, if so, the value will be True.

        '''

        if version == 'date':
            #1) check that the needed settings are present
            if not (date_settings.get('years',0)+date_settings.get('months',0)+date_settings.get('days',0)+date_settings.get('hours',0)+date_settings.get('minutes',0)+date_settings.get('seconds',0)) > 0: raise ValueError(f"At least one date step must be provided and greater than zero. Please check the 'years', 'months', 'days', 'hours', 'minutes', and 'seconds' in date_settings (Note, some of these may be excluded or zero, but at least one should be provided and >0 )") 
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_missing_date_samples 
            self.t,self.x,self.added_times = _fix_missing_date_samples(
                                     self.t,
                                     self.x,
                                     start_date,
                                       years             = date_settings.get('years',0),
                                       months            = date_settings.get('months',0),
                                       days              = date_settings.get('days',0),
                                       hours             = date_settings.get('hours',0),
                                       minutes           = date_settings.get('minutes',0),
                                       seconds           = date_settings.get('seconds',0),
                                       input_dateformat  = date_settings.get('input_dateformat',0),
                                       year_delta        = date_settings.get('year_delta',0),
                                       month_delta       = date_settings.get('month_delta',0),
                                       day_delta         = date_settings.get('day_delta',0),
                                       hour_delta        = date_settings.get('hour_delta',0),
                                       minute_delta      = date_settings.get('minute_delta',0),
                                       second_delta      = date_settings.get('second_delta',0),
                                       missing_value     = missing_value)
            
        elif version == 'numeric':
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_missing_numeric_samples 
            self.t,self.x,self.added_times = _fix_missing_numeric_samples(
                                        self.t, 
                                        self.x,
                                       t_step         = numeric_time_settings.get('t_step',1), 
                                       wiggleroom     = numeric_time_settings.get('wiggleroom',0.99), 
                                       missing_value  = missing_value
                                       )
        else: raise ValueError(f"Input parameter 'version' shpuld be one of 'date' or 'numeric', depending on the time data type. You entered: {version}.")
            
          
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)
        return self
        
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def post_run_frequency_time_analysis(self,
                                    data_per_period:    int,
                                    period_name:        str = '',
                                    t_unit:             str = '', 
                                    plot_frequency:     bool = True,
                                    plot_period:        bool = True,
                                    logplot_frequency:  bool = True,
                                    logplot_period:     bool = False,
                                    normalise_plots:    bool = False,
                                    height_variable:    str = 'power',
                                    height_unit:        str = '',):
        '''
        Function to generate frequency-time and period-time matrices and figures. 

        Parameters
        ----------
        Z : np.ndarray
            DESCRIPTION: Output CiSSA results.
        psd : np.ndarray
            DESCRIPTION: estimation of the the circulant matrix power spectral density
        t : np.ndarray
            DESCRIPTION: Array of input times.
        L : int
            DESCRIPTION: CiSSA window length.
        data_per_period : int
            DESCRIPTION: Number of data points/time steps in a user-defined period (for example, for monthly data and a user-desired period of years, data_per_period = 12)
        period_name : str, optional
            DESCRIPTION: Names of the user-defined period (e.g. years, months, hours etc). The default is ''. 
        t_unit : str, optional
            DESCRIPTION: Time unit (can also be generic such as 'date'). The default is ''.
        plot_frequency : bool, optional
            DESCRIPTION: Flag, whether to produce a frequency-time plot or not. The default is True.
        plot_period : bool, optional
            DESCRIPTION: Flag, whether to produce a period-time plot or not. The default is False.
        logplot_frequency : bool, optional
            DESCRIPTION: Flag, whether to plot the frequency-time plot on a log-scale or not. The default is False.
        logplot_period : bool, optional
            DESCRIPTION: Flag, whether to plot the period-time plot on a log-scale or not. The default is False.
        normalise_plots : bool, optional
            DESCRIPTION: Flag, whether to normalise the plots or not. The default is False.
        height_variable : str, optional
            DESCRIPTION: The height variable to plot. One of 'power', 'amplitude', or 'phase'. The default is 'power'.
        height_unit : str, optional
            DESCRIPTION: Unit of the height variable. The default is ''.

        Returns
        -------
        freq_list : list
            DESCRIPTION: List of frequencies of the signals obtained via CiSSA
        period_list : list
            DESCRIPTION: List of periods of the signals obtained via CiSSA
        amplitude_matrix : np.ndarray
            DESCRIPTION: Array of amplitudes - split via time and for each frequency
        power_matrix : np.ndarray
            DESCRIPTION: Array of power - split via time and for each frequency
        phase_matrix : np.ndarray
            DESCRIPTION: Array of phases - split via time and for each frequency
        frequency_matrix : np.ndarray
            DESCRIPTION: Array of Hilbert Frequencies - split via time and for each frequency
        fig_f : matplotlib.figure
            DESCRIPTION: Frequency-time figure
        fig_p : matplotlib.figure
            DESCRIPTION: Period-time figure

        '''
        from pycissa.postprocessing.frequency_time.frequency_time import _run_frequency_time_analysis
        
        #check that all necessary input variables exist 
        necessary_attributes = ["Z","psd","t","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the run_frequency_time_analysis method.")
        
        #run analysis
        self.frequency_list, self.period_list, self.amplitude_matrix, self.power_matrix, self.phase_matrix, _, fig_f,fig_p =_run_frequency_time_analysis(self.Z,self.psd,self.t,self.L,
                                     data_per_period=data_per_period,period_name=period_name,t_unit=t_unit,plot_frequency=plot_frequency,plot_period=plot_period,logplot_frequency=logplot_frequency,logplot_period=logplot_period,normalise_plots=normalise_plots,height_variable=height_variable,height_unit=height_unit)
                                        
        if fig_f is not None:
            # self.figure_frequency_time = fig_f
            self.figures.get('cissa').update({'figure_frequency_time':fig_f})
        if fig_p is not None:
            # self.figure_period_time = fig_p    
            self.figures.get('cissa').update({'figure_period_time':fig_p})
        
        #add the results to the results dictionary
        results = self.results
        results.get('cissa').update({'frequency_time_results':{
            'frequency_list'   : self.frequency_list, 
            'period_list'      : self.period_list, 
            'amplitude_matrix' : self.amplitude_matrix, 
            'power_matrix'     : self.power_matrix, 
            'phase_matrix'     : self.phase_matrix, }
            })
        
        results.get('cissa').setdefault('model parameters', {})
        results.get('cissa').get('model parameters').update({
            'data_per_period'   : data_per_period, 
            'period_name'       : period_name,
            't_unit'            : t_unit,
            })
        
        self.results = results
        
        #add input parameters to the class
        self.data_per_period = data_per_period
        self.period_name     = period_name
        self.t_unit          = t_unit
        
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def post_analyse_trend(self,
                      trend_type:        str = 'rolling_OLS',
                      t_unit:            str = '',
                      data_unit:         str = '',
                      alphas:            list = [x/20 for x in range(1,20)],
                      timestep:          float = 1,   
                      timestep_unit:     str = '', 
                      include_data:      bool = True, 
                      legend_loc:        int = 2, 
                      shade_area:        bool = False, 
                      xaxis_rotation:    float = 270,
                      window:            int = 12
                      ):
        '''
        Method to calculate and generate the trend slope and confidence for the "trend" component of the CiSSA results.
        Currently can be done using linear regression or rolling ordinary least squares.

        Parameters
        ----------
        trend_type : str, optional
            DESCRIPTION: The type of regression to perform. Current options are "linear" or "rolling_OLS". The default is 'rolling_OLS'.
        t_unit : str, optional
            DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
        data_unit : str, optional
            DESCRIPTION. Data unit. The default is ''.
        alphas : list, optional
            DESCRIPTION. A list of significance levels for the confidence interval. For example, alpha = [.05] returns a 95% confidence interval. The default is [0.05] + [x/20 for x in range(1,20)].
        timestep : float, optional
            DESCRIPTION. Numeric timestep size in t_unit units. The default is 60*60*24.                 
        timestep_unit : str, optional
            DESCRIPTION. Timestep unit (e.g. seconds, days, years). The default is 'day'.      
        include_data : bool, optional
            DESCRIPTION. Whether to include the original time-series in the plot or not. The default is True.
        legend_loc : int, optional
            DESCRIPTION: Location of the legend. The default is 2.
        shade_area : bool, optional
            DESCRIPTION: Whether to shade below the trend or not. The default is False.
        xaxis_rotation : float, optional
            DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.
        window : int, optional
            DESCRIPTION. Only used if trend_type = "rolling_OLS". Length of the rolling window. Must be strictly larger than the number of variables in the model. The default is 12.


        '''
        #check that all necessary input variables exist 
        necessary_attributes = ["t","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the run_frequency_time_analysis method.")
        
        
        if trend_type == 'linear':
            from pycissa.postprocessing.trend.trend_functions import trend_linear
            
            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_linear(
                             self.results.get('cissa').get('components').get('trend').get('reconstructed_data'),
                             self.t,
                             t_unit=t_unit,
                             Y_unit=data_unit,
                             alphas=alphas,
                             timestep=timestep,
                             timestep_unit=timestep_unit,
                             include_data=include_data,
                             legend_loc=legend_loc,
                             shade_area=shade_area,
                             xaxis_rotation=xaxis_rotation
                             )
            self.trend_type = 'Linear'
            self.figures.get('cissa').update({'figure_trend':figure_trend})
            #
        elif trend_type == 'rolling_OLS':
            from pycissa.postprocessing.trend.trend_functions import trend_rolling
            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_rolling(
                              self.results.get('cissa').get('components').get('trend').get('reconstructed_data'),
                              self.t,
                              t_unit=t_unit,
                              Y_unit=data_unit,
                              window=window,
                              alphas=alphas,
                              timestep=timestep,
                              timestep_unit=timestep_unit,
                              include_data=include_data,
                              legend_loc=legend_loc,
                              shade_area=shade_area,
                              xaxis_rotation=xaxis_rotation
                              )
            self.trend_type = 'rolling_OLS'
            self.figures.get('cissa').update({'figure_trend':figure_trend})
            
        else:
            raise ValueError(f"Input value trend_type = {trend_type} is incorrect. Please use one of 'linear' or 'rolling_OLS'.")
       
        
       #update results dictionary
        results = self.results
        results.get('cissa').setdefault('trend results', {})
        results.get('cissa').get('trend results').setdefault(self.trend_type, {})
        results.get('cissa').get('trend results').get(self.trend_type).update({
            'trend_slope'                       : self.trend_slope, 
            'trend_increasing_probability'      : self.trend_increasing_probability,
            'trend_increasing_probability_text' : self.trend_increasing_probability_text,
            'trend_confidence'                  : self.trend_confidence
            })
        self.results = results
            
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def post_run_monte_carlo_analysis(self,
                                 alpha:                    float = 0.05, 
                                 K_surrogates:             int = 1,
                                 surrogates:               str = 'random_permutation',
                                 seed:                     int|None = None,
                                 sided_test:               str = 'one sided', 
                                 remove_trend:             bool = True,
                                 trend_always_significant: bool = True,
                                 A_small_shuffle:          float = 1.,
                                 extension_type:           str = 'AR_LR',
                                 multi_thread_run:         bool = True,
                                 generate_toeplitz_matrix: bool = False):
        '''
        Function to run a monte carlo significance test on components of a signal, extracted via CiSSA.
        Signal psd/eigenvalues are compared to those obtained by applying CiSSA to surrogate data.
        Surrogates are generated using one of three available algorithms:
            random_permutation: randomly shuffle the input data
            small_shuffle: the small shuffle method of Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
            ar1_fit: Fits an autoregressive model of order 1 to the data.

        Parameters
        ----------
        alpha : float, optional
            DESCRIPTION: Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
        K_surrogates : int, optional
            DESCRIPTION: Multiplier for number of surrogates. Number of surrogate data allowed to be larger than the signal and signal to still be significant = K_surrogates - 1. 
                            For a one-sided test, the number of surrogate data series generated is K_surrogates/alpha - 1. For a two sided test it is 2*K_surrogates/alpha - 1.
                            The default is 1.
        surrogates : str, optional
            DESCRIPTION: The type of surrogates to generate for the hypothesis test.
                            One of "random_permutation", "small_shuffle", "ar1_fit".
                            The default is 'random_permutation'.
        seed : int|None, optional
            DESCRIPTION: Random seed for reproducability. The default is None.
        sided_test : str, optional
            DESCRIPTION: When assessing the null hypothesis, are we running a one or two-sided test? The default is 'one sided'.
        remove_trend : bool, optional
            DESCRIPTION: Some surrogate methods make assumptions that are violated when there is a trend in the input data. 
                            If remove_trend = True then the trend is removed before surrogates are generated, then added back to the surrogate data after generation. See  Lucio, J. H., Valdés, R., & Rodríguez, L. R. (2012). Improvements to surrogate data methods for nonstationary time series. Physical Review E, 85(5), 056202.
                            The default is True.
        trend_always_significant : bool, optional
            DESCRIPTION: Option to ensure the trend is always significant. (Possibly) necessary if remove_trend = True.The default is True.
        A_small_shuffle : float, optional
            DESCRIPTION: If surrogates = 'small_shuffle', then this parameter is the "A" parameter in the small shuffle paper, Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
                            The default is 1.
        extension_type : str, optional
            DESCRIPTION: extension type for left and right ends of the time series. The default is AR_LR.
        multi_thread_run : bool, optional
            DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.
        generate_toeplitz_matrix : bool, optional
            DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False.

        '''
        from pycissa.postprocessing.monte_carlo.montecarlo import run_monte_carlo_test
        #check that all necessary input variables exist 
        necessary_attributes = ["psd","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_run_monte_carlo_analysis method.")
        
        mc_results, figure_monte_carlo = run_monte_carlo_test(x = self.x,
                             L = self.L,
                             psd=self.psd,
                             results=self.results.get('cissa'),
                             alpha=alpha,
                             K_surrogates=K_surrogates,
                             surrogates=surrogates,
                             seed=seed,
                             sided_test=sided_test,
                             remove_trend=remove_trend,
                             trend_always_significant=trend_always_significant,
                             A_small_shuffle=A_small_shuffle,
                             extension_type=extension_type,
                             multi_thread_run=multi_thread_run,
                             generate_toeplitz_matrix=generate_toeplitz_matrix
                                 )
        self.results.get('cissa').update(mc_results)
        self.results['cissa']['model parameters'].update({'monte_carlo_surrogate_type':surrogates}) 
        self.results['cissa']['model parameters'].update({'monte_carlo_alpha':alpha}) 
        self.figures.get('cissa').update({'figure_monte_carlo':figure_monte_carlo})
        
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    # smallest_n number_of_groups_to_drop include_trend
    # smallest_proportion eigenvalue_proportion
    # monte_carlo
    # 'drop_smallest_n', 'drop_smallest_proportion', 'monte_carlo_significant_components'
    def post_group_components(self,
                                 grouping_type:            str = 'monte_carlo', 
                                 eigenvalue_proportion:    float = 0.9,
                                 number_of_groups_to_drop: int = 5,
                                 include_trend:            bool = True,
                                 check_noise_statistics:   bool = True,
                                 noise_alpha:              float = 0.05, 
                                 ljung_box_lags:           int = 12,  
                                 plot_result:              bool = True,):
        '''
        Function to group components into trend, periodic, or noise/residual.

        Parameters
        ----------
        grouping_type : str, optional
            DESCRIPTION. The default is 'monte_carlo'. Current options are 'monte_carlo', 'smallest_proportion', or 'smallest_n'.
        eigenvalue_proportion : float, optional
            DESCRIPTION. The default is 0.9. Only used of grouping type = 'smallest_proportion'
                    There are two options:
                    1) A number between 0 & 1. This number represents the accumulated
                    share of the psd achieved with the sum of the share associated to
                    the largest eigenvalues. The function computes the trend and periodic components
                    as these components, and the remaining as noise.
                    2) A number between -1 & 0. It is a percentile (in positive) of
                    the psd. The function classifies as trend/periodic the componentes by frequency
                    whose psd is greater that this percentile, and noise otherwise.
        number_of_groups_to_drop : int, optional
            DESCRIPTION. The default is 5. Only used if grouping_type == 'smallest_n'. Will order the components by psd proportion and classify the lowest "number_of_groups_to_drop" as noise.
        include_trend : bool, optional
            DESCRIPTION. The default is True. Only used if grouping_type == 'smallest_n'. If False, the trend will always be removed.
        check_noise_statistics : bool, optional
            DESCRIPTION. The default is True. If True, will check the noise component for normality and autocorrelation.
        noise_alpha : float, optional
            DESCRIPTION. The default is 0.05. Significance level for statistical tests.
        ljung_box_lags : int, optional    
            DESCRIPTION. The default is 12. number of lags to check in the Ljung-box test.
        plot_result : bool, optional
            DESCRIPTION. The default is True. Plot resulting breakdown of components or not.

        '''
        def combine_components(temp_results,group_indices):
            x_grouped = np.zeros(temp_results['components']['trend']['reconstructed_data'].shape)
            for key_j in temp_results['components'].keys():
                if temp_results['components'][key_j]['array_position'] in group_indices:
                    x_grouped += temp_results['components'][key_j]['reconstructed_data']
            return x_grouped        
            
        from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_n_components
        #check that all necessary input variables exist 
        necessary_attributes = ["psd","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_group_components method.")
        # 'monte_carlo', 'smallest_proportion', or 'smallest_n'.
        if grouping_type == 'monte_carlo':
            if self.results.get('cissa').get('components').get('trend').get('monte_carlo') is None: raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the post_group_components with grouping_type == 'monte_carlo' or use another grouping type.")
            from pycissa.postprocessing.grouping.grouping_functions import classify_monte_carlo_non_significant_components
            trend,  periodic, noise = classify_monte_carlo_non_significant_components(self.Z,
                                                                                      self.results.get('cissa'))
        elif grouping_type == 'smallest_proportion':
            from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_proportion_psd
            trend,  periodic, noise = classify_smallest_proportion_psd(self.Z,
                                                                       self.psd,
                                                                       self.L,
                                                                       eigenvalue_proportion)
        elif grouping_type == 'smallest_n':
            from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_n_components
            trend,  periodic, noise = classify_smallest_n_components(self.Z, 
                                                                     self.psd, 
                                                                     self.L,
                                                                     number_of_groups_to_drop,
                                                                     include_trend=include_trend)
        else: raise ValueError(f"Input parameter 'grouping_type' should be one of 'monte_carlo', 'smallest_proportion', or 'smallest_n'. You entered: {grouping_type}.")
        
        self.results['cissa']['noise component tests'].update({'trend_index':trend}) 
        self.results['cissa']['noise component tests'].update({'periodic_index':periodic}) 
        self.results['cissa']['noise component tests'].update({'noise_index':noise}) 
        
        self.x_trend = combine_components(self.results['cissa'],trend)
        self.x_periodic = combine_components(self.results['cissa'],periodic)
        self.x_noise = combine_components(self.results['cissa'],noise)


        if check_noise_statistics:
            from pycissa.postprocessing.statistics.von_neumann import rank_von_neumann_test
            from pycissa.postprocessing.statistics.ljung_box import run_ljung_box_test
            from pycissa.postprocessing.statistics.normality_tests import run_normality_test
            _,_,_,interpretation = rank_von_neumann_test(self.x_noise,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'rank von Neumann' : interpretation})
            _,_,_,_,interpretation = run_ljung_box_test(self.x_noise,lags = ljung_box_lags,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'ljung_box' : interpretation})
            _,_,_,interpretation = run_normality_test(self.x_noise,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'normality' : interpretation})
            
        
        if plot_result:
            from pycissa.utilities.plotting import plot_grouped_components
            fig = plot_grouped_components(self.t,
                                          self.x,
                                          self.x_trend,
                                          self.x_periodic,
                                          self.x_noise,)
            self.figures.get('cissa').update({'figure_split_components':fig})
            
        

        
        return self
    
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def plot_autocorrelation(self,
                                noise_components: list|None = None, 
                                monte_carlo_noise:bool = False,
                                acf_lags:         int|list|None=None, 
                                pacf_lags:        int|list|None=None, 
                                alpha:            float=0.05, 
                                use_vlines:       bool=True, 
                                adjusted:         bool=False, 
                                fft:              bool=False,
                                missing:          str='none',
                                zero:             bool=True, 
                                auto_ylims:       bool=False, 
                                bartlett_confint: bool=True,
                                pacf_method:      str='ywm',
                                acf_color:        str='blue', 
                                pacf_color:       str='blue', 
                                title_size:       float|int=14, 
                                label_size:       float|int=12
                                ):
        '''
        Function to generate and return a figure with three subplots: 
            1) time series data, 
            2) the autocorrelation function of the data, 
            3) the partial autocorrelation function of the data.
        Option to create a second plot with noise components after running cissa.

        Parameters
        ----------
        self.t : np.ndarray
            DESCRIPTION: array of input times/dates.
        self.x : np.ndarray
            DESCRIPTION: array of input data.
        noise_components : list|None, optional
            DESCRIPTION. The default is None. A list of noise components. If provided a second figure will be created with the autocorrelation of the sum of all noise components in the noise_components list.
        monte_carlo_noise : bool, optional
            DESCRIPTION. The default is False. If True, the noise_components are ignored and the components that fail the monte carlo analysis are combined and used as noise.
        acf_lags : int|list|None, optional
            DESCRIPTION. The default is None. An int or array of acf lag values, used on horizontal axis. If None, half the time series length is used as lags.
        pacf_lags : int|list|None, optional
            DESCRIPTION. The default is None. An int or array of acf lag values, used on horizontal axis. If None, acf_lags is used as lags.
        alpha : float, optional
            DESCRIPTION. The default is 0.05. If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to Bartlett’s formula. If None, no confidence intervals are plotted.
        use_vlines : bool, optional
            DESCRIPTION. The default is True. If True, vertical lines and markers are plotted. If False, only markers are plotted.
        adjusted : bool, optional
            DESCRIPTION. The default is False. For acf, if True, then denominators for autocovariance are n-k, otherwise n
        fft : bool, optional
            DESCRIPTION. The default is False. For acf, if True, computes the ACF via FFT.
        missing : str, optional
            DESCRIPTION. The default is 'none'. A string in [‘none’, ‘raise’, ‘conservative’, ‘drop’] specifying how the NaNs are to be treated.
        zero : bool, optional
            DESCRIPTION. The default is True. Flag indicating whether to include the 0-lag autocorrelation. 
        auto_ylims : bool, optional
            DESCRIPTION. The default is False. If True, adjusts automatically the y-axis limits to ACF values.
        bartlett_confint : bool, optional
            DESCRIPTION. The default is True. Confidence intervals for ACF values are generally placed at 2 standard errors around r_k. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2 in [2]. For the ACF of raw data, the standard error at a lag k is found as if the right model was an MA(k-1). This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an MA of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett’s formula. For more details on Bartlett formula result, see section 7.2 in [1]. [1] Brockwell and Davis, 1987. Time Series Theory and Methods [2] Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.
        pacf_method : str, optional
            DESCRIPTION. The default is 'ywm'. Specifies which method for the calculations to use:
                                                “ywm” or “ywmle” : Yule-Walker without adjustment. Default.
                                                “yw” or “ywadjusted” : Yule-Walker with sample-size adjustment in denominator for acovf. Default.
                                                “ols” : regression of time series on lags of it and on constant.
                                                “ols-inefficient” : regression of time series on lags using a single common sample to estimate all pacf coefficients.
                                                “ols-adjusted” : regression of time series on lags with a bias adjustment.
                                                “ld” or “ldadjusted” : Levinson-Durbin recursion with bias correction.
                                                “ldb” or “ldbiased” : Levinson-Durbin recursion without bias correction.
        acf_color : str, optional
            DESCRIPTION. The default is 'blue'. Colour of the acf plot markers.
        pacf_color : str, optional
            DESCRIPTION. The default is 'blue'. Colour of the pacf plot markers.
        title_size : float|int, optional
            DESCRIPTION. The default is 14. 
        label_size : float|int, optional
            DESCRIPTION. The default is 12.

        Returns
        -------
        fig : matplotlib figure
            DESCRIPTION. Figure of time series, acf, pacf.

        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------
        from pycissa.postprocessing.statistics.autocorrelation_function import plot_time_series_and_acf_pacf
        
        fig = plot_time_series_and_acf_pacf(self.t,self.x,
                                          acf_lags=acf_lags,
                                          pacf_lags=pacf_lags,
                                          alpha=alpha,
                                          use_vlines=use_vlines,
                                          adjusted=adjusted,
                                          fft=fft,
                                          missing=missing,
                                          zero=zero,
                                          auto_ylims=auto_ylims,
                                          bartlett_confint=bartlett_confint,
                                          pacf_method=pacf_method,
                                          acf_color=acf_color,
                                          pacf_color=pacf_color,
                                          title = 'Original Time Series',
                                          title_size=title_size,
                                          label_size=label_size
                                          )
        self.figures.get('cissa').update({'figure_autocorrelation':fig})
        
        
        ######################################################################
        if monte_carlo_noise:
            #check that fit has been run
            necessary_attributes = ["results"]
            for attr_i in necessary_attributes:
                if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the plot_autocorrelation method with noise components.")
            
            #check that monte carlo has been run
            monte_carlo_complete = self.results['cissa']['components']['trend'].get('monte_carlo',False)
            if not monte_carlo_complete:raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the plot_autocorrelation method with monte_carlo_noise = True.")

            
            monte_carlo_type = self.results['cissa']['model parameters']['monte_carlo_surrogate_type']
            monte_carlo_alpha = self.results['cissa']['model parameters']['monte_carlo_alpha']
            noise_components = []
            for key_j in self.results['cissa']['components'].keys():
                if not self.results['cissa']['components'][key_j]['monte_carlo'][monte_carlo_type]['alpha'][monte_carlo_alpha]['pass']:
                    noise_components.append(self.results['cissa']['components'][key_j]['array_position'])
        ######################################################################    

            
        
        ######################################################################
        if noise_components is not None:
            #check that fit has been run
            necessary_attributes = ["results"]
            for attr_i in necessary_attributes:
                if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the plot_autocorrelation method with noise components.")
            
            
            noise_array = np.zeros(self.results['cissa']['components']['trend']['reconstructed_data'].shape)
            for key_j in self.results['cissa']['components'].keys():
                if self.results['cissa']['components'][key_j]['array_position'] in noise_components:
                    noise_array += self.results['cissa']['components'][key_j]['reconstructed_data']

            
            fig2 = plot_time_series_and_acf_pacf(self.t,noise_array,
                                              acf_lags=acf_lags,
                                              pacf_lags=pacf_lags,
                                              alpha=alpha,
                                              use_vlines=use_vlines,
                                              adjusted=adjusted,
                                              fft=fft,
                                              missing=missing,
                                              zero=zero,
                                              auto_ylims=auto_ylims,
                                              bartlett_confint=bartlett_confint,
                                              pacf_method=pacf_method,
                                              acf_color=acf_color,
                                              pacf_color=pacf_color,
                                              title = 'Residual Time Series',
                                              title_size=title_size,
                                              label_size=label_size
                                              )
            self.figures.get('cissa').update({'figure_autocorrelation_noise':fig2}) 
        ######################################################################
        return self
                


    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def plot_seasonal_boxplots(self,
                               split_date:    datetime|None = None,
                               bar_width:     float = 0.25,
                               plot_type:     str = 'both',
                               remove_trend:  bool=False,
                               include_noise: bool=True,):
        '''
        Function to plot seasonal (either monthly or yearly or both) boxplots.
        Includes the option to split the data by a set date and will plot grouped boxplots with one group being the boxplot of the seasonal data before the set date, the other after.

        Parameters
        ----------
        split_date : datetime|None, optional
            DESCRIPTION. The default is None. A datetime object which splits the boxplots into groups, one before the set date, one after. If None, the data is not split.
        bar_width : float, optional
            DESCRIPTION. The default is 0.25. Width of each individual bar
        plot_type : str, optional
            DESCRIPTION. The default is 'both'. One of 'both', 'monthly', or 'yearly', depending on the type of boxplot to be plotted.
        remove_trend : bool, optional
            DESCRIPTION. The default is False. If True, the trend is remove. If True, then we first must have run post_group_components.
        include_noise : bool, optional
            DESCRIPTION. The default is True. Only used if remove_trend = True. If False, the noise component is removed, leaving only the periodic component to be box-plotted.


        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before plot_seasonal_boxplots.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before plot_seasonal_boxplots.")
        #----------------------------------------------------------------------
        
        if plot_type in ['both','monthly']:
            from pycissa.utilities.plotting import seasonal_boxplots
            if remove_trend:
                necessary_attributes = ["x_trend","x_periodic","x_noise"]
                for attr_i in necessary_attributes:
                    if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa post_group_components method before running the plot_seasonal_boxplots method with remove_trend = True.")
                if include_noise:x_plot = self.x_periodic + self.x_noise
                else:x_plot = self.x_periodic
            else:    
                x_plot = self.x
            fig = seasonal_boxplots(self.t,x_plot,split_date=split_date,bar_width=bar_width)
            self.figures.get('cissa').update({'figure_monthly_seasonal_box':fig}) 
            
        if plot_type in ['both','yearly']:
            from pycissa.utilities.plotting import yearly_boxplots
            if remove_trend:
                necessary_attributes = ["x_trend","x_periodic","x_noise"]
                for attr_i in necessary_attributes:
                    if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa post_group_components method before running the plot_seasonal_boxplots method with remove_trend = True.")
                if include_noise:x_plot = self.x_periodic + self.x_noise
                else:x_plot = self.x_periodic
            else:    
                x_plot = self.x
            fig = yearly_boxplots(self.t,x_plot,bar_width=bar_width)
            self.figures.get('cissa').update({'figure_yearly_seasonal_box':fig}) 
            
        return self        
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def auto_cissa(self,
                   L: int = None,
                   **kwargs):
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))
        
        #fix censoring and nan
        _ = self.auto_fix_censoring_nan(L,**kwargs)
        
        #run cissa
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))
        
        #run monte carlo
        _ = self.post_run_monte_carlo_analysis(
                                     alpha                    = kwargs.get('alpha',0.05), 
                                     K_surrogates             = kwargs.get('K_surrogates',1),  
                                     surrogates               = kwargs.get('surrogates','random_permutation'), 
                                     seed                     = kwargs.get('seed',None),   
                                     sided_test               = kwargs.get('sided_test','one sided'),   
                                     remove_trend             = kwargs.get('remove_trend',True),  
                                     trend_always_significant = kwargs.get('trend_always_significant',True),
                                     A_small_shuffle          = kwargs.get('A_small_shuffle',1.), 
                                     extension_type           = kwargs.get('extension_type','AR_LR'), 
                                     multi_thread_run         = kwargs.get('multi_thread_run',True), 
                                     generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False), 
                                     )
        
        #run grouping
        _ = self.post_group_components(
                                      grouping_type            = kwargs.get('grouping_type','monte_carlo'),
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      check_noise_statistics   = kwargs.get('check_noise_statistics',True),
                                      noise_alpha              = kwargs.get('noise_alpha',0.05),
                                      ljung_box_lags           = kwargs.get('ljung_box_lags',12),
                                      plot_result              = kwargs.get('plot_result',True))  
        
        #plot frequency time graphs
        _ = self.post_run_frequency_time_analysis(
                                        data_per_period   = kwargs.get('data_per_period',1),
                                        period_name       = kwargs.get('period_name',''),
                                        t_unit            = kwargs.get('t_unit',''),
                                        plot_frequency    = kwargs.get('plot_frequency',True),
                                        plot_period       = kwargs.get('plot_period',True),
                                        logplot_frequency = kwargs.get('logplot_frequency',True),
                                        logplot_period    = kwargs.get('logplot_period',False),
                                        normalise_plots   = kwargs.get('normalise_plots',False),
                                        height_variable   = kwargs.get('height_variable','power'),
                                        height_unit       = kwargs.get('height_unit',''))
        
        #plot trend
        _ = self.post_analyse_trend(
                          trend_type     = kwargs.get('trend_type','rolling_OLS'),
                          t_unit         = kwargs.get('t_unit',''),
                          data_unit      = kwargs.get('data_unit',''),
                          alphas         = kwargs.get('alphas',[x/20 for x in range(1,20)]),
                          timestep       = kwargs.get('timestep',1),
                          timestep_unit  = kwargs.get('timestep_unit',''),
                          include_data   = kwargs.get('include_data',True),
                          legend_loc     = kwargs.get('legend_loc',2),
                          shade_area     = kwargs.get('shade_area',False),
                          xaxis_rotation = kwargs.get('xaxis_rotation',270),
                          window         = kwargs.get('window',12)
                          )
        
        #plot autocorrelation
        if not kwargs.get('noise_components'):
            kwargs.update({'monte_carlo_noise':True})
        _ = self.plot_autocorrelation(
                                    noise_components  = kwargs.get('noise_components',None),
                                    monte_carlo_noise = kwargs.get('monte_carlo_noise',False),
                                    acf_lags          = kwargs.get('acf_lags',None),
                                    pacf_lags         = kwargs.get('pacf_lags',None),
                                    alpha             = kwargs.get('alpha',0.05),
                                    use_vlines        = kwargs.get('use_vlines',True),
                                    adjusted          = kwargs.get('adjusted',False),
                                    fft               = kwargs.get('fft',False),
                                    missing           = kwargs.get('missing','none'),
                                    zero              = kwargs.get('zero',True),
                                    auto_ylims        = kwargs.get('auto_ylims',False),
                                    bartlett_confint  = kwargs.get('bartlett_confint',True),
                                    pacf_method       = kwargs.get('pacf_method','ywm'),
                                    acf_color         = kwargs.get('acf_color','blue'),
                                    pacf_color        = kwargs.get('pacf_color','blue'),
                                    title_size        = kwargs.get('title_size',14),
                                    label_size        = kwargs.get('label_size',12)
                                    )
        print("Auto Cissa Complete!")
        return self
        
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def auto_denoise(self,
                     L:             int = None,
                     plot_denoised: bool = True,
                     **kwargs):
        '''
        Function to automatically denoise a time series using Cissa.
        Automatically:
            corrects censored and nan values.
            fits the time series using Cissa.
            groups the components into signal and noise.
            Plots the results.

        Parameters
        ----------
        L : int, optional
            DESCRIPTION. The default is None. The default is None. CiSSA window length.
        plot_denoised : bool, optional
            DESCRIPTION. The default is True.  If True, the resulting denoised figure is plotted.
        **kwargs : TYPE
            DESCRIPTION. key word arguments for the auto_fix_censoring_nan(), fit(), post_run_monte_carlo_analysis(), and  post_group_components() functions.

        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))
        
        #fix censoring and nan
        _ = self.auto_fix_censoring_nan(L,**kwargs)
        
        #run cissa
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))
        
        #run monte carlo if needed
        if kwargs.get('grouping_type','monte_carlo')=='monte_carlo':
            _ = self.post_run_monte_carlo_analysis(
                                         alpha                    = kwargs.get('alpha',0.05), 
                                         K_surrogates             = kwargs.get('K_surrogates',1),  
                                         surrogates               = kwargs.get('surrogates','random_permutation'), 
                                         seed                     = kwargs.get('seed',None),   
                                         sided_test               = kwargs.get('sided_test','one sided'),   
                                         remove_trend             = kwargs.get('remove_trend',True),  
                                         trend_always_significant = kwargs.get('trend_always_significant',True),
                                         A_small_shuffle          = kwargs.get('A_small_shuffle',1.), 
                                         extension_type           = kwargs.get('extension_type','AR_LR'), 
                                         multi_thread_run         = kwargs.get('multi_thread_run',True), 
                                         generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False), 
                                         )
            
         #   run grouping
        _ = self.post_group_components(
                                      grouping_type            = kwargs.get('grouping_type','monte_carlo'),
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      check_noise_statistics   = kwargs.get('check_noise_statistics',True),
                                      noise_alpha              = kwargs.get('noise_alpha',0.05),
                                      ljung_box_lags           = kwargs.get('ljung_box_lags',12),
                                      plot_result              = kwargs.get('plot_result',True))  

        self.x_denoised = self.x_trend + self.x_periodic
        
        if plot_denoised:
            from pycissa.utilities.plotting import plot_denoised_signal
            fig = plot_denoised_signal(self.t,
                                 self.x,
                                 self.x_denoised
                                        )
            self.figures.get('cissa').update({'figure_denoised':fig})
        
        return self
         
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def auto_detrend(self,
                     L:           int = None,
                     plot_result: bool = True,
                     **kwargs):
        '''
        Function to automatically detrend a signal using Cissa. 
        Automatically:
            corrects censored and nan values.
            fits the time series using Cissa.
            groups the components into trend and detrended signal.
            Plots the results.

        Parameters
        ----------
        L : int, optional
            DESCRIPTION. The default is None. CiSSA window length.
        plot_result : bool, optional
            DESCRIPTION. The default is True. If True, the resulting detrended figure is plotted.
        **kwargs : dict
            DESCRIPTION. key word arguments for the auto_fix_censoring_nan() and fit() functions.
        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))
        
        #fix censoring and nan
        _ = self.auto_fix_censoring_nan(L,**kwargs)
        
        #run cissa
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))
        
        #group components
        from pycissa.postprocessing.grouping.grouping_functions import group
        number_of_signal_components = int(len(self.psd)/2 - 1)
        I = {'trend'    :[0],
             'detrended':[x for x in range(1,number_of_signal_components+1)]}
        rc, sh, kg, psd_sh = group(self.Z,self.psd,I)
        
        self.x_trend = rc['trend'].reshape(len(rc['trend']),)
        self.x_detrended = rc['detrended'].reshape(len(rc['detrended']),)
        
        
        if plot_result:
            from pycissa.utilities.plotting import plot_detrended_signal
            fig=plot_detrended_signal(self.t,
                                    self.x,
                                    self.x_trend,
                                    self.x_detrended,
                                        )

            self.figures.get('cissa').update({'figure_detrended':fig})
            
            
        return self
        
        
        
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def auto_fix_censoring_nan(self,L : int,**kwargs):
        '''
        Function to automatically fix any censoring or nan values in the data.

        Parameters
        ----------
        L : int
            DESCRIPTION: CiSSA window length.
        **kwargs : dict
            DESCRIPTION. key word arguments for the pre_fix_censored_data() and pre_fill_gaps() functions.


        '''
        #check for censored, nan data
        if self.censored: 
            warnings.warn("Censored data detected. Running pre_fix_censored_data to fix...")
            _ = self.pre_fix_censored_data(
                                     replace_type        = kwargs.get('replace_type','raw'),
                                     lower_multiplier    = kwargs.get('lower_multiplier',0.5),
                                     upper_multiplier    = kwargs.get('upper_multiplier',1.1),
                                     default_value_lower = kwargs.get('default_value_lower',0.),
                                     default_value_upper = kwargs.get('default_value_upper',0.),
                                     hicensor_lower      = kwargs.get('hicensor_lower',False), 
                                     hicensor_upper      = kwargs.get('hicensor_upper',False),
                                     )
            
        if self.isnan: 
            warnings.warn("Censored data detected. Running pre_fill_gaps to fix...")
            from pycissa.utilities.helper_functions import get_keyword_args
            keys_to_remove = get_keyword_args(self.pre_fill_gaps)
            temp_kwargs = {key: value for key, value in kwargs.items() if key not in keys_to_remove}
            _ = self.pre_fill_gaps(                    
                          L,
                          convergence                = kwargs.get('convergence',['value', 1]),
                          extension_type             = kwargs.get('extension_type','AR_LR'),
                          multi_thread_run           = kwargs.get('multi_thread_run',True),
                          initial_guess              = kwargs.get('initial_guess',['previous', 1]),
                          outliers                   = kwargs.get('outliers',['nan_only',None]),
                          estimate_error             = kwargs.get('estimate_error',True),
                          test_number                = kwargs.get('test_number',10),
                          test_repeats               = kwargs.get('test_repeats',1),
                          z_value                    = kwargs.get('z_value',1.96),
                          component_selection_method = kwargs.get('component_selection_method','monte_carlo_significant_components'),
                          eigenvalue_proportion      = kwargs.get('eigenvalue_proportion',0.95),
                          number_of_groups_to_drop   = kwargs.get('number_of_groups_to_drop',1),
                          data_per_unit_period       = kwargs.get('data_per_unit_period',1),
                          use_cissa_overlap          = kwargs.get('use_cissa_overlap',False),
                          drop_points_from           = kwargs.get('drop_points_from','Left'),
                          max_iter                   = kwargs.get('max_iter',50),
                          verbose                    = kwargs.get('verbose',False),
                          alpha                      = kwargs.get('alpha', 0.05),
                          **temp_kwargs,
                          )
            
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 



     
    #List of stuff to add in here
    '''  
    check figure sizes
    lomb-scargle, 
    predict method (TO DO, maybe using AutoTS or MAPIE?)
    '''    
          
