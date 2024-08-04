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
        if self.censored: warnings.warn("WARNING: Censored data detected. Please run fix_censored_data before fitting.")
        #----------------------------------------------------------------------    
        
        
        #----------------------------------------------------------------------
        self.t = t #array of times
        self.x = x #array of corresponding data
        #----------------------------------------------------------------------
    def restore_original_data(self):
        '''
        Method to restore original data (x,t) = (x_raw,t_raw)
        '''
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data
        self.x = self.x_raw
        self.t = self.t_raw
        self.censored = detect_censored_data(self.x)  #if we restore the data we must check if the restored data is censored again...
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
        #ensure data is uncensored
        if self.censored:  raise ValueError("Censored data detected. Please run fix_censored_data before fitting.")
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
        results.setdefault('model parameters', {})
        results.get('model parameters').update({
            'extension_type'   : extension_type, 
            'L'                : L,
            'multi_thread_run' : multi_thread_run,
            })
        self.results = results
        self.figures = {}  #make a space for future figures
        
        
        #save settings
        self.extension_type = extension_type
        self.L = L
        self.multi_thread_run = multi_thread_run
        
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    def post_predict(self):
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
                  z_value:                    float = 1.96,
                  component_selection_method: str = 'drop_smallest_proportion',
                  eigenvalue_proportion:      float = 0.95,
                  number_of_groups_to_drop:   int = 1,
                  data_per_unit_period:       int = 1,
                  use_cissa_overlap:          bool = False,
                  drop_points_from:           str = 'Left',
                  max_iter:                   int = 10,
                  verbose:                    bool = False
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
        z_value : float, optional
            DESCRIPTION: z-value for confidence interval (= 1.96 for a 95% confidence interval, for example)           
        component_selection_method : str, optional
            DESCRIPTION. Method for choosing the way we drop components from the reconstruction. The default is 'drop_smallest_proportion'.
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
            DESCRIPTION. Maximum number of iterations to check for convergence. The default is 10.
        verbose : bool, optional
            DESCRIPTION. Whether to print some info to the console or not. The default is False.

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
                                 verbose=verbose)
        
        self.x = x_ca
        self.gap_fill_error_estimates            = error_estimates
        self.gap_fill_error_estimates_percentage = error_estimates_percentage
        self.gap_fill_error_rmse                 = error_rmse
        self.gap_fill_error_rmse_percentage      = error_rmse_percentage
        self.gap_fill_original_points            = original_points 
        self.gap_fill_imputed_points             = imputed_points, 
        # self.figure_gap_fill_error               = fig_errors,
        # self.figure_gap_fill                     = fig_time_series
        self.figures.update({'figure_gap_fill_error':fig_errors,
                            'figure_gap_fill'      :fig_time_series,
                            })

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
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_censored_data
            self.x,self.censoring = _fix_censored_data(self.x,
                                     replacement_type = replace_type,
                                     lower_multiplier = lower_multiplier,
                                     upper_multiplier = upper_multiplier, 
                                     default_value_lower = default_value_lower,
                                     default_value_upper = default_value_upper,
                                     hicensor_lower = hicensor_lower,
                                     hicensor_upper = hicensor_upper,)
            self.censored = False
        else: warnings.warn("WARNING: No censored data detected. Returning unchanged data.")    
        
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------    
    def pre_fix_missing_samples(
            self,
            input_dateformat:     str,
              years:              int = 0, 
              months:             int = 0, 
              days:               int = 0, 
              hours:              int = 0,
              minutes:            int = 0,
              seconds:            int = 0,
              wiggleroom_divisor: int = 2,
              missing_value:      int = np.nan
            ):
        '''
        Function that finds and corrects missing values in the time series.
        Missing dates result in adding a default value "missing_value" into the input data.
        
        **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

        Parameters
        ----------
        t : np.ndarray
            DESCRIPTION: array of input times/dates.
        x : np.ndarray
            DESCRIPTION: array of input data.
        input_dateformat : str
            DESCRIPTION: Datetime string format. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes    
        years : int, optional
            DESCRIPTION: (ideal) number of years between each timestep in input array t. The default is 1.
        months : int, optional
            DESCRIPTION: (ideal) number of months between each timestep in input array t. The default is 0.
        days : int, optional
            DESCRIPTION: (ideal) number of days between each timestep in input array t. The default is 0.
        hours : int, optional
            DESCRIPTION: (ideal) number of hours between each timestep in input array t. The default is 0.
        minutes : int, optional
            DESCRIPTION: (ideal) number of minutes between each timestep in input array t. The default is 0.
        seconds : int, optional
            DESCRIPTION: (ideal) number of seconds between each timestep in input array t. The default is 0.
        wiggleroom_divisor : int, optional
            DESCRIPTION: constant which ensures that the datetime has a bit of wiggleroom. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 2.
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
        from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_missing_samples
        self.t,self.x,self.added_times = _fix_missing_samples(
                                 self.t,
                                 self.x,
                                   years=years,
                                   months=months,
                                   days=days,
                                   hours=hours,
                                   minutes=minutes,
                                   seconds=seconds,
                                   input_dateformat=input_dateformat,
                                   wiggleroom_divisor=wiggleroom_divisor,
                                   missing_value=missing_value)
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
            self.figures.update({'figure_frequency_time':fig_f})
        if fig_p is not None:
            # self.figure_period_time = fig_p    
            self.figures.update({'figure_period_time':fig_p})
        
        #add the results to the results dictionary
        results = self.results
        results.update({'frequency_time_results':{
            'frequency_list'   : self.frequency_list, 
            'period_list'      : self.period_list, 
            'amplitude_matrix' : self.amplitude_matrix, 
            'power_matrix'     : self.power_matrix, 
            'phase_matrix'     : self.phase_matrix, }
            })
        
        results.setdefault('model parameters', {})
        results.get('model parameters').update({
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
                      data_unit:            str = '',
                      alpha:             list = [x/20 for x in range(1,20)],
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
        alpha : list, optional
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
                             self.results.get('components').get('trend').get('reconstructed_data'),
                             self.t,
                             t_unit=t_unit,
                             Y_unit=data_unit,
                             alpha=alpha,
                             timestep=timestep,
                             timestep_unit=timestep_unit,
                             include_data=include_data,
                             legend_loc=legend_loc,
                             shade_area=shade_area,
                             xaxis_rotation=xaxis_rotation
                             )
            self.trend_type = 'Linear'
            self.figures.update({'figure_trend':figure_trend})
            #
        elif trend_type == 'rolling_OLS':
            from pycissa.postprocessing.trend.trend_functions import trend_rolling
            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_rolling(
                              self.results.get('components').get('trend').get('reconstructed_data'),
                              self.t,
                              t_unit=t_unit,
                              Y_unit=data_unit,
                              window=window,
                              alpha=alpha,
                              timestep=timestep,
                              timestep_unit=timestep_unit,
                              include_data=include_data,
                              legend_loc=legend_loc,
                              shade_area=shade_area,
                              xaxis_rotation=xaxis_rotation
                              )
            self.trend_type = 'rolling_OLS'
            self.figures.update({'figure_trend':figure_trend})
            
        else:
            raise ValueError(f"Input value trend_type = {trend_type} is incorrect. Please use one of 'linear' or 'rolling_OLS'.")
       
        
       #update results dictionary
        results = self.results
        results.setdefault('trend results', {})
        results.get('trend results').setdefault(self.trend_type, {})
        results.get('trend results').get(self.trend_type).update({
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
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the run_frequency_time_analysis method.")
        
        self.results, figure_monte_carlo = run_monte_carlo_test(x = self.x,
                             L = self.L,
                             psd=self.psd,
                             results=self.results,
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
        self.figures.update({'figure_monte_carlo':figure_monte_carlo})
        return self
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 



     
    #List of stuff to add in here
    '''  
    check if t is a date, convert it to unix time. Keep t_raw as datetime format
    remove noise
    grouping
    predict method (TO DO, maybe using AutoTS or MAPIE?)
    calculate statistics for each component
    general plot (OG data, trend, periodic, noise/residual)
    gap fill/predict/noise conformal prediction?
    '''    
          
