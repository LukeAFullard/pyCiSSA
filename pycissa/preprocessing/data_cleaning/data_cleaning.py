import numpy as np

def detect_censored_data(x: np.ndarray)-> bool:
    '''
    This function simply checks to see if any of the entriesof the array x are either '<' or '>'.
    Data of this type are usually called censored and is common in environmental data, for example see
    Helsel, D. R. (2005). More than obvious: better methods for interpreting nondetect data. Environmental science & technology, 39(20), 419A-423A.
    
    Function returns True if any data is censored, otherwise returns False

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Array of input data.

    Returns
    -------
    bool
        DESCRIPTION: If there is censored data detected then returns True, otherwise returns False.

    '''
    censoring = len([y[0] for y in str(x) if y[0] in ['>','<']])
    return censoring > 0

def detect_nan_data(x: np.ndarray)-> bool:
    '''
    This function simply checks to see if any of the entries of the array x are either nan.
    
    Function returns True if any data is nan, otherwise returns False

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Array of input data.

    Returns
    -------
    bool
        DESCRIPTION: If there is nan data detected then returns True, otherwise returns False.

    '''
    def is_nan(value):
        try:
            return np.isnan(float(value))
        except (ValueError, TypeError):
            return False
    nan_entries = len([y for y in x if is_nan(y)])
    return nan_entries > 0
    
###############################################################################
###############################################################################
def _fix_censored_data(x: np.ndarray,
                       replacement_type:     str = 'raw',
                       lower_multiplier: float = 0.5,
                       upper_multiplier: float = 1.1,
                       default_value_lower:    float = 0.,
                       default_value_upper:    float = 0.,
                       hicensor_lower:   bool = False,
                       hicensor_upper:   bool = False) -> tuple[np.ndarray, np.ndarray]:
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
    
    
    # if value is numeric, leave as is, if starts with < or >, uncensor and store censoring,
    # if other, return np.nan
    if not replacement_type in ['raw', 'multiple', 'constant']:
        raise ValueError(f'Parameter "replacement_type" should be one of "raw", "multiple", or "constant". Currently  replacement_type = {replacement_type}')
    if not type(lower_multiplier) in [float, int]:
        raise ValueError(f'Parameter "lower_multiplier" = {lower_multiplier} but should be numeric.')
    if not type(upper_multiplier) in [float, int]:
        raise ValueError(f'Parameter "upper_multiplier" = {upper_multiplier} but should be numeric.')
    if not type(default_value_lower) in [float, int]:
        raise ValueError(f'Parameter "default_value_lower" = {default_value_lower} but should be numeric.')    
    if not type(default_value_upper) in [float, int]:
        raise ValueError(f'Parameter "default_value_upper" = {default_value_upper} but should be numeric.')        
    
    
    x_uncensored = []
    x_censoring  = []
    for entry_i in x:
        if type(entry_i) in [int,float]:  #is numeric
            x_uncensored.append(entry_i)
            x_censoring.append(None)
        elif type(entry_i) in [str]:
            try:  #if happens to be a number but stored as a string, convert to float
                x_uncensored.append(float(entry_i))
                x_censoring.append(None)
            except:
                if entry_i == '':
                    x_uncensored.append(np.nan)
                    x_censoring.append(None)
                elif entry_i[0] == '<':  #if first character of string is <, data is left censored
                    try:   
                        if replacement_type == 'raw':
                            x_uncensored.append(float(entry_i[1:]))
                        elif replacement_type == 'multiple':
                            x_uncensored.append(lower_multiplier*float(entry_i[1:]))
                        elif replacement_type == 'constant':    
                            x_uncensored.append(default_value_lower)
                    except:x_uncensored.append(np.nan)
                    x_censoring.append('<')
                elif entry_i[0] == '>': #if first character of string is >, data is right censored
                    try:   
                        if replacement_type == 'raw':
                            x_uncensored.append(float(entry_i[1:]))
                        elif replacement_type == 'multiple':
                            x_uncensored.append(upper_multiplier*float(entry_i[1:]))  
                        elif replacement_type == 'constant':    
                            x_uncensored.append(default_value_upper)    
                    except:x_uncensored.append(np.nan)
                    x_censoring.append('>')
                else: #if first character of the string is not < or > then return nan
                    x_uncensored.append(np.nan)
                    x_censoring.append(None)
        else: #return nan
            x_uncensored.append(np.nan)
            x_censoring.append(None)
    
    
    x_uncensored = np.array(x_uncensored, dtype=float) 
    x_censoring  = np.array(x_censoring, dtype=object)   

    ###########################################################################
    ###########################################################################
    # if hicensor_lower then we replace the lower censored values with the max 
    #  censored value
    if hicensor_lower:
        x_uncensored[x_censoring == '<'] = max(x_uncensored[x_censoring == '<'])
        
    # if hicensor_upper then we replace the upper censored values with the min 
    #  censored value
    if hicensor_upper:
        x_uncensored[x_censoring == '>'] = min(x_uncensored[x_censoring == '>'])
    ###########################################################################
    ###########################################################################        
    #need to ensure that the final data is float64 and not object    
    x_uncensored = np.array(x_uncensored, dtype=float)
    return x_uncensored,x_censoring

###############################################################################
###############################################################################

def _fix_missing_samples(t: np.ndarray, 
                         x: np.ndarray,
                           years:              int = 0, 
                           months:             int = 0, 
                           days:               int = 0, 
                           hours:              int = 0,
                           minutes:            int = 0,
                           seconds:            int = 0,
                           input_dateformat:   str = '%Y',
                           wiggleroom_divisor: int = 2,
                           missing_value:      int = np.nan
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function that finds and corrects misisng values in the time series.
    Missing dates result in adding a default value "missing_value" into the input data.
    
    **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: array of input times/dates.
    x : np.ndarray
        DESCRIPTION: array of input data.
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
    input_dateformat : str, optional
        DESCRIPTION: Datetime string format. The default is '%Y'. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
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
    import datetime
    from dateutil.relativedelta import relativedelta
    import copy
    
    #check that sum is not = 0
    if not years+months+days+hours+minutes+seconds>0: 
        raise ValueError('One of the input parameters years, months, weeks, days, hours, minutes, seconds must be greater than zero')
    
    t_ = copy.deepcopy(t)
    t_ = [np.datetime64(dt, 's') for dt in t_]
    new_t = []
    for time_i in t_:
        if type(time_i) in [str]:
            new_t.append(datetime.datetime.strptime(time_i, input_dateformat))
        elif type(time_i) in [datetime.datetime]:
            new_t.append(time_i)
        else:
            new_t.append(time_i)
            
    
    min_date = min(new_t)
    max_date = max(new_t)
    date_delta_with_wiggle_room = datetime.timedelta(weeks = 52.1429*years/wiggleroom_divisor + 4.34524*months/wiggleroom_divisor,
                                    days = days/wiggleroom_divisor,
                                    hours = hours/wiggleroom_divisor,
                                    minutes = minutes/wiggleroom_divisor + (seconds/60)/wiggleroom_divisor
                                    )
    date_delta_with_wiggle_room = np.timedelta64(date_delta_with_wiggle_room)
    date_delta = relativedelta(years = years, months = months, days = days, hours = hours, minutes = minutes, seconds = seconds)
    # date_delta = np.timedelta64(date_delta)
    
    all_dates = []
    all_x = []
    x_missing = []
    current_date = min_date
    for time_i, x_i in zip(new_t,x):
        
        print(str(current_date))
        print(date_delta)

        print(current_date - date_delta_with_wiggle_room)
        print(time_i)
        print(current_date + date_delta_with_wiggle_room)
        print((time_i > current_date - date_delta_with_wiggle_room))
        print((time_i < current_date + date_delta_with_wiggle_room))
        
        if (time_i > current_date - date_delta_with_wiggle_room) & (time_i < current_date + date_delta_with_wiggle_room):
            # Here date is within the acceptable range
            all_dates.append(time_i)
            all_x.append(x_i)
            x_missing.append(None)
            current_date += date_delta
        else:
            # Here date is missing
            while current_date < time_i + date_delta_with_wiggle_room:
                all_dates.append(current_date)
                if (time_i > current_date - date_delta_with_wiggle_room) & (time_i < current_date + date_delta_with_wiggle_room):
                    all_x.append(x_i)
                    x_missing.append(None)
                else:
                    all_x.append(missing_value)
                    x_missing.append(True)
                current_date += date_delta
    final_t    =  np.array(all_dates, dtype=object)    
    final_x    =  np.array(all_x, dtype=object)  
    x_missing  =  np.array(x_missing, dtype=object)    
          
    return final_t, final_x, x_missing


def _fix_missing_date_samples(t: np.ndarray, 
                         x: np.ndarray,
                           years:              int = 0, 
                           months:             int = 1, 
                           days:               int = 0, 
                           hours:              int = 0,
                           minutes:            int = 0,
                           seconds:            int = 0,
                           input_dateformat:   str = '%Y',
                           wiggleroom_divisor: int = 2,
                           missing_value:      int = np.nan
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function that finds and corrects misisng dates in the time series.
    Missing dates result in adding a default value "missing_value" into the input data.
    
    **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: array of input times/dates.
    x : np.ndarray
        DESCRIPTION: array of input data.
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
    # import datetime
    from datetime import datetime
    import copy
    if wiggleroom_divisor ==0: wiggleroom_divisor = 1e100
    def add_date_delta(mydate,years,months,days,hours,minutes,seconds,direction):
        if direction == 'subtract':
            years *= -1
            months *= -1
            days *= -1
            hours *= -1
            minutes *= -1
            seconds *= -1
        print( "since the replace function will only allow adding integers to the date, we need to think of a better way to handle the wiggleroom. COuld use the relative delta for the wiggleroom? Or define input parameters like we do for year, month, day etc...")
        mydate = mydate.replace(year=mydate.year+years)
        mydate = mydate.replace(month=mydate.month+months)
        mydate = mydate.replace(day=mydate.day+days)
        mydate = mydate.replace(hour=mydate.hour+hours)
        mydate = mydate.replace(minute=mydate.minute+minutes)
        mydate = mydate.replace(second=mydate.second+seconds)
        return mydate
    
    
    #check that sum is not = 0
    if not years+months+days+hours+minutes+seconds>0: 
        raise ValueError('One of the input parameters years, months, weeks, days, hours, minutes, seconds must be greater than zero')
    
    t_ = copy.deepcopy(t)
    t_ = [np.datetime64(dt, 's') for dt in t_]
    t_ = [dt.astype(datetime) for dt in t_]
    
    new_t = []
    for time_i in t_:
        if type(time_i) in [str]:
            new_t.append(datetime.strptime(time_i, input_dateformat))
        elif type(time_i) in [datetime]:
            new_t.append(time_i)
        else:
            new_t.append(time_i)
            
    
    min_date = min(new_t)
    max_date = max(new_t)

    # mydate = mydate.replace(day=mydate.day+1)
    
    all_dates = []
    all_x = []
    x_missing = []
    current_date = min_date
    for time_i, x_i in zip(new_t,x):
        
        lower_date = add_date_delta(current_date,years/wiggleroom_divisor,months/wiggleroom_divisor,days/wiggleroom_divisor,hours/wiggleroom_divisor,minutes/wiggleroom_divisor,seconds/wiggleroom_divisor,'subtract')
        upper_date = add_date_delta(current_date,years/wiggleroom_divisor,months/wiggleroom_divisor,days/wiggleroom_divisor,hours/wiggleroom_divisor,minutes/wiggleroom_divisor,seconds/wiggleroom_divisor,'add')
        if (time_i > lower_date) & (time_i < upper_date):
            # Here date is within the acceptable range
            all_dates.append(time_i)
            all_x.append(x_i)
            x_missing.append(None)
            current_date = add_date_delta(current_date,years,months,days,hours,minutes,seconds,'add')
        else:
            # Here date is missing
            while current_date < upper_date:
                all_dates.append(current_date)
                if (time_i > lower_date) & (time_i < upper_date):
                    all_x.append(x_i)
                    x_missing.append(None)
                else:
                    all_x.append(missing_value)
                    x_missing.append(True)
                current_date = add_date_delta(current_date,years,months,days,hours,minutes,seconds,'add')
    final_t    =  np.array(all_dates, dtype=object)    
    final_x    =  np.array(all_x, dtype=object)  
    x_missing  =  np.array(x_missing, dtype=object)    
          
    return final_t, final_x, x_missing
    
