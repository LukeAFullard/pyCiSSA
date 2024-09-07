import numpy as np

def create_time_series_features(x                   : np.ndarray, 
                                time                : np.ndarray|None=None, 
                                add_time_features   : bool=False, 
                                time_features       : list|None=None, 
                                lag                 : int=0, 
                                rolling_window      : int|None=None, 
                                differencing        : bool=False, 
                                lagged_differencing : bool=False):
    '''
    Process a time series to add time features, lag features, rolling statistics, and/or differencing.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION. numpy array of the time series data.
    time : np.ndarray|None, optional
        DESCRIPTION. The default is None. optional numpy array of time points (should be same length as x).
    add_time_features : bool, optional
        DESCRIPTION. The default is False. boolean to indicate if time features should be added.
    time_features : list|None, optional
        DESCRIPTION. The default is None. list of specific time features to generate, e.g., ['year', 'month', 'day', 'hour'].
    lag : int, optional
        DESCRIPTION. The default is 0. integer to create lag features.
    rolling_window : int|None, optional
        DESCRIPTION. The default is None. integer to specify the window size for rolling statistics.
    differencing : bool, optional
        DESCRIPTION. The default is False. boolean to indicate if differencing should be applied.
    lagged_differencing : bool, optional
        DESCRIPTION. The default is False. boolean to indicate if lagged differencing should be applied.

    Returns
    -------
    features : np.ndarray
        DESCRIPTION. Output numpy array features.

    '''
    
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array.")
    
    if time is not None:
        if not isinstance(time, np.ndarray):
            raise ValueError("time must be a numpy array if provided.")
        if len(time) != len(x):
            raise ValueError("Length of time array must match the length of x.")
    
    # Initialize feature matrix
    features_list = []
    
    if time is not None and add_time_features:
        time_features_list = []
        for feature in time_features or []:
            if feature == 'year':
                time_features_list.append(time.astype('datetime64[Y]').astype(int))
            elif feature == 'month':
                time_features_list.append(time.astype('datetime64[M]').astype(int) % 12 + 1)
            elif feature == 'day':
                time_features_list.append(time.astype('datetime64[D]').astype(int) % 365 + 1)
            elif feature == 'hour':
                time_features_list.append((time.astype('datetime64[h]').astype(int) % (24*365)) % 24)
            elif feature == 'minute':
                time_features_list.append((time.astype('datetime64[m]').astype(int) % (24*365*60)) % 60)
            elif feature == 'second':
                time_features_list.append((time.astype('datetime64[s]').astype(int) % (24*365*60*60)) % 60)
        if time_features_list:
            features_list.extend(time_features_list)
    
    # Create lag features
    if lag > 0:
        lag_features = np.hstack([np.roll(x, i).reshape(-1, 1) for i in range(1, lag + 1)])
        lag_features[:lag] = np.nan  # Fill the initial rows with NaNs for shifted values
        features_list.append(lag_features)
    
    # Apply differencing
    if differencing:
        diff_x = np.diff(x, axis=0)
        diff_features = np.concatenate(([np.nan], diff_x), axis=0).reshape(-1, 1)
        features_list.append(diff_features)
    
    # Apply lagged differencing
    if lagged_differencing:
        diff_x = np.diff(x, axis=0)
        lagged_diff_features = np.hstack([np.roll(diff_x, i).reshape(-1, 1) for i in range(1, lag + 1)])
        lagged_diff_features[:lag] = np.nan  # Fill the initial rows with NaNs for shifted values
        features_list.append(lagged_diff_features)
    
    # Apply rolling statistics
    if rolling_window is not None:
        if rolling_window > 1:
            rolling_mean = np.convolve(x, np.ones(rolling_window)/rolling_window, mode='valid')
            rolling_std = np.convolve((x - np.mean(x))**2, np.ones(rolling_window)/rolling_window, mode='valid')**0.5
            rolling_mean = np.concatenate(([np.nan] * (rolling_window - 1), rolling_mean))
            rolling_std = np.concatenate(([np.nan] * (rolling_window - 1), rolling_std))
            rolling_features = np.hstack([rolling_mean.reshape(-1, 1), rolling_std.reshape(-1, 1)])
            features_list.append(rolling_features)
    
    # Concatenate all features
    if features_list:
        features = np.hstack(features_list)
        # Drop rows with NaN values due to shifting or differencing
        valid_rows = ~np.isnan(features).any(axis=1)
        features = features[valid_rows]
    else:
        features = np.empty((0, 0))
    
    return features

# Example usage
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
time = np.arange(len(x))
features = create_time_series_features(x, time=time, add_time_features=True, time_features=['hour'], lag=2, rolling_window=3, differencing=True, lagged_differencing=True)
print(features)

