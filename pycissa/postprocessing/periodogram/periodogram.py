import numpy as np
import matplotlib.pyplot as plt
import piecewise_regression
###############################################################################
###############################################################################

def make_periodogram_arrays(psd         : np.array,
                            frequencies : dict) -> tuple[list,list]:
    '''
    Function to arrange the psd array and frequency dictionary from CiSSA into two lists ready for plotting.

    Parameters
    ----------
    psd : np.array
        DESCRIPTION. Power Spectral Density array.
    frequencies : dict
        DESCRIPTION. Dictionary of frequencies (per unit time-step) and array location.

    Returns
    -------
    my_freq,my_psd  : tuple[list,list]
        DESCRIPTION. list of frequencies and psd.

    '''
    psd = psd.reshape(len(psd),)
    reverse_dictionary  = {value[0]: key for key, value in frequencies.items()}
    reverse_dictionary = dict(sorted(reverse_dictionary.items()))
    my_psd  = []
    my_freq = []
    for key_j in reverse_dictionary.keys():
        if not reverse_dictionary.get(key_j) == 'trend':
            my_psd.append(psd[key_j])
            my_freq.append(reverse_dictionary.get(key_j))
    return my_freq,my_psd    
###############################################################################
###############################################################################
def plot_periodogram(my_freq: list,
                     my_psd : list):
    fig, axs = plt.subplots(figsize=(10, 8), sharex=True)
    
    # Plot
    # axs.plot(np.log(my_freq), np.log(my_psd), 'k', label='Periodogram')
    axs.plot(my_freq,my_psd, 'k', label='Periodogram')
    axs.set_xscale('log')
    axs.set_yscale('log')
    # Set labels and title for the top subplot
    axs.set_ylabel('psd')
    axs.set_xlabel('frequency (1/timestep)')
    axs.set_title('Periodogram')
    
    # Adjust layout
    plt.tight_layout()
    
    # Return the figure object
    return fig

###############################################################################
###############################################################################
def segmented_regression(my_freq         :  list,
                         my_psd          :  list,
                         max_breakpoints : int = 2,
                         n_boot          : int = 500
                         ):
    
    ''' 
    Pilgrim, C. (2021). piecewise-regression (aka segmented regression) in Python. Journal of Open Source Software, 6(68).
    https://joss.theoj.org/papers/10.21105/joss.03859
    '''
    
    ms = piecewise_regression.ModelSelection(np.log(my_freq), 
                                             np.log(my_psd), 
                                             max_breakpoints=max_breakpoints,n_boot=n_boot)
    
    for breakpoint_j in range(0,max_breakpoints+1):
        print(breakpoint_j)
        
    


###############################################################################
###############################################################################
     
