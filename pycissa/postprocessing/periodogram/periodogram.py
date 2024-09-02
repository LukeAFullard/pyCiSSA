import numpy as np
import matplotlib.pyplot as plt
import piecewise_regression
import statsmodels.api as sm
import warnings
###############################################################################
###############################################################################

def make_periodogram_arrays(psd                    : np.array,
                            frequencies            : dict,
                            significant_components :  list|None = None,
                            ) -> tuple[list,list]:
    '''
    Function to arrange the psd array and frequency dictionary from CiSSA into two lists ready for plotting.

    Parameters
    ----------
    psd : np.array
        DESCRIPTION. Power Spectral Density array.
    frequencies : dict
        DESCRIPTION. Dictionary of frequencies (per unit time-step) and array location.
    significant_components : list|None, optional   
        DESCRIPTION. The default is None. If provided, significant_components is a list of integers. These components will be ignored in the  list of frequencies.

    Returns
    -------
    my_freq,my_psd  : tuple[list,list]
        DESCRIPTION. list of frequencies and psd.

    '''
    if significant_components is None:
        significant_components = []
        
    psd = psd.reshape(len(psd),)
    reverse_dictionary  = {value[0]: key for key, value in frequencies.items()}
    reverse_dictionary = dict(sorted(reverse_dictionary.items()))
    my_psd  = []
    my_freq = []
    for key_j in reverse_dictionary.keys():
        if not key_j in significant_components:
            if not reverse_dictionary.get(key_j) == 'trend':
                my_psd.append(psd[key_j])
                my_freq.append(reverse_dictionary.get(key_j))
    return my_freq,my_psd    

###############################################################################
###############################################################################
def linear_fit(my_freq,my_psd,alpha=0.05):
    Z = np.array([np.log10(my_freq)])
    Z = Z.T
    Z = sm.add_constant(Z, has_constant='add')
    # Basic OLS fit
    results = sm.OLS(endog=np.array(np.log10(my_psd)), exog=Z).fit()
    ols_result = {
        'constant'  :  {
                        'result'              : results.params[0],
                        'confidence_interval' : results.conf_int(alpha=alpha)[0],
                        },
        'slope'     :  {'result'              : results.params[1],
                        'confidence_interval' : results.conf_int(alpha=alpha)[1],
                        }
        }
    return ols_result

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
    if max_breakpoints > 1:
        max_breakpoints = 1
        warnings.warn("For now max_breakpoints must be 0 or 1. This may change in the future. Resetting value to 1,")
        
    ms = piecewise_regression.ModelSelection(np.log10(my_freq), 
                                             np.log10(my_psd), 
                                             max_breakpoints=max_breakpoints,n_boot=n_boot)
    
    model_summaries = [x for x in ms.model_summaries if x['converged'] == True and x['n_breakpoints'] == 1]
    models = [x for x in ms.models if x.best_muggeo and x.n_breakpoints == 1]
    return model_summaries,models

###############################################################################
###############################################################################
def plot_linear_fit(my_freq,my_psd,alpha,ols_result,**kwargs):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    #scatter data
    ax.scatter(np.log10(my_freq), np.log10(my_psd),color='k',label='data', **kwargs)
    
    #add line:
    xx_plot = np.linspace(min(np.log10(my_freq)), max(np.log10(my_freq)), 100)
    yy_plot = ols_result.get('constant').get('result') + xx_plot*ols_result.get('slope').get('result')
    ax.plot(xx_plot, yy_plot,'r',label='linear fit', **kwargs)    
    
    #axes labels etc
    ax.set_ylabel('log(psd)')
    ax.set_xlabel('log(frequency) (cycles per timestep)')
    ax.legend(loc='upper right')
    fig.suptitle('Periodogram - linear fit', y=1.15, fontsize=18)
    constant_result = f"{ols_result['constant']['result']:.3f}"
    constant_ci = [f"{ci:.3f}" for ci in ols_result['constant']['confidence_interval']]
    slope_result = f"{ols_result['slope']['result']:.3f}"
    slope_ci = [f"{ci:.3f}" for ci in ols_result['slope']['confidence_interval']]
    ax.set_title(
    f"Fitting parameters with {int(100*(1-alpha))}% confidence interval.\n"   
    f"Constant: {constant_result} "
    f"({constant_ci[0]} - {constant_ci[1]})\n"
    f"Slope: {slope_result} "
    f"({slope_ci[0]} - {slope_ci[1]})",
    fontsize=10
    )
    
    return fig
     
###############################################################################
###############################################################################
def plot_segmented_fit(my_freq,my_psd,alpha,model_summaries,models,**kwargs):
    
    for model_summary_j in model_summaries:
        #only using 1 breakpoint plot. May change this in the future.
        if model_summary_j.get('n_breakpoints') == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            #scatter data
            ax.scatter(np.log10(my_freq), np.log10(my_psd),color='k',label='data', **kwargs)

            #plot lines
            for model_k in models:
                if model_k.n_breakpoints == 1:
                    xx_plot = np.linspace(min(np.log10(my_freq)), max(np.log10(my_freq)), 100)
                    yy_plot = model_k.predict(xx_plot)
                    ax.plot(xx_plot, yy_plot,'r',label='segmented linear fit', **kwargs)
                    #axes labels etc
                    ax.set_ylabel('log(psd)')
                    ax.set_xlabel('log(frequency) (cycles per timestep)')
                    ax.legend(loc='upper right')
                    fig.suptitle('Periodogram - segmented linear fit', y=1.15, fontsize=18)
                    
                    #plot breakpoint
                    breakpoints = model_k.best_muggeo.best_fit.next_breakpoints
                    for bp in breakpoints:
                        ax.axvline(bp, **kwargs)
                    
                    # Extract and format values
                    breakpoint1_estimate = np.power(10, model_summary_j['estimates']['breakpoint1']['estimate'])
                    alpha1_estimate = model_summary_j['estimates']['alpha1']['estimate']
                    alpha1_ci = model_summary_j['estimates']['alpha1']['confidence_interval']
                    alpha1_ci_lower, alpha1_ci_upper = alpha1_ci  # Unpack tuple
                    
                    alpha2_estimate = model_summary_j['estimates']['alpha2']['estimate']
                    alpha2_ci = model_summary_j['estimates']['alpha2']['confidence_interval']
                    alpha2_ci_lower, alpha2_ci_upper = alpha2_ci  # Unpack tuple
                    
                    # Set title with formatted values
                    title_text = (
                        f"Slopes with {int(100 * (1 - alpha))}% confidence interval.\n"
                        f"For frequency < {breakpoint1_estimate:.4f}, slope: {alpha1_estimate:.3f} "
                        f"({alpha1_ci_lower:.3f} - {alpha1_ci_upper:.3f})\n"
                        f"For frequency > {breakpoint1_estimate:.4f}, slope: {alpha2_estimate:.3f} "
                        f"({alpha2_ci_lower:.3f} - {alpha2_ci_upper:.3f})"
                    )
                    
                    ax.set_title(title_text, fontsize=10)
                        
    return fig                    
            



###############################################################################
###############################################################################
def generate_peridogram_plots(psd                    : np.array,
                            frequencies            : dict,
                            significant_components : list|None = None,
                            alpha                  : float = 0.05,
                            max_breakpoints        : int = 2,
                            n_boot                 : int = 500,
                            **kwargs):
    #get psd and frequencies of interest
    my_freq,my_psd   = make_periodogram_arrays(psd, frequencies,significant_components=significant_components)
    
    #make linear plot.
    ols_result = linear_fit(my_freq,my_psd,alpha=alpha)
    fig_linear = plot_linear_fit(my_freq,my_psd,alpha,ols_result,**kwargs)
    linear_slopes = { 'slope' : ols_result['slope']['result'],
     'confidence_interval' : ols_result['slope']['confidence_interval']}
    
    #make segmented linear plot
    model_summaries,models = segmented_regression(my_freq,my_psd,max_breakpoints=max_breakpoints,n_boot=n_boot)
    fig_segmented = plot_segmented_fit(my_freq,my_psd,alpha,model_summaries,models,**kwargs)
    segmented_slopes = {
        'breakpoint' : np.power(10, model_summaries[0]['estimates']['breakpoint1']['estimate']),
        'slope_less_than_breakpoint' : {
                                    'slope'               : model_summaries[0]['estimates']['alpha1']['estimate'],
                                    'confidence_interval' : model_summaries[0]['estimates']['alpha1']['confidence_interval'],
                                    },
        'slope_greater_than_breakpoint' : {
                                    'slope'               : model_summaries[0]['estimates']['alpha2']['estimate'],
                                    'confidence_interval' : model_summaries[0]['estimates']['alpha2']['confidence_interval'],
                                    }
        }
    
    return fig_linear, fig_segmented, linear_slopes, segmented_slopes
    
    
         



###############################################################################
###############################################################################
     



###############################################################################
###############################################################################
     



###############################################################################
###############################################################################
     

