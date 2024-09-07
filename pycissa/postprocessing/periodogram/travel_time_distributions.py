import numpy as np
from math import pi
from scipy.optimize import curve_fit

def gamma_function(x,tau0,exponent):
    return np.power(1 + np.power(2*pi*x*tau0/exponent,2),-exponent)

###############################################################################
###############################################################################
###############################################################################
def fit_gamma(my_freq,my_psd):
    try:popt, pcov = curve_fit(gamma_function, my_freq, my_psd,method = 'trf', x_scale = [1, 1])
    except RuntimeError as e:
        raise RuntimeError(f"{e}")
    condition_number = np.linalg.cond(pcov)
    print(condition_number)
    
    tau0,exponent = popt[0],popt[1]
    return tau0,exponent