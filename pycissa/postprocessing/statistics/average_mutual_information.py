import numpy as np
import scipy.sparse as sp
import sys

def calculate_average_mutual_information_Stergiou(data, L, n_bins = 0):
    """
    SEE https://github.com/Nonlinear-Analysis-Core/NONANLibrary/blob/main/python/AMI_Stergiou.py
    inputs    - data, column oriented time series
              - L, maximal lag to which AMI will be calculated
              - bins, number of bins to use in the calculation, if empty an
                adaptive formula will be used

    outputs   - tau, first minimum in the AMI vs lag plot
              - v_AMI, vector of AMI values and associated lags
    
    inputs    - x, single column array with the same length as y.
              - y, single column array with the same length as x.
    outputs   - ami, the average mutual information between the two arrays

    Remarks
    - This code uses average mutual information to find an appropriate lag
      with which to perform phase space reconstruction. It is based on a
      histogram method of calculating AMI.
    - In the case a value of atu could not be found before L the code will
      automatically re-execute with a higher value of L, and will continue to
      re-execute up to a ceiling value of L.

    Future Work
    - None currently.

    Mar 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Modified code to output a plot and notify the user if a value
                of tau could not be found.
    Sep 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Previously the number of bins was hard coded at 128. This
                created a large amount of error in calculated AMI value and
                vastly decreased the sensitivity of the calculation to changes
                in lag. The number of bins was replaced with an adaptive
                formula well known in statistics. (Scott 1979
              - The previous plot output was removed.
    Oct 2017 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Added print commands to display progress.
    May 2019 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - In cases where L was not high enough to find a minimun the
                code would reexecute with a higher L, and the binned data.
                This second part is incorrect and was corrected by using
                data2.
              - The reexecution part did not have the correct input
                parameters.
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha

    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:

    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright 
        notice, this list of conditions and the following disclaimer in the 
        documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its 
        contributors may be used to endorse or promote products derived from 
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    eps = np.finfo(float).eps # smallest floating point value

    if isinstance(L, int):
      N = len(data) 


      data = np.array(data)

      if n_bins == 0:
        bins = np.ceil((np.max(data) - np.min(data))/(3.49 * np.nanstd(data * N**(-1/3), axis=0)))
      else:
        bins = n_bins
      
      bins = int(bins) 

      data = data - min(data) # make all data points positive
      y = np.floor(data/(np.max(data)/(bins - eps)))
      y = np.array(y,dtype=int) # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

      v = np.zeros((L,1)) # preallocate the vector
      overlap = N - L
      increment = 1/overlap

      pA = sp.csr_matrix((np.full(overlap,increment),(y[0:overlap],np.ones(overlap,dtype=int)))).toarray()[:,1]

      v = np.zeros((2, L))

      for lag in range(L): # used to be from 0:L-1 (BS)
        v[0,lag]=lag   
        
        pB = sp.csr_matrix((np.full(overlap,increment),(y[lag:overlap+lag],np.ones(overlap,dtype=int)))).toarray()[:,1]
        # find joint probability p(A,B)=p(x(t),x(t+time_lag))
        pAB = sp.csr_matrix((np.full(overlap,increment), (y[0:overlap], y[lag:overlap+lag])))
        
        (A, B) = np.nonzero(pAB)
        AB = pAB.data

        v[1,lag] = np.sum(np.multiply(AB,np.log2(np.divide(AB,np.multiply(pA[A],pB[B]))))) # Average Mutual Information
          
      tau = np.array(np.full((L,2),-1,dtype=float))

      j = 0
      for i in range(v.shape[1] - 1):                       # Finds first minimum
        if v[1,i-1]>=v[1,i] and v[1,i]<=v[1,i+1]: 
          ami = v[1,i]
          tau[j,:] = np.array([i,ami])
          j+=1

      tau = tau[:j]   # only include filled in data.

      initial_AMI = v[1,0]
      for i in range(v.shape[1]):                       # Finds first AMI value that is 20% initial AMI
        if v[1,i] < (0.2*initial_AMI):
          tau[0,1] = i
          break

      v_AMI=v

      return (tau, v_AMI)
    elif isinstance(L, np.ndarray) or isinstance(L, list):
      x = data if isinstance(data,np.ndarray) else np.array(data)
      y = L if isinstance(L,np.ndarray) else np.array(L)

      if len(x) != len(y):
        raise ValueError('X and Y must be the same size.')
      
      increment = 1/len(y)
      one = np.ones(len(y),dtype=int)

      bins1 = np.ceil((max(x)-min(x))/(3.49*np.nanstd(x)*len(x)**(-1/3))) # Scott 1979
      bins2 = np.ceil((max(y)-min(y))/(3.49*np.nanstd(y)*len(y)**(-1/3))) # Scott 1979
      x = x - min(x) # make all data points positive
      x = np.floor(x/(max(x)/(bins1 - eps))) # scaling the data
      y = y - min(y) # make all data points positive
      y = np.floor(y/(max(y)/(bins2 - eps))) # scaling the data

      x = np.array(x,dtype=int)
      y = np.array(y,dtype=int)

      increment = np.full(len(y),increment)
      pA = sp.csr_matrix((increment,(x,one))).toarray()[:,1]
      pB = sp.csr_matrix((increment,(y,one))).toarray()[:,1]
      pAB = sp.csr_matrix((increment,(x,y)))
      (A, B) = np.nonzero(pAB)
      AB = pAB.data
      ami = np.sum(np.multiply(AB,np.log2(np.divide(AB,np.multiply(pA[A],pB[B])))))
      
      
      return ami
    else:
      raise ValueError('Invalid input, read documentation for input options.')
      
      
###############################################################################
###############################################################################
###############################################################################
###############################################################################

import numpy as np
import timeit as ti
from scipy import stats
from typing import Union
import os

def AMI_Thomas(x : np.ndarray, L : Union[int,np.ndarray,list]) -> Union[np.ndarray,float]:
    """
    SEE https://github.com/Nonlinear-Analysis-Core/NONANLibrary/blob/main/python/AMI_Thomas.py
    Usage: (tau,ami)=AMI_Thomas(x,L)
    inputs:    x - time series, vertically orientedtrc files selected by user
               L - Maximum lag to calculate AMI until
    outputs:   tau - first true minimum of the AMI vs lag plot
               AMI - a vertically oriented vector containing values of AMI
               from a lag of 0 up the input L
    [ami]=AMI_Thomas(x,y)
    inputs:   - x, single column array with the same length as y
              - y, single column array with the same length as x
    outputs   - ami, the average mutual information between the two arrays
    Remarks
    - This code uses a published method of calculating AMI to find an 
      acceptable lag with which to perform phase space reconstruction.
    - The algorithm is publically available at the citation below. Make sure
      to cite this work. The subroutine is fully their work.
    - In the case a value of tau could not be found before L the code will
      return an empty tau and the ami vector.
    - If it does find multiple values of tau but no definative minimum it
      will return all of these values.
    Future Work
    - None.
    References
    - Thomas, R. D., Moses, N. C., Semple, E. A., & Strang, A. J. (2014). An 
      efficient algorithm for the computation of average mutual information: 
      Validation and implementation in Matlab. Journal of Mathematical 
      Psychology, 61(September 2015), 45–59. 
      https://doi.org/10.1016/j.jmp.2014.09.001
    Sep 2015 - Adapted by Ben Senderling, email: bensenderling@gmail.com
                     Below I've set the code published by Thomas, Semple and
                     Strang to calculate AMI at various lags and to suggest
                     an appropriate tau.
    Apr 2021 - Modified by Ben Senderling, email bmchnonan@unomaha.edu
             - Modified in conjunction with NONAN validation efforts.
               Added the variable input arguements and second implementation.
    Validation
    
    Damped oscillator (approximate tau ~ 33)
    
    L=35
    t=(1:500)'
    a=0.005
    w=0.05
    x=exp(-a*t).*sin(w*t)
    
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha
    
    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:
    
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright 
       notice, this list of conditions and the following disclaimer in the 
       documentation and/or other materials provided with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its 
       contributors may be used to endorse or promote products derived from 
       this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR 
    PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if isinstance(L, int):
        if not isinstance(x, np.ndarray):
            x = np.array(x,ndmin=2)
        elif x.ndim == 1:
            x = np.array(x,ndmin=2)

        # check size of input x
        (m,n) = np.shape(x)

        if m > 1 and n > 1:
            raise ValueError('Input time series is not one dimensional.')

        # calculate AMI at each lag
        ami = np.zeros((L,2))
        for i in range(L):
            ami[i,0] = i+1
            X = x[0,:-i-1] if i != 0 else x[0,0:-1]
            Y = x[0,i+1:] if i != 0 else x[0,1:]
            ami[i,1] = average_mutual_information(np.vstack((X,Y)))

        tau = np.array([])
        row = 1 # to keep shape consistent.
        for i in range(1,ami.shape[0]-1):
            if ami[i-1,1] >= ami[i,1] and ami[i,1] <= ami[i+1,1]:
                #NOTE: Axis might be wrong.
                tau = np.append(tau, ami[i,:]).reshape(row,2)
                row += 1
        ind = np.argmax(ami[:,1]<=(0.2*ami[0,1]))
        if ind != 0:    # argmax returns 0 if not found.
            tau = np.append(tau, ami[ind,:]).reshape(row,2)
        return (tau,ami)
    elif isinstance(L, np.ndarray) or isinstance(L, list): 
        if not isinstance(x, np.ndarray):
            x = np.array(x,ndmin=2)
        elif x.ndim == 1:
            x = np.array(x,ndmin=2)
        y = L   # Because L is not a maximal lag, but it is another time series entirely.

        if not isinstance(y, np.ndarray):
            y = np.array(y,ndmin=2)
        elif y.ndim == 1:
            y = np.array(y,ndmin=2)

        if len(y) != len(x):
            raise ValueError('x and y must have the same length.')   

        return average_mutual_information(np.vstack((x,y))) 

def average_mutual_information(data : np.ndarray) -> float:
    """
    Usage: AMI = average_mutual_information(data) 
    Calculates average mutual information between 
    two 
    columns of data. It uses kernel density 
    estimation, 
    with a globally adjusted Gaussian kernel. 
    
    Input should be an n-by-2 matrix, with data sets 
    in adjacent 
    column vectors. 
    
    Output is a scalar.
    """
    n = data.shape[1]
    X = data[0,:]
    Y = data[1,:]
    # Example below is for normal reference rule in 
    # 2 dims, Scott (1992).
    hx = np.std(X,ddof=1)/(n**(1/6))    # MATLAB version has ddof := 1 as default, NumPy := 0 as default
    hy = np.std(Y,ddof=1)/(n**(1/6))
    # Compute univariate marginal density functions.
    P_x = np.array([])
    P_y = np.array([])

    P_x = univariate_kernel_density(X, X, hx) 
    P_y = univariate_kernel_density(Y, Y, hy) 
    # Compute joint probability density. 
    JointP_xy = bivariate_kernel_density(data, data, hx, hy) 
    AMI = np.sum(np.log2(np.divide(JointP_xy,np.multiply(P_x,P_y))))/n
    return AMI

def univariate_kernel_density(value : np.ndarray, data : np.ndarray, window : float) -> np.ndarray:
    """
    Usage:  y = univariate_kernel_density(value, data, window) 
    Estimates univariate density using kernel 
    density estimation. 
    Inputs are: - value (m-vector), where density is estimated 
                - data (n-vector), the data used to estimate the density 
                - window (scalar), used for the width of density estimation. 
    Output is an m-vector of probabilities.
    """
    h = window 
    n = len(data) 
    m = len(value) 
    # We use matrix operations to speed up computation 
    # of a double-sum. 
    prob = np.zeros((n,m))

    G = extended(value, n, True) 
    H = extended(data, m, False) 
    prob = stats.norm.pdf((G-H)/h)
    fhat = np.sum(prob,axis=0)/(n*h) 
    return fhat

def bivariate_kernel_density(value : np.ndarray, data : np.ndarray, Hone : float, Htwo : float) -> np.ndarray:
    """
    Usage: y = bivariate_kernel_density(value, data, Hone, Htwo) 
    Calculates bivariate kernel density estimates 
    of probability. 
    Inputs are: - value (m x 2 matrix), where density is estimated 
                - data (n x 2 matrix), the data used to estimate the density 
                - Hone (scalar) and Htwo (scalar) to use for the widths of density estimation. 
    Output is an m-vector of probabilities estimated at the values in ’value’. 
    """
    s = np.shape(data)
    n = s[1]
    t = np.shape(value) 
    number_pts = t[1] 
    rho_matrix = np.corrcoef(data)
    rho = rho_matrix[0,1]
    # The adjusted covariance matrix: 
    W = np.array([Hone**2,rho*Hone*Htwo,rho*Hone*Htwo,Htwo**2]).reshape((2,2))
    differences = linear_depth(value,np.negative(data))
    prob = stats.multivariate_normal.pdf(differences, cov=W)
    # jitted function would need: n, prob, number_pts
    return bivariate_kernel_density_sub(n,prob,number_pts)


def bivariate_kernel_density_sub(n,prob,number_pts):
    cumprob = np.cumsum(prob)
    y = np.zeros(number_pts)
    y[0] = (1/n)*cumprob[n-1]
    for i in range(1,number_pts):
        index = n*(i+1)
        y[i] = (1/n)*(cumprob[index-1]-cumprob[index-n-1])
    y = y.T.copy()
    return y

def linear_depth(feet : np.ndarray, toes : np.ndarray) -> np.ndarray:
    """
    linear_depth takes a matrix ‘feet’ and lengthens 
    it in blocks, takes a matrix ‘toes’ and lengthens 
    it in Extended repeats, and then adds the
    lengthened ‘feet’ and ‘toes’ matrices to achieve 
    all sum combinations of their rows. 
    feet and toes have the same number of columns 
    """
    if feet.shape[0] == toes.shape[0]:
        a = feet.shape[1]
        b = toes.shape[1]
        blocks = np.zeros((a*b, toes.shape[0]))
        toes = toes.T.copy()
        bricks = blocks.copy()
        for i in range(a):
            blocks[i*b: (i+1)*b,:] = feet[:,i]
            bricks[i*b: (i+1)*b,:] = toes
    y = blocks + bricks 
    return y

def extended(vector : np.ndarray, n : int, vertical : bool) -> np.ndarray:
    """
    Takes an m-dimensional row vector and outputs an 
    n-by-m matrix with n-many consecutive repeats of 
    the vector. Similarly,  it takes an 
    m-dimensional column vector and outputs an 
    m-by-n matrix. 
    Else, it returns the original input. 
    """
    M = np.zeros((len(vector),n))
    for i in range(n):
        if vertical:
            M[:,i] = vector
        else: # if horizontal
            M[i,:] = vector
    return M # where M previously was y.      
