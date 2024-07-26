import numpy as np


###############################################################################
###############################################################################
def create_trajactory_matrix(x_e: np.ndarray,
                             L  : int) -> np.ndarray:
    '''
    Function that generates the required trajectory matrix for CiSSA.
    See https://en.wikipedia.org/wiki/Singular_spectrum_analysis#Methodology

    Parameters
    ----------
    x_e : np.ndarray
        DESCRIPTION: Input 1D array data
    L : int
        DESCRIPTION: CiSSA window length.

    Returns
    -------
    X : np.ndarray
        DESCRIPTION: Trajectory matrix

    '''
    from scipy.linalg import hankel
    #Trajectory matrix
    col = x_e[0:L]
    row = x_e[L-1:]
    X = hankel(col,row)
    return X
###############################################################################
###############################################################################
def create_autocovariance_function(x: np.ndarray,
                                   L: int,
                                   T: int) -> np.ndarray:
    '''
    Function to calculate the # autocovariance function https://arxiv.org/pdf/2102.01742

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input numpy array
    L : int
        DESCRIPTION: CiSSA window length.
    T : int
        DESCRIPTION: Input array length

    Returns
    -------
    gam : np.ndarray
        DESCRIPTION: autocovariance function

    '''
    
    gam = np.zeros((L,1))
    for k in range(0,L):
        gam[k] = np.matmul((x[0:T-k]-np.mean(x)).transpose(),(x[k:T+1]-np.mean(x))/(T-k))
    return gam    
###############################################################################
###############################################################################
def create_toeplitz_circulant_matrices(x: np.ndarray,
                                       L: int,
                                       T: int,
                                       generate_toeplitz_matrix: bool = False
                                       ) -> tuple[np.ndarray|None, np.ndarray]:
    '''
    Function to generate the circulant matrix (C) and optonally the symmetric Toeplitz matrix (S).
    See https://arxiv.org/pdf/2102.01742.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input numpy array
    L : int
        DESCRIPTION: CiSSA window length.
    T : int
        DESCRIPTION: Input array length
    generate_toeplitz_matrix : bool, optional
        DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False.

    Returns
    -------
    S : np.ndarray OR None
        DESCRIPTION: If generate_toeplitz_matrix == True, the symmetric Toeplitz matrix, otherwise None
    C : np.ndarray
        DESCRIPTION: Circulant matrix.

    '''
    
    S = None
    gam = create_autocovariance_function(x,L,T)
    if generate_toeplitz_matrix:
        S = gam[0]*np.eye(L) 
    C = gam[0]*np.eye(L) 
    for i in range(0,L):
        for j in range(i+1,L):
            k = np.abs(i-j)
            if generate_toeplitz_matrix:
                S[i,j] = gam[k] 
                S[j,i] = S[i,j]
            C[i,j] = ((L-k)/L)*gam[k]+(k/L)*gam[L-k] # Pearl (1973)
            C[j,i] = C[i,j];
    return S,C        

###############################################################################
###############################################################################
def calculate_number_of_frequencies(L: int) -> int:
    '''
    Simple function to calculate the number of frequencies for a given window length L.

    Parameters
    ----------
    L : int
        DESCRIPTION: CiSSA window length.

    Returns
    -------
    nf2 : int
        DESCRIPTION: Rounded series half length.
    nft : int
        DESCRIPTION: Number of expected frequencies.

    '''
    #Number of symmetryc frequency pairs around 1/2
    if np.mod(L,2):
        nf2 = (L+1)/2-1
    else:
        nf2 = L/2-1
    #Number of frequencies <= 1/2
    nft = nf2+np.abs(np.mod(L,2)-2)
    return  nf2,nft

###############################################################################
###############################################################################

def calculate_elementary_matrix(L:int)-> np.ndarray:
    '''
    Function to create an elementary matrix

    Parameters
    ----------
    L : int
        DESCRIPTION: CiSSA window length.

    Returns
    -------
    U : np.ndarray
        DESCRIPTION: elementary matrix

    '''
    from scipy.linalg import dft
    
    nf2,nft = calculate_number_of_frequencies(L)
    
    # Eigenvectors of circulant matrix (unitary base)
    U = dft(L)/np.sqrt(L)

    #Real eigenvectors (orthonormal base)
    U[:,0] = np.real(U[:,0])
    
    for k in range(1,int(nf2+1)):
        u_k = U[:,k]

        new_col_1,new_col_2 = None,None
        new_col_1 = ((np.sqrt(2))*(np.real(u_k)))
        new_col_2 = np.sqrt(2)*np.imag(u_k)
        U[:,k] = new_col_1
        U[:,L+2-(k+1)-1] = new_col_2
    U = np.real(U)
    del u_k,new_col_1,new_col_2
      
    if not np.mod(L,2):
        U[:,int(nft-1)] = np.real(U[:,int(nft-1)]);
    return U    

###############################################################################
###############################################################################
def estimate_power_spectral_density(U: np.ndarray,
                                    C: np.ndarray
                                    ) -> np.ndarray:
    '''
    Function to estimate the circulant matrix power spectral density (https://arxiv.org/pdf/2102.01742)

    Parameters
    ----------
    U : np.ndarray
        DESCRIPTION: elementary matrix
    C : np.ndarray
        DESCRIPTION: Circulant matrix.

    Returns
    -------
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density

    '''
    # #Eigenvalues of circulant matrix: estimated power spectral density
    psd = np.abs(np.diag(np.matmul(U.transpose(),np.matmul(C,U))))
    return psd

###############################################################################
###############################################################################
def calculate_principal_components(U: np.ndarray,
                                   X: np.ndarray) -> np.ndarray:
    '''
    Function to calculate the principal components of (https://arxiv.org/pdf/2102.01742)

    Parameters
    ----------
    U : np.ndarray
        DESCRIPTION: elementary matrix
    X : np.ndarray
        DESCRIPTION: Trajectory matrix

    Returns
    -------
    W : np.ndarray
        DESCRIPTION: Matrix of principal components

    '''
    # #Principal components
    W = np.matmul(U.transpose(),X)
    return W

###############################################################################
###############################################################################
def diagonal_average(Y: np.ndarray) -> np.ndarray:
    '''
    Function to perform diagonal averaging for Singular Spectrum Analysis. https://doi.org/10.1016/j.sigpro.2020.107824
    This function transforms the numpy matrix, Y, into the time series,
    y, by diagonal averaging. This entails averaging the elements of Y over its antidiagonals.
    See also https://github.com/jbogalo/CiSSA
    
    This function uses multi-threaded computing, by default uses all cpu cores, This may be modified at a later date. 
    For single threaded version please see the diagonal_average_single_thread function

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021). 
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction". 
          Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    ------------------------------------------------------------------------- 
    
    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input 2D numpy array/matrix

    Returns
    -------
    y : np.ndarray
        DESCRIPTION: Output diagonally averaged 1D array

    '''
   
    import multiprocessing

    LL, NN = Y.shape
    if LL > NN: 
        Y = Y.transpose()
    L = min(LL, NN)
    N = max(LL, NN)
    T = N + L - 1
    y = np.zeros((T, 1))
    
    # Create a list of tuples to pass to the worker processes
    tasks = [(t, Y, N, L, T) for t in range(T)]
    
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Use map to apply the function to each tuple in the list in parallel
        results = pool.starmap(diagonal_average_worker, tasks)
    
    # Merge the results of the worker processes back into a single array
    y = np.concatenate(results, axis=0)
    
    return y

def diagonal_average_worker(t: int, 
                            Y: np.ndarray, 
                            N: int, 
                            L: int,
                            T: int) -> np.ndarray:
    '''
    Calculation function for the multiprocessing version of diagaver.

    Parameters
    ----------
    t : int
        DESCRIPTION: integer giving the current value, t in T. NOT the cissa input time array 
    Y : np.ndarray
        DESCRIPTION: Input array
    N : int
        DESCRIPTION: array size 1
    L : int
        DESCRIPTION: array size 2 (NOT THE CISSA WINDOW LENGTH)
    T : int
        DESCRIPTION: T = N + L - 1

    Returns
    -------
    np.ndarray
        DESCRIPTION: diagonally averaged array

    '''
    import numpy as np
    if (1 <= t + 1) & (t + 1 <= L - 1):
        j_inf = 1
        j_sup = t + 1
    elif (L <= t + 1) & (t + 1 <= N):
        j_inf = 1
        j_sup = L
    else:
        j_inf = t + 1 - N + 1
        j_sup = T - N + 1
    nsum = j_sup - j_inf + 1
    y_t = 0
        
    # # Create a NumPy array of indices
    indices = np.arange(j_inf, j_sup + 1)
    # # Use the indices to select the relevant elements from Y
    y_t = np.sum(Y[indices - 1, t + 1 - indices] / nsum)
    
    return y_t.reshape(-1, 1)

###############################################################################
###############################################################################
def diagonal_average_single_thread(Y: np.ndarray) -> np.ndarray:
    '''
    Function to perform diagonal averaging for Singular Spectrum Analysis. https://doi.org/10.1016/j.sigpro.2020.107824
    This function transforms the numpy matrix, Y, into the time series,
    y, by diagonal averaging. This entails averaging the elements of Y over its antidiagonals.
    See also https://github.com/jbogalo/CiSSA
    
    This function uses single threaded computing. For faster results please see the diagonal_average function

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021). 
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction". 
          Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    ------------------------------------------------------------------------- 
    
    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input 2D numpy array/matrix

    Returns
    -------
    y : np.ndarray
        DESCRIPTION: Output diagonally averaged 1D array

    '''
    
    ###########################################################################
    # 1) Realignment
    ###########################################################################
    #Get shape of matrix
    LL, NN = Y.shape
    
    # If number of columns greater than number of rows, transpose the matrix 
    if LL>NN: 
        Y = Y.transpose()
    ###########################################################################    
    
    
    ###########################################################################
    # 2) Dimensions
    ###########################################################################
    L = min(LL,NN);
    N = max(LL,NN);
    T = N+L-1;
    ###########################################################################
    
    
    ###########################################################################
    # 3) Diagonal averaging
    ###########################################################################
    #create empty vector of size (T,1)
    y = np.zeros((T, 1))
    
    #perform diagonial averaging
    for t in range(1,T+1):  
        if (1<=t) & (t<=L-1):
            j_inf = 1; j_sup = t;
        elif (L<=t) & (t<=N):
            j_inf = 1; j_sup = L;
        else:
            j_inf = t-N+1; j_sup = T-N+1;
        nsum = j_sup-j_inf+1;
        for m in range(j_inf,j_sup+1):
            y[t-1] = y[t-1]+Y[m-1,t-m]/nsum

    return y

###############################################################################
###############################################################################
def define_left_and_right_extension_lengths(extension_type: str,
                                            T: int,
                                            L: int) -> tuple[int,int]:
    '''
    

    Parameters
    ----------
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series 
    T : int
        DESCRIPTION: Input array length
    L : int
        DESCRIPTION: CiSSA window length.

    Returns
    -------
    left_ext: int 
        DESCRIPTION: length of left time series extension
    right_ext: int 
        DESCRIPTION: length of right time series extension

    '''
    if extension_type == 'Mirror':
        left_ext  = T
        right_ext = T
    elif extension_type == 'NoExt':
        left_ext  = 0
        right_ext = 0
    elif extension_type == 'AR_LR':
        left_ext  = L
        right_ext = L
    elif extension_type == 'AR_L':
        left_ext  = L
        right_ext = 0
    elif extension_type == 'AR_R':
        left_ext  = 0
        right_ext = L
    else:
        raise ValueError(f'Input parameter extension_type ({extension_type}) should be one of AR_LR, AR_L, AR_R, Mirror, NoExt')
    
    return left_ext, right_ext        
        
def perform_reconstruction(U: np.ndarray,
                           W: np.ndarray,
                           T: int,
                           L: int,
                           extension_type: str,
                           multi_thread_run: bool = True
                               ) -> np.ndarray:
    '''
    Function to perform matrix reconstruction (see 4th CiSSA step https://arxiv.org/pdf/2102.01742)

    Parameters
    ----------
    U : np.ndarray
        DESCRIPTION: elementary matrix
    W : np.ndarray
        DESCRIPTION: Matrix of principal components
    T : int
        DESCRIPTION: Input array length
    L : int
        DESCRIPTION: CiSSA window length.
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series   
    multi_thread_run : bool, optional
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.

    Returns
    -------
    R : np.ndarray
        DESCRIPTION: Reconstructed array

    '''
    left_ext,right_ext = define_left_and_right_extension_lengths(extension_type,T,L)
    # Elementary reconstructed series
    R = np.zeros((T+(right_ext+left_ext),L))
    for k in range(0,L):
        if multi_thread_run:
            R[:,[k]] = diagonal_average(np.matmul(U[:,[k]],W[[k],:]))
        else:
            R[:,[k]] = diagonal_average_single_thread(np.matmul(U[:,[k]],W[[k],:]))
            
    return R        

###############################################################################
###############################################################################

def group_paired_frequencies(R: np.ndarray,
                             L: int,
                             T: int,
                             extension_type: str
                             ) -> np.ndarray:
    '''
    Function to group paired frequencies. (see 3rd CiSSA step https://arxiv.org/pdf/2102.01742)

    Parameters
    ----------
    R : np.ndarray
        DESCRIPTION: Reconstructed array
    L : int
        DESCRIPTION: CiSSA window length.
    T : int
        DESCRIPTION: Input array length
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series   

    Returns
    -------
    Z : np.ndarray
        DESCRIPTION: Grouped array

    '''

    left_ext, right_ext   = define_left_and_right_extension_lengths(extension_type,T,L)
    nf2,nft = calculate_number_of_frequencies(L) 
    Z = np.zeros((T+(right_ext+left_ext),int(nft)))
    Z[:,[0]] = R[:,[0]]
    for k in range(1,int(nf2)+1):
        Z[:,[k]] = R[:,[k]]+R[:,[L+2-(k+1)-1]];

    if not np.mod(L,2):
        Z[:,int(nft-1)] = R[:,int(nft-1)]

    lcol,lrow = Z.shape
    Z = Z[left_ext:lcol-right_ext,:]
    return Z
###############################################################################
###############################################################################

###############################################################################
###############################################################################
def cissa_input_checks(x: np.ndarray,
              extension_type: str,
              L: int) -> tuple[np.ndarray,int,int]:
    '''
    Function that performs input checking for cissa input parameters.
    Also calculates the lenth of the series.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: input time-series array
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series   
    L : int
        DESCRIPTION: CiSSA window length

    Returns
    -------
    x : np.ndarray
        DESCRIPTION: Possible reshaped data array
    T : int
        DESCRIPTION: Input array length
    N : int
        DESCRIPTION: Integer to ensure window length OK

    '''
    if not type(extension_type) == str:
        raise('Input parameter "extension_type" should be a string, one of AR_LR, AR_L, AR_R, Mirror, NoExt')
    if not type(L) == int:
        raise('Input parameter "L" should be an integer')  
    #check x is a numpy array
    if not type(x) is np.ndarray:
        try: 
            x = np.array(x)
            x = x.reshape(len(x),1)
        except: raise ValueError(f'Input "x" is not a numpy array, nor can be converted to one.')
    myshape = x.shape
    if len(myshape) == 2:
        rows, cols = myshape[0],myshape[1]
    else:
        try: 
            x = x.reshape(len(x),1)
            rows, cols = x.shape
        except:
            raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({myshape})')
    
    #check that no values are nan
    if len([y for y in x if np.isnan(y)]) > 0:
        raise ValueError('Input "x" should be free of nan values. Please fix these nan values either manually or using the fill_gaps function')
        
    
    if rows==1: #we want a column vector
        x = x.transpose()
    T = len(x)
    N = T-L+1
    if L>N:
        raise ValueError(f'***  The window length must be less than T/2. Currently  L = {L}, T = {T}.  ***');
    return x,T,N        
            
       
###############################################################################
###############################################################################
def run_cissa(x: np.ndarray,
              L: int,
              extension_type: str = 'AR_LR',
              multi_thread_run: bool = True,
              generate_toeplitz_matrix: bool = False
              ) -> tuple[np.ndarray,np.ndarray]:
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
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series 
    L : int
        DESCRIPTION: CiSSA window length.
    multi_thread_run : bool, optional
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.. The default is True.
    generate_toeplitz_matrix : bool, optional
        DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False. The default is False.

    Returns
    -------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results.
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density

    '''
    
    #1. run input checks
    x,T,N = cissa_input_checks(x,extension_type,L)
    
    #2. Extend series 
    from pycissa.utilities.extendseries import extend_series
    left_ext,right_ext = define_left_and_right_extension_lengths(extension_type,T,L)
    x_e = extend_series(x,extension_type,left_ext,right_ext)
    
    #3. Calcvulate trajectory matrix
    X = create_trajactory_matrix(x_e,L)
    
    #4. Calculate elementary matrix 
    U = calculate_elementary_matrix(L)
    
    #5. calculate circulant matirx
    S,C = create_toeplitz_circulant_matrices(x,L,T,generate_toeplitz_matrix=generate_toeplitz_matrix,                                           )
    
    #6. estimate_power_spectral_density 
    psd = estimate_power_spectral_density(U,C)
    psd = psd.reshape(len(psd),1)
    del C
    
    #7. Calculate principal components
    W = calculate_principal_components(U,X)
    del X
    
    #8.  Perform reconstruction
    R = perform_reconstruction(U,W,T,L,extension_type,multi_thread_run = multi_thread_run)
    del U, W
    
    #9. Group paired frequencies
    Z = group_paired_frequencies(R,L,T,extension_type)
    
    #10 generate results dictionary
    
    
    return Z,psd


       
###############################################################################
###############################################################################
def run_cissa_psd_step(x: np.ndarray,
              L: int,
              extension_type: str = 'AR_LR',
              multi_thread_run: bool = True,
              generate_toeplitz_matrix: bool = False
              ) -> np.ndarray:
    '''
    Function to get an approximation to the the psd using CiSSA.
    Same as function run_cissa, but does not do the diagonalisation (hence is much quicker).
    Is used in the Monte Carlo process to generate a psd.
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
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series 
    L : int
        DESCRIPTION: CiSSA window length.
    multi_thread_run : bool, optional
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.. The default is True.
    generate_toeplitz_matrix : bool, optional
        DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False. The default is False.

    Returns
    -------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results.
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density

    '''
    
    #1. run input checks
    x,T,N = cissa_input_checks(x,extension_type,L)
    
    #2. Extend series 
    from pycissa.utilities.extendseries import extend_series
    left_ext,right_ext = define_left_and_right_extension_lengths(extension_type,T,L)
    x_e = extend_series(x,extension_type,left_ext,right_ext)
    
    #3. Calcvulate trajectory matrix
    X = create_trajactory_matrix(x_e,L)
    
    #4. Calculate elementary matrix 
    U = calculate_elementary_matrix(L)
    
    #5. calculate circulant matirx
    S,C = create_toeplitz_circulant_matrices(x,L,T,generate_toeplitz_matrix=generate_toeplitz_matrix,                                           )
    
    #6. estimate_power_spectral_density 
    psd = estimate_power_spectral_density(U,C)
    psd = psd.reshape(len(psd),1)
    
    
    return psd