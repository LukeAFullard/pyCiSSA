###############################################################################
###############################################################################
###############################################################################
def build_groupings(period_ranges,data_per_unit_period,psd,z,include_noise = True):
    
    
    import numpy as np
    ###########################################################################
    
    L = len(psd)
    T, F = z.shape
    # Proportionality of L
    if np.mod(L,data_per_unit_period):
        raise ValueError(f'***  L is not proportional to the number of data per unit period  (modulo of L/data_per_unit_period = {np.mod(L,data_per_unit_period)}) ***');
    
    # Number of groups
    G = len(period_ranges)-1
    if include_noise:
        G+=1
    
    
    # Number of data per year
    s = data_per_unit_period
    # Inizialitation of empty dict
    kg = {}
    min_k = 1_000_000_000 
    for key_p,value_p in period_ranges.items():
        myarray = None
        if value_p[0] == value_p[1]:
            myarray = L*np.arange(1,s/2)/(value_p[0]*s)
            kg.update({key_p  :  myarray  })
            min_k = min(min_k,min(myarray))
        else:
            myarray = np.arange(max(2,np.floor(L/(value_p[1]*s)+1))-1,min(F,np.floor(L/(value_p[0]*s)+1)),dtype=int)
            kg.update({key_p  :  myarray  })
            min_k = min(min_k,min(myarray))
    
    #append trend
    kg.update({'trend': np.arange(0,int(min_k))})

    #Noise: The left over frequencies
    if include_noise:
        current_k = []
        for index_j in kg.values():
            current_k = current_k + [int(x) for x in index_j]
        missing_k = [x for x in range(0,int(np.floor(L/2))) if x not in current_k]
        # kg.update({4: np.array(missing_k)})
        kg.update({'noise': np.array(missing_k)})
    
    return kg

###############################################################################
###############################################################################
###############################################################################
def diagaver(Y):
    '''
     DIAGAVER - Diagonal averaging for Singular Spectrum Analysis. https://doi.org/10.1016/j.sigpro.2020.107824
    
     This function transforms the numpy matrix, Y, into the time series,
     y, by diagonal averaging. This entails averaging the elements of Y over
     its antidiagonals.
     
     Syntax:     y = diagaver(Y)
    
    Conversion from Matlab, https://github.com/jbogalo/CiSSA

    Parameters
    ----------
    Y : numpy 2D array
        DESCRIPTION: Input 2D numpy array/matrix

    Returns
    -------
    y : numpy 1D array
        DESCRIPTION: Output diagonally averaged 1D array
        
    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., and Senra, E. "Circulant Singular Spectrum
        Analysis: A new automated procedure for signal extraction". Signal
        Processing. Vol. 179, 2021, in progress.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    -------------------------------------------------------------------------    

    '''
    
    import numpy as np
    
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
###############################################################################
def extend(x,H):
    '''
     EXTEND - Extends time series to perform Singular Spectrum Analysis.  https://doi.org/10.1016/j.sigpro.2020.107824
    
     This function extends the time series at the beginning and end. 
    
     Syntax:       xe = extend(x,H)
     
     Conversion from Matlab, https://github.com/jbogalo/CiSSA
    
    Parameters
    ----------
    x : numpy array
        DESCRIPTION: Column vector with the original time series. Must be size (N,1) where N is the length of the vector.
    H : int
        DESCRIPTION: A number which determines the extension type.

    Returns
    -------
    xe : numpy array
        DESCRIPTION: The extended time series.
        
    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., and Senra, E. "Circulant Singular Spectrum
        Analysis: A new automated procedure for signal extraction". Signal
        Processing. Vol. 179, 2021, in progress.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    -------------------------------------------------------------------------     

    '''
    import numpy as np
    import statsmodels.api as sm
    from spectrum import aryule
    from scipy.signal import lfilter
    
    
    ###########################################################################
    # 0) x size checking, H type checking
    ###########################################################################
    #check H is an integer
    if not type(H) == int:
        raise('Input parameter "H" should be an integer')
    
    rows, cols = x.shape
    if cols != 1:    
        raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({rows},{cols})')
    ###########################################################################
    
    
    ###########################################################################
    # 1) Dimensions
    ###########################################################################
    T = len(x)
    ###########################################################################
    
     
    ###########################################################################
    # 2) Extend
    ###########################################################################
    if H == 0:    #No extension
        xe = x.copy()
    elif H == T:  #Mirroring
        xe = np.append(np.append(np.flipud(x),x),np.flipud(x))
        xe = xe.reshape(len(xe),1)
    else:         #Autoregressive extension  
        # AR coefficients of the differentiated series
        p = np.fix(T/3)
        dx = np.diff(x, axis = 0)       
        Aold, cccold = sm.regression.yule_walker(dx, order=int(p),method="adjusted")
        [A, var, reflec] = aryule(dx, int(p))
        # Right extension
        y = x.copy()
        dy = np.diff(y, axis = 0)
        # er = lfilter(A, [1], dy)
        er = lfilter(np.append(1,A), 1, [x[0] for x in dy])
        # er = er.reshape(len(er),1)
        
        dy = lfilter([1],np.append(1,A),np.append(er,np.zeros((H,1))))
        y = y[0]+np.append(0,np.cumsum(dy))
        # Left extension
        y = np.flipud(y)
        dy = np.diff(y, axis = 0)
        er = lfilter(np.append(1,A), 1, dy)
        dy = lfilter([1],np.append(1,A),np.append(er,np.zeros((H,1))))
        y = y[0]+np.append(0,np.cumsum(dy))
        # Extended series
        xe = np.flipud(y)
        xe = xe.reshape(len(xe),1)
    ###########################################################################
    
    return xe




###############################################################################
###############################################################################
###############################################################################
def group(Z,psd,I,season_length = 1, cycle_length = [1.5,8], include_noise = True):
    '''
    GROUP - Grouping step of CiSSA.  https://doi.org/10.1016/j.sigpro.2020.107824.
   
    This function groups the reconstructed components by frequency
    obtained with CiSSA into disjoint subsets and computes the share of the
    corresponding PSD.
   
    Syntax:     [rc, sh, kg] = group(Z,psd,I)
    
    Conversion from Matlab, https://github.com/jbogalo/CiSSA


    Parameters
    ----------
    Z : numpy array/matrix
        DESCRIPTION: Matrix whose columns are the reconstructed components by frequency obtained with CiSSA.
    psd : numpy column vector
        DESCRIPTION: Column vector with the estimated power spectral density at frequencies w(k)=(k-1)/L, k=1,2,...,L, obtained with CiSSA.
    I : multiple
        DESCRIPTION: 
             Four options:
             1) A positive integer. It is the number of data per year in
             time series. The function automatically computes the
             trend (oscillations with period greater than 8 years), the
             business cycle (oscillations with period between 8 & 1.5 years)
             and seasonality.
             2) A dictionary. Each value contains a numpy row vector with the desired
             values of k to be included in a group, k=1,2,...,L/2. The function
             computes the reconstructed components for these groups.
             3) A number between 0 & 1. This number represents the accumulated
             share of the psd achieved with the sum of the share associated to
             the largest eigenvalues. The function computes the original
             reconstructed time series as the sum of these components.
             4) A number between -1 & 0. It is a percentile (in positive) of
             the psd. The function computes the original reconstructed time
             series as the sum of the reconstructed componentes by frequency
             whose psd is greater that this percentile.
    season_length : int, optional
        DESCRIPTION: The default is 1. How many data points per season period. If season is annual, season_length is number of data points in a year. Only used for case 1.
    cycle_length : list, optional
        DESCRIPTION: The default is [1.5,8]. List of longer term cycle periods. Only used for case 1.
    include_noise : bool, optional
        DESCRIPTION: The default is True. Output noise as a vector component or not. Only used for case 1. 


    Returns
    -------
    rc : numpy array
        DESCRIPTION: Matrix whose columns are the reconstructed components for each
             group or subset of frequencies. In the case of economic time series
             the trend, business cycle and seasonality are in the first, second
             and third columns, respectively.
    sh : numpy array
        DESCRIPTION: Column vector with the share() of the psd for each group.
    kg : dict
        DESCRIPTION: Dictionary where each entry contains a row vector with the values
             of k belonging to a group. Option 1) produces 3 groups, option 2)
             gives the goups introduced by the user and options 3) and 4) produce
             a single group. In option 3), the values of k are sorted according
             to the share in total psd of their corresponding eigenvalues.

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., and Senra, E. "Circulant Singular Spectrum
        Analysis: A new automated procedure for signal extraction". Signal
        Processing. Vol. 179, 2021, in progress.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    -------------------------------------------------------------------------

    '''
    
    import numpy as np
    
    ###########################################################################
    # 0) psd size checking
    ###########################################################################    
    rows, cols = psd.shape
    if cols != 1:    
        raise ValueError(f'Input "psd" should be a column vector (i.e. only contain a single column). The size of x is ({rows},{cols})')
    ###########################################################################
    
    
    ###########################################################################
    # 1) Checking the input arguments
    ###########################################################################
    # Length and number of the reconstruted series
    T, F = Z.shape
    
    # Window length
    L = len(psd)
    
    
    # Type and value of input argument #3
    if type(I) is dict:
        opc = 2;
    elif type(I) == int or type(I) == float:
        if ((I-np.floor(I))==0) & (I>0):
            opc = 1
        elif (0<I)&(I<1):
            opc = 3
        elif (-1<I)&(I<0):
            opc = 4
        else:
            raise ValueError('***  Input argument #3 (I): Value ({I}) not valid  ***')
        
    else:
        raise ValueError(f'***  Input argument #3 ({I}): Type ({type(I)}) not valid  ***')
    
    
    #NOTE, Matlab version uses switch statements, but this was only implemented in Python in 3.10, so here we use if else statements for compatibility with older Python versions. Definately could improve this in the future...
    if opc == 1:
        # Proportionality of L
        if np.mod(L,I):
            raise ValueError(f'***  L is not proportional to the number of data per year  (modulo of L/I = {np.mod(L,I)}) ***');
        
    elif opc == 2:
        # Number of groups
        G = len(I.keys())
        if G>F:
            raise ValueError(f'***  The number of groups ({G}) is greater than the number of frequencies ({F})  ***')
        
        key_names = list(I.keys())     
        # Disjoint groups
        for j in range(1,G):
            for m in range(j+1,G+1):
                if len(np.intersect1d(I[key_names[j-1]],I[key_names[m-1]]))>0:
                    raise ValueError(f'''***  The groups are not disjoint  ***
                                     {key_names[j-1]}: {I[key_names[j-1]]}
                                     {key_names[m-1]}: {I[key_names[m-1]]}                        ''');
               

    ###########################################################################
    
    
    ###########################################################################
    # 2) PSD for frequencies <= 1/2
    ###########################################################################    
    if np.mod(L,2):
        pzz = np.append(psd[0], 2*psd[1:F-1]) 
    else:
        pzz = np.append(
            np.append(psd[0], 2*psd[1:F-2]),
            psd[F-1]
            )

    ###########################################################################
    
    
    ###########################################################################
    # 3) Indexes k for each group
    ###########################################################################    
    #NOTE, Matlab version uses switch statements, but this was only implemented in Python in 3.10, so here we use if else statements for compatibility with older Python versions. Definately could improve this in the future...
    if opc == 1:
        # Number of groups
        G = 3
        if include_noise:
            G=4

        # Number of data per year
        s = I
        # Inizialitation of empty dict
        kg = {}
        # Seasonality    season_length cycle_length
        kg.update({'seasonality': L*np.arange(1,s/2)/(season_length*s)})

        # Long term cycle
        kg.update({'long term cycle': np.arange(max(2,np.floor(L/(cycle_length[1]*s)+1))-1,min(F,np.floor(L/(cycle_length[0]*s)+1)),dtype=int)})
        # Trend
        kg.update({'trend': np.arange(0,kg['long term cycle'][0])})

        #Noise: The left over frequencies
        if include_noise:
            current_k = []
            for index_j in kg.values():
                current_k = current_k + [int(x) for x in index_j]
            missing_k = [x for x in range(0,int(np.floor(L/2))) if x not in current_k]
            kg.update({'noise': np.array(missing_k)})
        
        
    elif opc == 2:    
        # Groups
        kg = I.copy()
    elif opc == 3:
        
        # Number of groups
        G = 1
        # Inizialitation of list array
        kg = {}
        # Eigenvalues in decreasing order
        psor=np.sort(pzz)[::-1]
        ks=np.argsort(-pzz)
        # Cumulative share in percentage
        pcum = 100*np.cumsum(psor)/sum(psd)
        # Group for the reconstructed time series
        kg.update({1: ks[np.arange(1,len(ks[pcum<100*I])+1)]})
    elif opc == 4:
        # Number of groups
        G = 1
        # Inizialitation of cell array
        kg = {}
        # All k values
        ks = np.arange(0,F-1)
        # Group for the reconstructed time series
        
        kg.update({1:  ks[pzz>np.percentile(pzz,-100*I)]    })
    else:
        raise ValueError(f'*** Value of opc ({opc}) is incorrect. Something has gone terribly wrong... ***')
    ###########################################################################
    
    
    ###########################################################################
    # 4) Output arguments
    ###########################################################################    
    # Inizialitation of output arguments
    rc = np.zeros((T,G))
    sh = np.zeros((G,1))
    

    # # Computing output arguments
    rc = {}
    sh = {}
    for index_j,key_j in enumerate(kg.keys()):
        # % Reconstructed component for each group
        indx=[int(x) for x in kg[key_j]]
        rc.update({key_j:np.sum(Z[:,indx],axis = 1, keepdims = True)})
        
        # % Psd share for each group
        sh.update({key_j: 100*np.sum(pzz[indx])/np.sum(pzz)})
     

    return rc, sh, kg




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
def cissa(x,L,H = 0):
    '''
     CiSSA - Circulant Singular Spectrum Analysis. https://doi.org/10.1016/j.sigpro.2020.107824
    
     This function returns the elementary reconstructed series by
     frequency, Z, and the estimated power spectral density, psd, of the
     input time series, x, using Circulant Singular Spectrum Analysis
     given a window length, L.
    
     Syntax:     Z, psd = cissa(x,L)
                 Z, psd = cissa(x,L,H)
                 
    Conversion from Matlab, https://github.com/jbogalo/CiSSA
   
    
     Input arguments:
     x: Columm vector containing the time series original data.
     L: Window length.
     H: Optional. Related to the characteristics of the time series.
        H=0 Autoregressive extension (default). It is indicated for stationary
            and stochastic trend time series as well.
        H=1 Mirroring. It can be used with stationary time series and works
            well for AM-FM signals.
        H=2 No extension. It is suitable for deterministic time series.
    
     Output arguments:
     Z:   Matrix whose columns are the reconstructed components by frequency.
     psd: Column vector with the estimated power spectral density at
          frequencies w(k)=(k-1)/L, k=1,2,...,L. This is, the eigenvalues of
          the circulant matrix of second moments.
    
     See also: group
    
     -------------------------------------------------------------------------
     References:
     [1] Bógalo, J., Poncela, P., and Senra, E. "Circulant Singular Spectrum
         Analysis: A new automated procedure for signal extraction". Signal
         Processing. Vol. 179, 2021, in progress.
         https://doi.org/10.1016/j.sigpro.2020.107824.
     -------------------------------------------------------------------------

    '''
    import numpy as np
    from scipy.linalg import hankel,dft
    
    ###########################################################################
    # 0)H,L type checking
    ###########################################################################    
    #check H,L is an integer
    if not type(H) == int:
        raise('Input parameter "H" should be an integer')
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
   
        
    ###########################################################################
    
    
    ###########################################################################
    # 1) Checking the input arguments
    ###########################################################################    
    # Dimensions
    if rows==1: #we want a column vector
        x = x.transpose()

    T = len(x)
    N = T-L+1
    if L>N:
        raise ValueError(f'***  The window length must be less than T/2. Currently  L = {L}, T = {T}.  ***');
    

    #Type of extension depending on H
    #NOTE, Matlab version uses switch statements, but this was only implemented in Python in 3.10, so here we use if else statements for compatibility with older Python versions. Definately could improve this in the future...
    if H == 1:
        H = T
    elif H == 2:
        H = 0
    else:
        H = L
        

    #Number of symmetryc frequency pairs around 1/2
    if np.mod(L,2):
        nf2 = (L+1)/2-1
    else:
        nf2 = L/2-1
    

    #Number of frequencies <= 1/2
    nft = nf2+np.abs(np.mod(L,2)-2)

    ###########################################################################
    
    
    ###########################################################################
    # 2) Trajectory matrix
    ###########################################################################    
    #Extended series
    xe = extend(x,H)

    #Trajectory matrix
    col = xe[0:L]
    row = xe[L-1:]
    X = hankel(col,row);
    
    del col, row
    ###########################################################################
    
    
    ###########################################################################
    # 3) Decomposition
    ###########################################################################    
    
    
    # Autocovariance function
    gam = np.zeros((L,1))
    for k in range(0,L):
        gam[k] = np.matmul((x[0:T-k]-np.mean(x)).transpose(),(x[k:T+1]-np.mean(x))/(T-k))
    
       
    

    #Symmetric Toeplitz covariance matrix S and equivalent circulant matrix C
    S = gam[0]*np.eye(L) 
    C = S.copy()
    for i in range(0,L):
        for j in range(i+1,L):
            k = np.abs(i-j)
            S[i,j] = gam[k] 
            S[j,i] = S[i,j]
            C[i,j] = ((L-k)/L)*gam[k]+(k/L)*gam[L-k] # Pearl (1973)
            C[j,i] = C[i,j];
    del gam
    
    

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
      
    if not np.mod(L,2):
        U[:,int(nft-1)] = np.real(U[:,int(nft-1)]);
    

    

    # #Eigenvalues of circulant matrix: estimated power spectral density
    psd = np.abs(np.diag(np.matmul(U.transpose(),np.matmul(C,U))))

    # #Principal components
    W = np.matmul(U.transpose(),X)


    # ###########################################################################
    
    
    # ###########################################################################
    # # 4) Reconstruction
    # ###########################################################################    
    # Elementary reconstructed series
    R = np.zeros((T+2*H,L))
    for k in range(0,L):
        R[:,[k]] = diagaver(np.matmul(U[:,[k]],W[[k],:]))
    # ###########################################################################
    
    
    # ###########################################################################
    # # 5) Grouping by frequency
    # ###########################################################################    
    # Elementary reconstructed series by frequency
    Z = np.zeros((T+2*H,int(nft)))
    Z[:,[0]] = R[:,[0]]
    for k in range(1,int(nf2)+1):
        Z[:,[k]] = R[:,[k]]+R[:,[L+2-(k+1)-1]];

    if not np.mod(L,2):
        Z[:,int(nft-1)] = R[:,int(nft-1)]

    lcol,lrow = Z.shape
    Z = Z[H:lcol-H,:]

    # ###########################################################################
    # reshape psd
    psd = psd.reshape(len(psd),1)

    return Z, psd

###############################################################################
###############################################################################
###############################################################################
