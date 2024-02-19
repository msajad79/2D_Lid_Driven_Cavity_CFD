import numpy as np

def thomas_method(a, b, c, D):
    beta = [b[0]]
    for i in range(1,len(a)):
        beta.append(b[i]-a[i]*c[i-1]/beta[i-1])

    gama = [D[0]/b[0]]
    for i in range(1, len(a)):
        gama.append((D[i] - a[i]*gama[i-1]) / beta[i])

    y = [gama[-1]]
    for i in range(len(a)-2, -1, -1):
        y.insert(0, gama[i]-c[i]*y[0]/beta[i])
    return y

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a[1:], k1) + np.diag(b, k2) + np.diag(c[:-1], k3)

def f_harmonic(x):
    return (np.sin(x*np.pi-np.pi/2.0) + 1.0) / 2.0

def f_linear(x):
    return x

def f_polynomial(x, b=.2):
    return (1-b)*x**2.0 + b*x

def f_inv_polynomial(x):
    return np.sqrt(x)

def f_polynomial_cubic(x, c=1.5):
    return (2*c-2)*x**3.0+(3*(1-c))*x**2.0+c*x

def f_c(x):
    if x < 0.75:
        return (np.sin(4.0/3.0*x*np.pi-np.pi/2.0) + 1.0) / 4.0
    return (np.sin(4.0*x*np.pi-3.0*np.pi/2.0) + 3.0) / 4.0

def f_c2(x, b=0.5):
    if x < .5:
        return 1.
    return 

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



INFO_DEFAULT_PSI_ZERO   = {
    -1: 0.0,
    1 : 0.0,
    2 : 0.0,
    3 : 0.0,
    4 : 0.0,
    5 : 0.0,
    6 : 0.0,
    7 : 0.0,
    8 : 0.0,
    9 : 0.0,
    10: 0.0,
    11: 0.0,
    12: 0.0,
}
INFO_DEFAULT_W_ZERO = {
    -1: 0.0,
    1 : 0.0,
    2 : 0.0,
    3 : 0.0,
    4 : 0.0,
    5 : 0.0,
    6 : 0.0,
    7 : 0.0,
    8 : 0.0,
    9 : 0.0,
    10: 0.0,
    11: 0.0,
    12: 0.0,
}
