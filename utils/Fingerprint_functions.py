"""
Library for generation of diffusional fingerprints

Henrik Dahl Pinholt
"""
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from probfit import Chi2Regression
from iminuit import Minuit
import inspect
import scipy.stats as stats

def Chi2Fit(
    x,
    y,
    sy,
    f,
    plot=True,
    print_level=0,
    labels=None,
    ax=None,
    savefig=None,
    valpos=None,
    exponential=False,
    fitcol=None,
    markersize=5,
    plotcol=None,
    name=None,
    fontsize=15,
    linewidth=3,
    png=False,
    custom_cost=None,
    **guesses,
):
    """Function that peforms a Chi2Fit to data given function
    ----------
    Parameters
    ----------
    x: ndarray of shape for input in f
        - input values to fit
    y: ndarray of shape output from f
        - output values to fit
    sy: ndarray of length y
        - errors on the y values
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    name: str
        -Label to call fit in legend
    fontsize: int
        - Size of font in plot
    linewidth: float
        - Width of line on data
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    Ndof: int
        - Number of  degrees of freedom for fit
    Chi2: float
        - Chi2 for fit
    pval: float
        -pvalue for the fit
    """
    xmin, xmax = np.min(x), np.max(x)
    names = inspect.getfullargspec(f)[0][1:]
    if custom_cost is None:
        chi2_object = Chi2Regression(f, x, y, sy)
    else:
        chi2_object = custom_cost
    if len(guesses) != 0:
        paramguesses = {}
        lims = {}
        for key, value in guesses.items():
            if key.split("_")[0] == "limit":
                lims[key.split("_")[1]] = value
            else:
                paramguesses[key] = value
        minuit = Minuit(chi2_object, **paramguesses)
        if len(lims) > 0:
            for key, value in lims.items():
                minuit.limits[key] = value
        minuit.print_level = print_level
    else:
        minuit = Minuit(chi2_object)
    minuit.errordef = 1
    minuit.migrad()
    chi2 = minuit.fval
    Ndof = len(x) - len(guesses)
    Pval = stats.chi2.sf(chi2, Ndof)
    params = minuit.values
    errs = minuit.errors

    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.4E}", "Ndof": f"{Ndof:4.4E}", "Pval": f"{Pval:4.4E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4E} +/- {py:4.4E}"
    return params, errs, Pval


def SquareDist(x0, x1, y0, y1, z0, z1):
    """Computes the squared distance between the two points (x0,y0) and (y1,y1)

    Returns
    -------
    float
        squared distance between the two input points

    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def QuadDist(x0, x1, y0, y1, z0, z1):
    """Computes the four-norm (x1-x0)**4+(y1-y0)**4+(z1-z0)**4.

    Returns
    -------
    float
        Four-norm.

    """
    return (x1 - x0) ** 4 + (y1 - y0) ** 4 + (z1 - z0) ** 4


def GetMax(x, y, z):
    """Computes the maximum squared distance between all points in the (x,y,z) set.

    Parameters
    ----------
    x : list-like
        x-coordinates.
    y : list-like
        y-coordinates.
    z : list-like
        z-coordinates.

    Returns
    -------
    float
        Largest squared distance between any two points in the set.

    """
    from itertools import combinations
    from random import randint

    A = np.array([x, y, z]).T

    def square_distance(x, y):
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])

    max_square_distance = 0
    for pair in combinations(A, 2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
            max_pair = pair
    return max_square_distance


def msd(x, y, z, frac):
    """Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    frac*len(x) of the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    frac : float in [0,1]
        Fraction of trajectory duration to compute msd up to.
        if length of x is more than 20 else collapse to len(x)

    Returns
    -------
    iterable of lenght int(len(x)*frac)
        msd for the trajectory

    """
    N = int(len(x) * frac) if len(x)>20 else len(x)
    msd = []
    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def Scalings(msds, dt, dim=2, difftype='Normal'):
    """Fit mean squared displacements to a power law.

    Parameters
    ----------
    msds : list-like
        mean squared displacenemts.

    Returns
    -------
    tuple of length 3
        The first index is the fitted generalized diffusion constant,
        the second is the scaling exponent alpha, and the final is the pvalue for the fit.

    """
    if difftype=='Normal' or difftype=='Subdiffusive':
        def power(x, D, alpha):
            return 2 * dim * D * (x) ** alpha

        params, errs, Pval = Chi2Fit(
        np.arange(1, len(msds) + 1)*dt,
        msds,
        1e-10 * np.ones(len(msds)),
        power,
        plot=False,
        D=1,
        alpha=1,
        limit_alpha=(-10, 10),
        )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1), *params))
        if sy>0:    
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=1,
                alpha=1,
                limit_alpha=(-10, 10),
            )
    elif difftype=='Directed':
        def power(x, D, alpha, v):
            return 2 * dim * D * (x) ** alpha + (v**2) * (x**2)

        params, errs, Pval = Chi2Fit(
            np.arange(1, len(msds) + 1)*dt,
            msds,
            1e-10 * np.ones(len(msds)),
            power,
            plot=False,
            D=1,
            alpha=1,
            v=1,
            limit_alpha=(-10, 10),
            limit_v=(0, 10)
            )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1), *params))
        if sy>0: 
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=1,
                alpha=1,
                v=1,
                limit_alpha=(-10, 10),
                limit_v=(0, 10)
            )
    elif difftype=='Confined':
        def power(x, D, alpha, R, A1, A2):
            return (R**2) * (1-A1*np.exp(-2 * A2 * dim * D * (x) / (R**2)))

        params, errs, Pval = Chi2Fit(
            np.arange(1, len(msds) + 1)*dt,
            msds,
            1e-10 * np.ones(len(msds)),
            power,
            plot=False,
            D=1,
            alpha=1,
            R=1,
            A1=1, 
            A2=1,
            limit_alpha=(-10, 10)
            )
        sy = np.std(msds - power(np.arange(1, len(msds) + 1), *params))
        if sy>0: 
            params, errs, Pval = Chi2Fit(
                np.arange(1, len(msds) + 1)*dt,
                msds,
                sy * np.ones(len(msds)),
                power,
                plot=False,
                D=1,
                alpha=1,
                R=1,
                A1=1, 
                A2=1,
                limit_alpha=(-10, 10)
            )
    return params, Pval


def Efficiency(x, y, z):
    """Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
    and the sum of squared distances.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.

    Returns
    -------
    float
        Efficiency.

    """
    top = SquareDist(x[0], x[-1], y[0], y[-1], z[0], z[-1])
    bottom = sum(
        [SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]) for i in range(0, len(x) - 1)]
    )
    return np.log((top) / ((len(x) - 1) * bottom)), (top) / ((len(x) - 1) * bottom)


def FractalDim(x, y, z, max_square_distance):
    """Computes the fractal dimension using the estimator suggested by Katz & George
    in Fractals and the analysis of growth paths, 1985.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.
    max_square_distance : float
        Maximum squared pair-wise distance for the poinst in the trajectory.

    Returns
    -------
    float
        Estimated fractal dimension.

    """
    totlen = sum(
        [
            np.sqrt(SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]))
            for i in range(0, len(x) - 1)
        ]
    )
    return np.log(len(x)) / (
        np.log(len(x)) + np.log(np.sqrt(max_square_distance) / totlen)
    )


def Gaussianity(x, y, z, r2):
    """Computes the Gaussianity.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.
    r2 : list-like
        Mean squared displacements for the trajectory.

    Returns
    -------
    float
        Gaussianity.

    """
    gn = []
    for lag in range(1, len(r2)):
        r4 = np.mean(
            [QuadDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag]) for j in range(len(x) - lag)]
        )
        gn.append(r4 / (2 * r2[lag] ** 2))
    return np.mean(gn)


def Kurtosis(x, y, z, dim=2):
    """Computes the kurtosis for the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    z : list-like
        z-coordinates for the trajectory.

    Returns
    -------
    float
        Kurtosis.

    """
    from scipy.stats import kurtosis

    if dim == 3:
        # https://en.wikipedia.org/wiki/Covariance_matrix
        # np.cov(x, y)[0,1] to get covariance of x and y in covariance matrix
        C = np.array([[np.cov(x, x)[0,1],np.cov(x, y)[0,1],np.cov(x, z)[0,1]],
                      [np.cov(y, x)[0,1],np.cov(y, y)[0,1],np.cov(y, z)[0,1]],
                      [np.cov(z, x)[0,1],np.cov(z, y)[0,1],np.cov(z, z)[0,1]]]) 
        val, vec = np.linalg.eig(C)
        dominant = vec[:, np.argsort(val)][:, -1]
        kurt = kurtosis([np.dot(dominant, v) for v in np.array([x, y, z]).T], fisher=False)
    else:
        val, vec = np.linalg.eig(np.cov(x, y))
        dominant = vec[:, np.argsort(val)][:, -1]
        kurt = kurtosis([np.dot(dominant, v) for v in np.array([x, y]).T], fisher=False)
    return kurt


def MSDratio(mval):
    """Computes the MSD ratio.

    Parameters
    ----------
    mval : list-like
        Mean squared displacements.

    Returns
    -------
    float
        MSD ratio.

    """
    return np.mean(
        [mval[i] / mval[i + 1] - (i) / (i + 1) for i in range(len(mval) - 1)]
    )


def Trappedness(x, maxpair, out):
    """Computes the trappedness.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    maxpair : float
        Maximum squared pair-wise distance for the poinst in the trajectory.
    out : list-like
        Mean squared displacements.

    Returns
    -------
    float
        Trappedness.

    """
    r0 = np.sqrt(maxpair) / 2
    D = out[1] - out[0]
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1262511/pdf/biophysj00087-0118.pdf 
    # has 0.2048 which differs from diffusional fingerprinting PNAS
    return 1 - np.exp(0.2045 - 0.25117 * (D * len(x)) / r0 ** 2)


def Time_in(state):
    """Computes the fraction of time spent in each of four states in a state
    history.

    Parameters
    ----------
    state : list-like
        State history for the trajectory.

    Returns
    -------
    list of length 4
        Fraction of time spent in each state.

    """
    times = []
    N = len(state)
    for o in range(4):
        time = 0
        for s in state:
            if s == o:
                time += 1
        times.append(time)
    return np.array(times) / N


def Lifetime(state):
    """Computes the average duration of states.

    Parameters
    ----------
    state : list-like
        State history for the trajectory.

    Returns
    -------
    float
        average duration of a state

    """
    jumps = []
    for i in range(len(state) - 1):
        if state[i + 1] != state[i]:
            jumps.append(i)
    if len(jumps) == 1:
        return max(jumps[0], len(state) - jumps[0])
    if len(jumps) == 0:
        return len(state)
    else:
        lifetimes = np.array(jumps[1:]) - np.array(jumps[:-1])
        return np.mean(lifetimes)


def GetStates(SL, model):
    """Predict the viterbi path for a series of steplengths based on a fitted HMM model.

    Parameters
    ----------
    SL : list-like
        step lengths for the trajectory.
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    list-like
        State trajectories.
    pomegranate model
        The model used to predict the states

    """
    for i in range(len(SL)):
        if SL[i] == 0:
            SL[i] = 1e-15
    states = model.predict(SL, algorithm="viterbi")
    ms = [s.distribution.parameters[0] for s in model.states[:4]]
    statemap = dict(zip(np.arange(4)[np.argsort(ms)], np.arange(4)))
    newstates = [statemap[s] for s in states[1:]]
    return newstates, model


def dotproduct_traces(trace):
    vecs = trace[1:]-trace[:-1]
    dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
    return dots


def GetFeatures(x, y, z, SL, model, dim, dt, difftype):
    """Compute the diffusional fingerprint for a trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    SL : list-like
        step lengths for the trajectory.
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    ndarray
        The features describing the diffusional fingerprint

    """
    out = msd(x, y, z, 0.5)
    maxpair = GetMax(x, y, z)
    params, pval = Scalings(out, dt, dim=dim, difftype=difftype)
    if difftype == 'Normal' or difftype == 'Subdiffusive' or difftype == None:
        beta = params[0]
        alpha = params[1]
        extra = None
    if difftype == 'Directed':
        beta = params[0]
        alpha = params[1]
        extra = params[2]
    if difftype == 'Confined':
        beta = params[0]
        alpha = params[1]
        extra = params[2]

    states, model = GetStates(SL, model)

    t0, t1, t2, t3 = Time_in(states)
    lifetime = Lifetime(states)
    return np.array(
        [
            alpha,
            beta,
            extra,
            pval,
            Efficiency(x, y, z)[0],
            Efficiency(x, y, z)[1],
            FractalDim(x, y, z, maxpair),
            Gaussianity(x, y, z, out),
            Kurtosis(x, y, z, dim=dim),
            MSDratio(out),
            Trappedness(x, maxpair, out),
            t0,
            t1,
            t2,
            t3,
            lifetime,
            len(x),
            np.mean(SL),
            np.mean(out),
            np.mean(dotproduct_traces(np.array([x,y,z]).T))
        ]
    )

def ThirdAppender(d, model, dim, dt, difftype):
    """Wrapper function around GetFeatures.

    Parameters
    ----------
    d : tuple of length 3
        (x,y,SL).
    model : pomegranate model
        Fitted pomegranate model used to compute the viterbi path.

    Returns
    -------
    ndarray or str
        Returns the features describing the diffusional fingerprint
    """
    if dim==2:
        x, y, SL = d
        z = np.zeros_like(x)
    if dim==3:
        x, y, z, SL = d

    return GetFeatures(x, y, z, SL, model, dim, dt, difftype)
