
import numpy as np
from scipy.stats import binned_statistic
import astropy.units as u
import astropy.constants as const

DEFAULT_OMEGA = 0.301712
DEFAULT_H = 0.6909
DEFAULT_NU = 1.12

def alpha(mwdm, omega=DEFAULT_OMEGA, h=DEFAULT_H):
    return 0.049 * mwdm**(-1.11) * (omega / 0.25)**0.11 * (h / 0.7)**1.22

def k_hm(mwdm, omega=DEFAULT_OMEGA, h=DEFAULT_H, nu=DEFAULT_NU):
    return 1. / alpha(mwdm, omega, h) * (2**(nu/5) - 1)**(1 / 2 / nu)

def lambda_hm(mwdm, omega=DEFAULT_OMEGA, h=DEFAULT_H, nu=DEFAULT_NU):
    return 2 * np.pi / k_hm(mwdm, omega, h, nu)

def M_hm(mwdm, omega=DEFAULT_OMEGA, h=DEFAULT_H, nu=DEFAULT_NU):
    H0 = 100 * u.km / u.s / u.Mpc
    rho_crit = (3 * H0**2 / (8 * np.pi * const.G)).to_value(u.Msun / u.Mpc**3)
    rho_bar =  omega * rho_crit
    return (4 * np.pi / 3) * rho_bar * (lambda_hm(mwdm, omega, h, nu) / 2)**3

def get_bins(indices, select):
    """ Return the bin mask for a given selection. """
    mask = np.ones(indices[0].shape, dtype=bool)
    for i, s in zip(indices, select):
        if s is not None:
            mask &= i == s
    return mask

def get_average_distribution(samples, bins, cumulative=False, normalized=False):
    """ Compute the average distribution of a set of samples. """
    c = [np.histogram(samples[i], bins=bins)[0] for i in range(samples.shape[0])]
    c = np.stack(c, axis=0)
    if normalized:
        c = c / c.sum(1, keepdims=True)
    if int(cumulative) == 1:
        c = np.cumsum(c, axis=1)
    elif int(cumulative) == -1:
        c = np.sum(c, axis=1, keepdims=True) - np.cumsum(c, axis=1)
    c_mean = np.nanmean(c, axis=0)
    c_stdv = np.nanstd(c, axis=0)
    return c_mean, c_stdv

def get_binned_stats(x_samples, y_samples, bins, statistic='mean'):
    """ Compute binned statistics for a set of samples and take the mean. """
    y_binned_stats = []
    for (x, y) in zip(x_samples, y_samples):
        select = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(select) == 0:
            continue
        y_stats = binned_statistic(
            x[select], y[select], bins=bins, statistic=statistic)[0]
        y_binned_stats.append(y_stats)
    y_binned_stats = np.stack(y_binned_stats, axis=0)
    y_mean = np.nanmean(y_binned_stats, axis=0)
    y_stdv = np.nanstd(y_binned_stats, axis=0)
    return y_mean, y_stdv
