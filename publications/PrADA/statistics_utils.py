import numpy
from scipy.stats import kde


def ks_statistic(data1, data2):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    The empirical cumulative distribution function for each set of
    set of 1-dimensional data points is calculated. The K-S statistic
    the maximum absolute distance between the two cumulative distribution
    functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
    """
    x1 = numpy.sort(data1)
    x2 = numpy.sort(data2)
    x = numpy.sort(numpy.concatenate([x1, x2]))
    y1 = numpy.linspace(0, 1, len(x1) + 1)[
         1:]  # empirical CDF for data1: curve going up by 1/len(data1) at each observed data-point
    y2 = numpy.linspace(0, 1, len(x2) + 1)[1:]  # as above but for data2
    cdf1 = numpy.interp(x, x1, y1, left=0)  # linearly interpolate both CDFs onto a common set of x-values.
    cdf2 = numpy.interp(x, x2, y2, left=0)
    return abs(cdf1 - cdf2).max()


def ks_statistic_kde(data1, data2, num_points=None):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the K-S statistic is
    calculated: the maximum absolute distance between the two cumulative
    distribution functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points)
    with numpy.errstate(under='ignore'):
        cdf1 = numpy.array([kd1.integrate_box_1d(-numpy.inf, x) for x in xs])
        cdf2 = numpy.array([kd2.integrate_box_1d(-numpy.inf, x) for x in xs])
    return abs(cdf1 - cdf2).max()


def js_metric(data1, data2, num_points=None):
    """Calculate the Jensen-Shannon metric to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the J-S metric (square root of
    J-S divergence) is calculated.
    Note: KDE will often underestimate the probability at the far tails of the
    distribution (outside of where supported by the data), which can lead to
    overestimates of K-L divergence (and hence J-S divergence) for highly
    non-overlapping datasets.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, p1, p2 = _get_point_estimates(data1, data2, num_points)
    m = (p1 + p2) / 2
    return ((_kl_divergence(xs, p1, m) + _kl_divergence(xs, p2, m)) / 2) ** 0.5


def kl_divergence(data1, data2, num_points=None):
    """Calculate the Kullback-Leibler divergence between two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the K-L divergence is
    calculated.
    Note: KDE will often underestimate the probability at the far tails of the
    distribution (outside of where supported by the data), which can lead to
    overestimates of K-L divergence for highly non-overlapping datasets.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, p1, p2 = _get_point_estimates(data1, data2, num_points)
    return _kl_divergence(xs, p1, p2)


def _get_kd_estimator_and_xs(data, num_points):
    """Get KDE estimator for a given dataset, and generate a good set of
    points to sample the density at."""
    data = numpy.asarray(data, dtype=numpy.float)
    kd_estimator = kde.gaussian_kde(data)
    data_samples = kd_estimator.resample(num_points // 2)[0]
    xs = numpy.sort(data_samples)
    return kd_estimator, xs


def _get_estimators_and_xs(data1, data2, num_points):
    """Get KDE estimators for two different datasets and a set of points
    to evaluate both distributions on."""
    if num_points is None:
        num_points = min(5000, (len(data1) + len(data2)) // 2)
    kd1, xs1 = _get_kd_estimator_and_xs(data1, num_points // 2)
    kd2, xs2 = _get_kd_estimator_and_xs(data2, num_points // 2)
    xs = numpy.sort(numpy.concatenate([xs1, xs2]))
    return xs, kd1, kd2


def _get_point_estimates(data1, data2, num_points):
    """Get point estimates for KDE distributions for two different datasets.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points)
    with numpy.errstate(under='ignore'):
        p1 = kd1(xs)
        p2 = kd2(xs)
    return xs, p1, p2


def _kl_divergence(xs, p1, p2):
    """Calculate Kullback-Leibler divergence of p1 and p2, which are assumed to
    values of two different density functions at the given positions xs.
    Return divergence in nats."""
    with numpy.errstate(divide='ignore', invalid='ignore'):
        kl = p1 * (numpy.log(p1) - numpy.log(p2))
    kl[~numpy.isfinite(kl)] = 0  # small numbers in p1 or p2 can cause NaN/-inf, etc.
    return numpy.trapz(kl, x=xs)  # integrate curve
