# The Selector library provides a set of tools for selecting a
# subset of the dataset and computing diversity.
#
# Copyright (C) 2023 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Similarity Module."""

from itertools import combinations_with_replacement

import numpy as np

__all__ = ["pairwise_similarity_bit", "tanimoto", "modified_tanimoto"]


def pairwise_similarity_bit(X: np.array, metric: str) -> np.ndarray:
    """Compute pairwise similarity coefficient matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional space.
    metric : str
        The metric used when calculating similarity coefficients between samples in a feature array.
        Method for calculating similarity coefficient. Options: `"tanimoto"`, `"modified_tanimoto"`,
        `"bub"`, `"fai"`, `"ja"`, `"jt"`, `"rt"`, `"rr"`, `"sm"`, `"ss1"`, `"ss2"`.

    Returns
    -------
    s : ndarray of shape (n_samples, n_samples)
        A symmetric similarity matrix between each pair of samples in the feature matrix.
        The diagonal elements are directly computed instead of assuming that they are 1.
    """

    available_methods = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }
    supported_metrics = list(available_methods.keys()) + [
        "bub",
        "fai",
        "ja",
        "jt",
        "rt",
        "rr",
        "sm",
        "ss1",
        "ss2",
    ]
    if metric not in supported_metrics:
        raise ValueError(
            f"Argument metric={metric} is not recognized! Choose from {available_methods.keys()}"
        )
    if X.ndim != 2:
        raise ValueError(f"Argument features should be a 2D array, got {X.ndim}")

    # make pairwise m-by-m similarity matrix
    n_samples = X.shape[0]
    n_bits = X.shape[1]
    s = np.zeros((n_samples, n_samples))
    # compute similarity between all pairs of points (including the diagonal elements)
    for i, j in combinations_with_replacement(range(n_samples), 2):
        x = X[i]
        y = X[j]
        if metric in available_methods.keys():
            s[i, j] = s[j, i] = available_methods[metric](x, y)
        else:
            # a: number of common on bits
            a = np.dot(x, y)
            # d: number of common off bits
            d = np.dot(1 - x, 1 - y)
            # dis = b + c : 1-0 mismatches
            dis = n_bits - a - d
            s[i, j] = s[j, i] = sim_indices(metric=metric, n_bits=n_bits, a=a, d=d, dis=dis)

    return s


def sim_indices(metric, n_bits, a, d, dis):
    """Compute similarity indices.

    Parameters
    ----------
    metric : str
        The metric used when calculating similarity coefficients,
        options: bub, fai, ja, jt, rt, rr, sm, ss1, ss2.
    n_bits : int
        Number of bits in the fingerprint.
    a : int
        Number of common on bits.
    d : int
        Number of common off bits.
    dis : int
        Number of 1-0 mismatches.

    Returns
    -------
    sim : float
        Similarity index between two fingerprints.

    Notes
    -----
    The definitions were taken from the following paper [1]_

    .. [1] Miranda-Quintana, Ramón Alain, et al. Extended similarity indices: the benefits of
    comparing more than two objects simultaneously. Part 1: Theory and characteristics.
    Journal of Cheminformatics 13.1 (2021): 32.

    """
    sim_func_dict = {
        # BUB: Baroni-Urbani-Buser
        "bub": ((a * d) ** 0.5 + a) / ((a * d) ** 0.5 + a + dis),
        # Fai: Faith
        "fai": (a + 0.5 * d) / n_bits,
        # Ja: Jaccard
        "ja": (3 * a) / (3 * a + dis),
        # JT: Jaccard-Tanimoto
        "jt": a / (a + dis),
        # RT: Rogers-Tanimoto
        "rt": (a + d) / (n_bits + dis),
        # RR: Russel-Rao
        "rr": a / n_bits,
        # SM: Sokal-Michener
        "sm": (a + d) / n_bits,
        # SS1: Sokal-Sneath 1
        "ss1": a / (a + 2 * dis),
        # SS2: Sokal-Sneath 2
        "ss2": (2 * (a + d)) / (n_bits + (a + d)),
    }

    sim = sim_func_dict[metric]

    return sim


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient or index (a.k.a. Jaccard similarity coefficient).

    For two binary or non-binary arrays :math:`A` and :math:`B`, Tanimoto coefficient
    is defined as the size of their intersection divided by the size of their union:

    ..math::
        T(A, B) = \frac{| A \cap B|}{| A \cup B |} =
        \frac{| A \cap B|}{|A| + |B| - | A \cap B|} =
        \frac{A \cdot B}{\|A\|^2 + \|B\|^2 - A \cdot B}

    where :math:`A \cdot B = \sum_i{A_i B_i}` and :math:`\|A\|^2 = \sum_i{A_i^2}`.

    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`B` in an `n_features` dimensional space.

    Returns
    -------
    coeff : float
        Tanimoto coefficient between feature arrays :math:`A` and :math:`B`.

    Bajusz, D., Rácz, A., and Héberger, K.. (2015)
    Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
    Journal of Cheminformatics 7.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Arguments a and b should be 1D arrays, got {a.ndim} and {b.ndim}")
    if a.shape != b.shape:
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )
    coeff = sum(a * b) / (sum(a**2) + sum(b**2) - sum(a * b))
    return coeff


def modified_tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute the modified tanimoto coefficient from bitstring vectors of data points A and B.

    Adjusts calculation of the Tanimoto coefficient to counter its natural bias towards
    shorter vectors using a Bernoulli probability model.

    ..math::
    MT = \frac{2-p}{3}T_1 + \frac{1+p}{3}T_0

    where :math:`p` is success probability of independent trials,
    :math:`T_1` is the number of common '1' bits between data points
    (:math:`T_1 = | A \cap B |`), and :math:`T_0` is the number of common '0'
    bits between data points (:math:`T_0 = |(1-A) \cap (1-B)|`).


    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`B` in an `n_features` dimensional space.

    Returns
    -------
    mt : float
        Modified tanimoto coefficient between bitstring feature arrays :math:`A` and :math:`B`.

    Notes
    -----
    The equation above has been derived from

    ..math::
    MT_\alpha= {\alpha}T_1 + (1-\alpha)T_0

    where :math:`\alpha = \frac{2-p}{3}`. This is done so that the expected value
    of the modified tanimoto, :math:`E(MT)`, remains constant even as the number of
    trials :math:`p` grows larger.

    Fligner, M. A., Verducci, J. S., and Blower, P. E.. (2002)
    A Modification of the Jaccard-Tanimoto Similarity Index for
    Diverse Selection of Chemical Compounds Using Binary Strings.
    Technometrics 44, 110-119.
    """
    if a.ndim != 1:
        raise ValueError(f"Argument `a` should have dimension 1 rather than {a.ndim}.")
    if b.ndim != 1:
        raise ValueError(f"Argument `b` should have dimension 1 rather than {b.ndim}.")
    if a.shape != b.shape:
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )

    n_features = len(a)
    # number of common '1' bits between points A and B
    n_11 = sum(a * b)
    # number of common '0' bits between points A and B
    n_00 = sum((1 - a) * (1 - b))

    # calculate Tanimoto coefficient based on '0' bits
    t_1 = 1
    if n_00 != n_features:
        # bit strings are not all '0's
        t_1 = n_11 / (n_features - n_00)
    # calculate Tanimoto coefficient based on '1' bits
    t_0 = 1
    if n_11 != n_features:
        # bit strings are not all '1's
        t_0 = n_00 / (n_features - n_11)

    # combine into modified tanimoto using Bernoulli Model
    # p = independent success trials
    #       evaluated as total number of '1' bits
    #       divided by 2x the fingerprint length
    p = (n_features - n_00 + n_11) / (2 * n_features)
    # mt = x * T_1 + (1-x) * T_0
    #       x = (2-p)/3 so that E(mt) = 1/3, no matter the value of p
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt
