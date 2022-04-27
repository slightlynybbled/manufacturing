"""
Contains several lookup table constants.
"""
import numpy as np

c4_table = [
    np.nan,
    np.nan,
    0.7979,
    0.8862,
    0.9213,
    0.94,
    0.9515,
    0.9594,
    0.965,
    0.9693,
    0.9727,
    0.9754,
    0.9776,
    0.9794,
    0.981,
    0.9823,
    0.9835,
    0.9845,
    0.9854,
    0.9862,
    0.9869,
    0.9876,
    0.9882,
    0.9887,
    0.9892,
    0.9896,
    0.9901,
    0.9904,
    0.9908,
    0.9911,
    0.9914,
]
d2_table = [
    np.nan,
    np.nan,
    1.1284,
    1.6926,
    2.0588,
    2.3259,
    2.5344,
    2.7044,
    2.8472,
    2.97,
    3.0775,
    3.1729,
    3.2585,
    3.336,
    3.4068,
    3.4718,
    3.532,
    3.5879,
    3.6401,
    3.689,
    3.7349,
    3.7783,
    3.8194,
    3.8583,
    3.8953,
    3.9306,
    3.9643,
    3.9965,
    4.0274,
    4.057,
    4.0855,
]
d3_table = [
    np.nan,
    np.nan,
    0.8525,
    0.8884,
    0.8798,
    0.8641,
    0.848,
    0.8332,
    0.8198,
    0.8078,
    0.7971,
    0.7873,
    0.7785,
    0.7704,
    0.763,
    0.7562,
    0.7499,
    0.7441,
    0.7386,
    0.7335,
    0.7287,
    0.7242,
    0.7199,
    0.7159,
    0.7121,
    0.7084,
    0.705,
    0.7017,
    0.6986,
    0.6955,
    0.6927,
]
d4_table = [
    np.nan,
    np.nan,
    0.9539,
    1.5878,
    1.9783,
    2.2569,
    2.4717,
    2.6455,
    2.7908,
    2.9154,
    3.0242,
    3.1205,
    3.2069,
    3.2849,
    3.3562,
    3.4217,
    3.4821,
    3.5383,
    3.5907,
    3.6398,
    3.6859,
    3.7294,
    3.7706,
    3.8096,
    3.8468,
    3.8822,
    3.9159,
    3.9482,
    3.9791,
    4.0088,
    4.0374,
]


def calc_A2(n: int) -> float:
    """
    Calculate the A2 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of A2
    """
    try:
        d2 = d2_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(d2_table)}")
    if np.isnan(d2):
        raise ValueError(f"the constant `n` must be 2 or greater")
    return 3 / (d2 * np.sqrt(n))


def calc_A3(n: int) -> float:
    """
    Calculate the A3 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of A3
    """
    try:
        c4 = c4_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(c4_table)}")
    if np.isnan(c4):
        raise ValueError(f"the constant `n` must be 2 or greater")
    return 3 / (c4 * np.sqrt(n))


def calc_B3(n: int) -> float:
    """
    Calculate the B3 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of B3
    """
    try:
        c4 = c4_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(c4_table)}")
    if np.isnan(c4):
        raise ValueError(f"the constant `n` must be 2 or greater")

    value = 1 - (3 / c4) * np.sqrt(1 - c4**2)
    value = max(0.0, value)
    return value


def calc_B4(n: int) -> float:
    """
    Calculate the B4 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of B4
    """
    try:
        c4 = c4_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(c4_table)}")
    if np.isnan(c4):
        raise ValueError(f"the constant `n` must be 2 or greater")

    return 1 + (3 / c4) * np.sqrt(1 - c4**2)


def calc_D3(n: int) -> float:
    """
    Calculate the D3 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of D3
    """
    try:
        d3 = d3_table[n]
        d2 = d2_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(c4_table)}")
    if np.isnan(d3):
        raise ValueError(f"the constant `n` must be 2 or greater")

    value = 1 - 3 * d3 / d2
    value = max(0.0, value)
    return value


def calc_D4(n: int) -> float:
    """
    Calculate the D4 constant for a given value `n`

    :param n: the number of samples in each group
    :return: a floating point representation of D4
    """
    try:
        d3 = d3_table[n]
        d2 = d2_table[n]
    except KeyError:
        raise ValueError(f"the constant `n` must be less than {len(c4_table)}")
    if np.isnan(d3):
        raise ValueError(f"the constant `n` must be 2 or greater")

    return 1 + 3 * d3 / d2
