"""
reference: https://sixsigmastudyguide.com/p-attribute-charts/
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manufacturing.visual import control_chart_base

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


def p_control_chart(data: pd.DataFrame):
    columns = data.columns
    if 'lot_id' not in columns and 'datetime' in columns:

        # separate data into reasonable lots
        rules = [
            'H',   # hourly
            'B',   # daily (business days)
            'W',   # weekly
            '2W',  # every 2 weeks
            'M',   # monthly
        ]

        sample_rule = None
        for rule in rules:
            pass_rates = df['pass'].resample(rule).mean()  # pass rate will be 1.0
            size = df['pass'].resample(rule).size().sum()

            value_counts = pass_rates.value_counts()
            ratio = value_counts.iloc[0] / size

            if ratio <= 0.005:  # less than 5% of intervals show 100% pass rate
                sample_rule = rule
                break

        if sample_rule is None:
            raise ValueError('no valid sample interval resulted in an appropriate nonconformance ratio')

        sampled_dfs = df['pass'].resample(sample_rule)
    elif 'lotid' in columns:
        sampled_dfs = df.groupby(by='lotid')['pass']
    else:
        raise ValueError('"datetime" or "lotid" must be columns within the dataframe')

    nonconformity_rates = 1 - sampled_dfs.mean()

    k = len(sampled_dfs)
    nbar = len(df) / k
    pbar = len(df[df['pass'] == False]) / len(df)

    ps = []
    ucls = []
    lcls = []
    for ts, sampled_df in sampled_dfs:
        n_i = sampled_df.size

        if n_i >= 20:
            n_i_inverse = 1.0 / n_i

            if n_i_inverse is not None:
                ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) * n_i_inverse)
                lcl = pbar - 3 * np.sqrt(pbar * (1 - pbar) * n_i_inverse)
            else:
                ucl = lcl = np.nan

            if lcl < 0.0:
                lcl = 0.0
            if ucl > 1.0:
                ucl = 1.0

            ps.append(1.0 - sampled_df.mean())
            ucls.append(ucl)
            lcls.append(lcl)
        else:
            _logger.warning(f'sample set eliminated due to insufficient lot size of {n_i}')

    fig, ax = plt.subplots()
    control_chart_base(
        data=ps,
        upper_control_limit=ucls,
        lower_control_limit=lcls,
        highlight_beyond_limits=True,
        highlight_zone_a=False,
        highlight_zone_b=False,
        highlight_zone_c=False,
        highlight_trend=False,
        highlight_mixture=False,
        highlight_stratification=False,
        highlight_overcontrol=False,
        ax=ax,
    )

    y_low, y_high = ax.get_ylim()
    if y_high > 1.0:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.0)


if __name__ == '__main__':
    path = Path('../data/pchart_data_by_lots.tsv')
    df = pd.read_csv(path, delimiter='\t')
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # df.set_index('datetime', inplace=True)
    print(df.head())
    p_control_chart(df)

    plt.show()
