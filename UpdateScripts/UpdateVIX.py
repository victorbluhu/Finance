# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:22:43 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

import pandas as pd

path = r'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'

df = pd.read_csv(path, index_col=0, parse_dates = True)
df.index.name = 'dtref'

ax = df.loc[df.index.year < 2004, ['CLOSE']].plot()
df.loc[df.index.year >= 2004, ['CLOSE']].plot(ax = ax)

df.to_csv(r'Data\VIX.csv')
# pd.read_csv(r'Data\VIX.csv', index_col = 0, parse_dates=True)
