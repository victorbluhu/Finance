# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:37:07 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

import pandas as pd

path = r'RawData\SPX 500.csv'

df = pd.read_csv(path, index_col=0, parse_dates = True).sort_index()
df.index.name = 'dtref'
df.columns = [x.replace(' ','').upper() for x in df]

df.to_csv(r'Data\SPX.csv')

ax = df.loc[df.index.year < 2004, ['CLOSE']].plot()
df.loc[df.index.year >= 2004, ['CLOSE']].plot(ax = ax)

pd.read_csv(r'Data\SPX.csv', index_col = 0, parse_dates=True)
