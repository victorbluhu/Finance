# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:42:28 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

import pandas as pd

for path in [r'CPIAUCSL.csv', r'CPIAUCNS.csv']:
    df = pd.read_csv(
        os.path.join(r'RawData', path),
        parse_dates = True,
        index_col = 0
        )
    df.index = [x + pd.offsets.MonthEnd() for x in df.index]
    df.index.name = 'dtref'
    df.to_csv(os.path.join(r'Data', path))

pd.read_csv(os.path.join(r'Data', path), index_col = 0, parse_dates=True)
