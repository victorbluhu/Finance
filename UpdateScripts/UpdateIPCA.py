# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:25:02 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

import pandas as pd
import datetime as dt

path = r'RawData\IPCA.csv'

df = pd.read_csv(
    path,
    sep = ';',
    encoding = 'latin1').iloc[:-1]
df.columns = ['dtref', 'IPCA']
df.set_index('dtref', inplace = True)
df = df.applymap(lambda x : float(x.replace(',','.')))
new_dates = [x.split('/') for x in df.index]
df.index = [
    dt.datetime(int(x[1]),int(x[0]),1) + pd.offsets.MonthEnd()
    for x in new_dates]

df.to_csv(r'Data\IPCA.csv')

pd.read_csv(r'Data\IPCA.csv', index_col = 0, parse_dates=True)
