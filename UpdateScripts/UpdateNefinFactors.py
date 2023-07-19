# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 00:40:26 2023

@author: Victor
"""

import pandas as pd
import datetime as dt

import os
# Para simplificar, estou rodando tudo importado scripts como se fossem pacotes
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')


# %% 
# Baixar manualmente de https://nefin.com.br/data/risk_factors.html
sources = {
    'MKT': r'Market_Factor.xls',
    'SMB': r'SMB_Factor.xls',
    'HML': r'HML_Factor.xls',
    'WML': r'WML_Factor.xls',
    'IML': r'IML_Factor.xls',
    'RF':  r'Risk_Free.xls'
}

df = []
for key, val in sources.items():
    
    print(key)
    
    temp = pd.read_excel(
        r'https://nefin.com.br/resources/risk_factors/' + val
        # https://nefin.com.br/resources/risk_factors/Market_Factor.xls
        )
    
    # temp = pd.read_excel(os.path.join('RawData', val))
    temp['dtref'] = [dt.datetime(x[0],x[1],x[2]) for x in temp[['year','month','day']].values]
    
    df += [temp.iloc[:,-2:].set_index('dtref')]
    
df = pd.concat(df, axis = 1)
df.columns = ['Risk Factor - Nefin - ' + col for col in df.columns]

# Reordena os dados
df = df[[
    'Risk Factor - Nefin - Rm_minus_Rf', 'Risk Factor - Nefin - SMB',
    'Risk Factor - Nefin - HML', 'Risk Factor - Nefin - WML',
    'Risk Factor - Nefin - IML', 'Risk Factor - Nefin - Risk_free'
]]

df.columns = [
    'Risk Factor - Nefin - Rm_minus_Rf - Retorno de Mercado',
    'Risk Factor - Nefin - SMB - Size',
    'Risk Factor - Nefin - HML - Value',
    'Risk Factor - Nefin - WML - Momentum',
    'Risk Factor - Nefin - IML - Iliquidez',
    'Risk Factor - Nefin - Risk_free - Livre de Risco'
]

df.to_csv(r'Data\NefinFactors.csv')

# read_df = pd.read_csv(r'Data\NefinFactors.csv', index_col=0, parse_dates=True)
