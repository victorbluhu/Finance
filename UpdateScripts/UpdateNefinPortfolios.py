# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:08:40 2023

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
    'Port_Size': r'3_portfolios_sorted_by_size.xls',
    'Port_BM': r'3_portfolios_sorted_by_book-to-market.xls',
    'Port_Momentum': r'3_portfolios_sorted_by_momentum.xls',
    'Port_Illiquidity': r'3_portfolios_sorted_by_illiquidity.xls',
    # 'Port_Size-BM': r'4_portfolios_sorted_by_book-to-market.xls',
    # 'Port_Size-Momentum': r'4_portfolios_sorted_by_momentum.xls',
    'Port_Size-Illiquidity': r'4_portfolios_sorted_by_size_and_illiquidity_2x2.xls',
    'Port_Industry': r'7_portfolios_sorted_by_industry.xls',
}

df = []
columns = []
for sheet_, name in zip(
        ['Equally Weighted Returns', 'Value Weighted Returns'],
        ['EW', 'VW']
        ):
    for key, val in sources.items():
        
        print(key)
        
        temp = pd.read_excel(
            r'https://nefin.com.br/data/Portfolios/' + val,
            # https://nefin.com.br/resources/risk_factors/Market_Factor.xls,
            sheet_name=sheet_
            )
        
        # temp = pd.read_excel(os.path.join('RawData', val))
        temp['dtref'] = [
            dt.datetime(x[0],x[1],x[2])
            for x in temp[['year','month','day']].values]
        temp.drop(['year','month','day'], axis = 1, inplace = True)
        
        df += [temp.set_index('dtref')]
        columns += [
            f'Portfolio - Nefin - {name} - {x}' for x in temp if 'dtref' != x]
    
df = pd.concat(df, axis = 1)
df.columns = columns

# # Reordena os dados
# df = df[[
#     'Risk Factor - Nefin - Rm_minus_Rf', 'Risk Factor - Nefin - SMB',
#     'Risk Factor - Nefin - HML', 'Risk Factor - Nefin - WML',
#     'Risk Factor - Nefin - IML', 'Risk Factor - Nefin - Risk_free'
# ]]

# df.columns = [
#     'Risk Factor - Nefin - Rm_minus_Rf - Retorno de Mercado',
#     'Risk Factor - Nefin - SMB - Size',
#     'Risk Factor - Nefin - HML - Value',
#     'Risk Factor - Nefin - WML - Momentum',
#     'Risk Factor - Nefin - IML - Iliquidez',
#     'Risk Factor - Nefin - Risk_free - Livre de Risco'
# ]

df.to_csv(r'Data\NefinPorfolios.csv')

# read_df = pd.read_csv(r'Data\NefinFactors.csv', index_col=0, parse_dates=True)
