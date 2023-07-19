# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:07:40 2023

@author: Victor
"""

# %% Preâmbulo de importação

import pandas as pd
import datetime as dt
import numpy as np

import os
# Para simplificar, estou rodando tudo importado scripts como se fossem pacotes
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Factor Analysis')

import matplotlib.pyplot as mp

# %% Organização dos dados

source = 'NefinFactors.csv'

data = pd.read_csv(os.path.join('Data', source), index_col=0, parse_dates=True)

# Os dados do nefin são retornos lineares por dia útil. Note que a taxa Risk Free
### utilizada é o CDI do dia, então é possível fazer um minisanity-check construindo
### anualizando a taxa por 252 du por ano e recuperar o cdi diário (ou quase).
# (1 + data)**252

# mode = 'D'
# mode = 'W'
mode = 'M'

if mode in ['M', 'W']:
    data = (1 + data).cumprod().resample(mode).last()
    if mode == 'W':
        data.loc[data.index[0] - dt.timedelta(7)] = 1
    elif mode == 'M':
        data.loc[data.index[0] - pd.offsets.MonthEnd()] = 1
    data = data.sort_index().pct_change().dropna()

if source == 'NefinFactors.csv':
    excess = 100*data.drop('Risk Factor - Nefin - Risk_free - Livre de Risco', axis = 1)
    riskFreeAvailable = True
    riskFree = 100*data[['Risk Factor - Nefin - Risk_free - Livre de Risco']]
else:
    excess = data.copy()
    riskFreeAvailable = False
    riskFree = None

# %% Visualização dos excessos de retorno levantados
mean_cov = pd.concat([
    excess.mean().to_frame('Mean Excess Return'),
    excess.std().to_frame('Std'),
    ], axis = 1)
if riskFreeAvailable:
    mean_cov.loc['Risk-free'] = 0

ax = mean_cov.plot.scatter(
    y='Mean Excess Return', x='Std', c='DarkBlue',
    # xlabel =  '',
    fontsize = 12
    )
ax.set_title('Risco x Retorno esperados disponíveis', fontsize = 14)
mp.show()

# %% Cálculo do Portfolio Eficiente
"""A primeira construção da fronteira eficiente que implementaremos calcula a fronteira
eficiente diretamente das séries de excessos de retorno. Isso é possível pela
possibilidade de representar qualquer retorno por meio da decomposição ortogonal
de retornos:
    1) O retorno livre de risco (suposto disponível no mercado);
    2) Um múltiplo do excesso de retorno formado pela (i) projeção do retorno livre de risco
    na direção do fator estocástico de desconto e (ii) a taxa livre de risco; e
    3) Um terceiro excesso de retorno ortogonal aos dois primeiros.
O excesso de retorno descrito em 2 é o excesso de retorno obtido pela diferença
do retorno do portfolio eficiente contra a taxa livre de risco.
"""

mean_return = excess.mean().to_frame('Mean Excess Return')
var_cov = excess.cov()
efficient_weights = pd.DataFrame(
    np.linalg.solve(var_cov, mean_return),
    columns = ['Re_emv'],
    index = excess.columns)

Re_emv = excess @ efficient_weights

Re_emv_mean_cov = pd.concat([
    Re_emv.mean().to_frame('Mean Excess Return'),
    Re_emv.std().to_frame('Std'),
    ], axis = 1)

# %% Propriedades do portfolio Eficiente
"""O excesso de retorno do portfolio eficiente é uma combinação linear dos excessos
de retorno disponíveis, então é ele mesmo um excesso de retorno. Logo, 
    1) Podemos ver quanto de cada portfolios disponível se compra;
    2) Podemos representar o risco-retorno dos portfolios disponíveis.
"""

optimal_SR = Re_emv_mean_cov.loc['Re_emv', 'Mean Excess Return']/Re_emv_mean_cov.loc['Re_emv', 'Std']

ax = efficient_weights.rename(
    {key: key.split('Nefin - ')[1] for key in efficient_weights.index} if source == 'NefinFactors.csv'
        else {},
    axis = 0
    ).plot.bar( figsize = (14,7), fontsize = 12)
ax.set_title('Pesos dos Excessos de Retorno no Portfolio Eficiente', fontsize = 14)
mp.show()


ax = mean_cov.plot.scatter(
    y='Mean Excess Return', x='Std', c='DarkBlue',
    fontsize = 12
    )
ax.plot(
        Re_emv_mean_cov.iloc[0,1], Re_emv_mean_cov.iloc[0,0],
        label = 'rmrf', marker = 'o')

max_std = mean_cov.max().loc['Std']
ax.plot(
    np.arange(0,max_std*1.05, max_std*1.05/100),
    optimal_SR*np.arange(0,max_std*1.05, max_std*1.05/100),
    label = 'Efficient Frontier - Re_emv'
        )
ax.legend()

ax.set_title('Risco x Retorno esperados e Fronteira Eficiente', fontsize = 14)
mp.show()



# %% Creating Re^*
# Re^* = E[Re].T @ E[Re @ Re.T]^(-1) @ Re
"""Outra maneira de encontrar a fronteira eficiente é por meio de outra decomposição 
de retornos:
    1) um retorno que seja *múltiplo* do fator de desconto estocástico;
    2) um excesso de retorno que seja a projeção do retorno constante
    contra o espaço de excessos de retornos; e
    3) um excesso de retorno com retorno esperado 0
Da mesma maneira que antes, esses 3 componentes são ortogonais entre si. A diferença
desta construção para a anterior é que ela não assume a existência do retorno livre
de risco.
"""

# By default, cov retrieves unbiased estimates
mu = excess.mean().values[:,np.newaxis]
# var_cov = ff25.cov().values
E_ReRe = (excess.values.T @ excess.values)/excess.shape[0]

Re_star_weights = pd.DataFrame(
    np.linalg.solve(E_ReRe, mu),
    columns = ['Re_star'],
    index = excess.columns)

Re_star = excess @ Re_star_weights

Re_star_mean_cov = pd.concat([
    Re_star.mean().to_frame('Mean Excess Return'),
    Re_star.std().to_frame('Std'),
    ], axis = 1)

# %% Properties of Re_star in comparison to Re_emv

star_SR = Re_star_mean_cov.loc['Re_star', 'Mean Excess Return']/Re_star_mean_cov.loc['Re_star', 'Std']

ax = efficient_weights.rename(
    {key: key.split('Nefin - ')[1] for key in efficient_weights.index} if source == 'NefinFactors.csv'
        else {},
    axis = 0
    ).plot.bar( figsize = (14,7), fontsize = 12)
# ax.set_title('Pesos dos Excessos de Retorno no Portfolio Eficiente', fontsize = 14)
# mp.show()

Re_star_weights.rename(
    {key: key.split('Nefin - ')[1] for key in Re_star_weights.index} if source == 'NefinFactors.csv'
        else {},
    axis = 0
    ).plot.bar(ax = ax, color = 'orange')
ax.set_title('Pesos dos Excessos de Retorno nos Excessos de Retorno', fontsize = 14)
mp.show()

# Efficient Frontier
ax = mean_cov.plot.scatter(
    y='Mean Excess Return', x='Std', c='DarkBlue',
    label = 'Regular Assets',
    fontsize = 12
    )
ax.plot(
        Re_emv_mean_cov.iloc[0,1], Re_emv_mean_cov.iloc[0,0],
        label = 'rmrf', marker = 'o')
ax.plot(
        Re_star_mean_cov.iloc[0,1], Re_star_mean_cov.iloc[0,0],
        label = 're_star', marker = 'o')

max_std = mean_cov.max().loc['Std']
ax.plot(
    np.arange(0,max_std*1.05, max_std*1.05/100),
    optimal_SR*np.arange(0,max_std*1.05, max_std*1.05/100),
    label = 'Efficient Frontier - Re_emv'
        )
ax.plot(
    np.arange(0,max_std*1.05, max_std*1.05/100),
    star_SR*np.arange(0,max_std*1.05, max_std*1.05/100),
    label = 'Efficient Frontier - Re_star', ls = ':'
        )
ax.legend()

ax.set_title('Risco x Retorno esperados e Fronteira Eficiente', fontsize = 14)
mp.show()


# %% 
# Re^* = E[Re].T @ E[Re @ Re.T]^(-1) @ Re
"""Outra maneira de encontrar a fronteira eficiente é por meio de outra decomposição 
de retornos:
    1) um retorno que seja *múltiplo* do fator de desconto estocástico;
    2) um excesso de retorno que seja a projeção do retorno constante
    contra o espaço de excessos de retornos; e
    3) um excesso de retorno com retorno esperado 0
Da mesma maneira que antes, esses 3 componentes são ortogonais entre si. A diferença
desta construção para a anterior é que ela não assume a existência do retorno livre
de risco.
"""