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
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance')

import matplotlib.pyplot as mp

# %% Organização dos dados

source_excess = 'NefinFactors.csv'
source_portfolio = 'NefinPortfolios.csv'

data = pd.read_csv(
    os.path.join('Data', source_excess), index_col=0, parse_dates=True)
portfolio = pd.read_csv(
    os.path.join('Data', source_portfolio), index_col=0, parse_dates=True)

# Os dados do nefin são retornos lineares por dia útil. Note que a taxa Risk Free
### utilizada é o CDI do dia, então é possível fazer um minisanity-check construindo
### anualizando a taxa por 252 du por ano e recuperar o cdi diário (ou quase).
# (1 + data)**252

# mode = 'D'
# mode = 'W'
mode = 'M'

if mode in ['M', 'W']:
    
    data = (1 + data).cumprod().resample(mode).last().rename({
        x: x.replace('Risk Factor - Nefin - ', '') for x in data
        }, axis = 1)
    portfolio = (1 + portfolio).cumprod().resample(mode).last().rename({
        x: x.replace('Portfolio - Nefin - ', '') for x in portfolio
        }, axis = 1)
    
    if mode == 'W':
        data.loc[data.index[0] - dt.timedelta(7)] = 1
        portfolio.loc[portfolio.index[0] - dt.timedelta(7)] = 1
    elif mode == 'M':
        data.loc[data.index[0] - pd.offsets.MonthEnd()] = 1
        portfolio.loc[portfolio.index[0] - pd.offsets.MonthEnd()] = 1
        
    data_dt = data.index[0] - pd.offsets.MonthEnd()
    data = data.sort_index().pct_change(
        fill_method = None
        ).drop(data_dt)
    
    port_dt = portfolio.index[0] - pd.offsets.MonthEnd()
    portfolio = portfolio.sort_index().pct_change(
        fill_method = None
        ).drop(port_dt)

if source_excess == 'NefinFactors.csv':
    
    portfolio.drop(
        ['EW - Industry_2', 'VW - Industry_2'], 
        inplace = True, axis = 1)
    
    factors = 100*data.drop('Risk_free - Livre de Risco', axis = 1)
    riskFreeAvailable = True
    riskFree = 100*data[['Risk_free - Livre de Risco']]
    returns = 100*portfolio.copy()
    # returns_in_excess = returns.loc[
    #     returns.index.isin(riskFree.index)
    #     ] - riskFree.loc[returns.index.isin(riskFree.index)].values
    # factors_mais_riskFree = factors.loc[
    #     factors.index.isin(riskFree.index)
    #     ] + riskFree.loc[factors.index.isin(factors.index)].values
else:
    factors = data.copy()
    riskFreeAvailable = False
    riskFree = None
    returns = 100*portfolio.copy()

if riskFree is None:
    riskFree = (returns.dropna() @ np.linalg.solve(
        returns.dropna().T @ returns.dropna(),
        returns.dropna().T @ np.ones((returns.dropna().shape[0],1)))
        ).rename({0: 'Mimicking Risk-Free'}, axis = 1)
    
    riskFree.plot()   
    
# returns_in_excess = returns.loc[
#     returns.index.isin(riskFree.index)
#     ] - riskFree.loc[returns.index.isin(riskFree.index)].values
# returns_in_excess2 = returns.iloc[:,:-1] - returns.iloc[:,-1:].values
# factors_mais_riskFree = factors.loc[
#     factors.index.isin(riskFree.index)
#     ] + riskFree.loc[factors.index.isin(factors.index)].values


# %% Definição da classe de retornos
class returnSet:
    
    def __init__(self, returns, riskFree):
        
        self.debugMode = True
        
        self.returns = returns.copy()
        self.riskFree = riskFree.copy()
        
        self.riskFreeAvailable = self.riskFree is not None
        
        if self.riskFreeAvailable:
            self.excessRiskFree = self.returns - self.riskFree.values
        
        self.excessReturns = \
            self.returns.iloc[:,:-1] - self.returns.iloc[:,-1:].values
        
        # Constrói o discount Factor para os excessos de retorno (
        ## sem considerar a RiskFree para compor os excessos ou o portfolio)
        self.getDiscountFactor(mode = 'Excess')
        self.getDiscountFactor(mode = 'Returns')
        
        # Cria os retornos da Mean-Variance Frontier e a decomposição ortogonal
        self.getDiscountFactorReturn()
        self.getReStar()
        self.decomposicaoRetornos()
    
    def plotMeanVariance(self, ax = None):
        
        if ax is None:
            fig, ax = mp.subplots(1,1,figsize = (14,7))
        
        self.plotScatterPlot(
            ax = ax,
            returns_df=self.returns,
            **{'label': 'Portfolios', 'color': 'Red'}
            )

        self.plotScatterPlot(
            ax = ax,
            returns_df=pd.DataFrame(2*[[self.Rf]]),
            **{'label': 'Risk-Free', 'color': 'Green'}
            )
        
        self.plotScatterPlot(
            ax = ax,
            returns_df = self.getRetornosFronteira(lb = -2, ub = 10, Nstep = 1000),
            **{'label': 'Fronteira de Risky-Assets',
               'color': 'Blue',
               'title': 'Risco vs Retorno',
               's': 20*.05
               }
            )
        
        return ax
        
    def plotScatterPlot(self,
                        ax = None,
                        returns_df = None,
                        **kwargs
                        ):
        
        mean_name = 'Mean Return'
        std_name = 'Std'
        mean_cov = pd.concat([
            returns_df.mean().to_frame(mean_name),
            returns_df.std().to_frame(std_name),
            ], axis = 1)
        
        if ax is None:
            ax = mean_cov.plot.scatter(
                y=mean_name, x=std_name, **kwargs
                )
        else:
            mean_cov.plot.scatter(
                ax = ax,
                y=mean_name, x=std_name, **kwargs
                )
        
        ax.grid(True)
            
        return ax
        
    def getDiscountFactor(self, returns_df = None, mode = 'Returns'):
        
        if mode == 'Returns':
            if returns_df is None:
                self.getDiscountFactorWithReturns(self.returns)
            else:
                self.getDiscountFactorWithReturns(returns_df)
        else:
            if returns_df is None:
                self.getDiscountFactorWithExcessReturns(self.excessReturns)
            else:
                self.getDiscountFactorWithExcessReturns(returns_df)
    
    def getDiscountFactorWithReturns(self, returns_df):
        
        if riskFreeAvailable:
            self.Rf = self.riskFree.mean().iloc[0]
        else:
            self.Rf = 1.
        
        mu = returns_df.mean().to_frame('Portfolios - Mean')
        
        xStar_direction = (
            np.linalg.solve(returns_df.cov(ddof=0), (
                np.ones(mu.shape) - 1/self.Rf * mu )
                ).T @ (returns_df.T - mu.values)
        ).T
        
        xStar = 1/self.Rf + xStar_direction
        xStar.columns = ['Discount Factor']
        
        self.DiscountFactorWithReturns = xStar.copy()
        self.DiscountFactor = xStar.copy()
    
    def getDiscountFactorWithExcessReturns(self, returns_df):
        
        if riskFreeAvailable:
            self.Rf = riskFree.iloc[-1,0]
        else:
            self.Rf = 1.

        mu_ex = returns_df.mean().to_frame('Portfolios menos RF - Mean')

        xStar_direction = (
            np.linalg.solve(returns_df.cov(ddof=0), mu_ex ).T
            @ (returns_df.T - mu_ex.values)
        ).T

        xStar = 1/self.Rf - 1/self.Rf * xStar_direction
        xStar.columns = ['Discount Factor']
        
        self.DiscountFactorWithExcessReturns = xStar.copy()
        self.DiscountFactor = xStar.copy()
    
    def getReturnPrice(self, returns_df):
        
        return ((self.DiscountFactor.T @ returns_df).T).rename(
            {'Discount Factor': 'Preço do Retorno'}, axis = 1
            ) / self.DiscountFactor.shape[0]
    
    def getDiscountFactorReturn(self):
        
        pdiscount = self.getReturnPrice(self.DiscountFactor).iloc[0,0]
        
        self.RStar = self.DiscountFactor / pdiscount
        self.RStar.columns = ['R*']

        # # Sanity check: preço do retorno associado ao discount factor
        # print(f"pRStar: {self.getReturnPrice(self.RStar)}")
    
    # Projeta Y no espaço de (excesso de) retornos returns_df
    def projetaYemRetornos(self, Y, returns_df):   
        return returns_df @ self.getCoeffProjecaoYemRetornos(Y, returns_df)
    
    def getCoeffProjecaoYemRetornos(self, Y, returns_df):   
        return np.linalg.solve(returns_df.T @ returns_df, returns_df.T @ Y)
    
    def getReStar(self):
        returns_df = self.excessReturns
        Y = np.ones(
            (self.returns.shape[0],1)
            )
        self.ReStar = self.projetaYemRetornos(Y, returns_df).rename({
            0: 'Re*'
            }, axis = 1)
        
    # decompõe os retornos em RStar, w_i*ReStar e n_i
    def decomposicaoRetornos(self):
        
        self.w_returns = pd.DataFrame(
            self.getCoeffProjecaoYemRetornos(
                self.returns - self.RStar.values,
                self.ReStar
                ),
            columns = self.returns.columns,
            index = self.ReStar.columns
            )
        
        self.eta_returns = self.returns - self.RStar.values \
            - self.ReStar @ self.w_returns
        
        if self.debugMode:
            
            retornos = self.getReturnPrice(self.returns)
            retornos.columns = ['Preço Inicial']
            
            RStar = self.getReturnPrice(self.RStar)
            RStar = pd.DataFrame(
                self.RStar.iloc[0,0]*np.ones(retornos.shape),
                columns = ['Preço RStar'],
                index = retornos.index
                )
            
            ReStar = self.getReturnPrice(self.ReStar @ self.w_returns)
            ReStar.columns = ['Preço w_i*ReStar']
            
            eta_returns = self.getReturnPrice(self.eta_returns)            
            eta_returns.columns = ['Preço Eta Returns']
            
            self.precos_dos_componentes = pd.concat([
                retornos, RStar, ReStar, eta_returns
                ], axis = 1)
        
    def getRetornosFronteira(
            self,
            lb = -1, ub = 4, Nstep = 1000):
        
        step = (ub - lb)/Nstep

        w_plot = np.array([[lb + i*step for i in range(Nstep+1)]])

        return self.RStar.values + self.ReStar @ w_plot

# %% Definição da classe de fatores
class factorSet(returnSet):
    
    def __init__(self, factors, riskFree):
        
        self.debugMode = True
        
        self.excessReturns = factors.copy()
        self.riskFree = riskFree.copy()
        
        self.riskFreeAvailable = self.riskFree is not None
        if self.riskFreeAvailable:
            self.returns = self.excessReturns + self.riskFree.values
        
        # Constrói o discount Factor para os excessos de retorno (
        ## sem considerar a RiskFree para compor os excessos ou o portfolio)
        self.getDiscountFactor(mode = 'Excess')
        self.getDiscountFactor(mode = 'Returns')
        
        # Cria os retornos da Mean-Variance Frontier e a decomposição ortogonal
        self.getDiscountFactorReturn()
        self.getReStar()
        self.decomposicaoRetornos()
        
    def plotMeanVariance(self, ax = None):
        
        if ax is None:
            fig, ax = mp.subplots(1,1,figsize = (14,7))
        
        ax = self.plotScatterPlot(
            ax = ax,
            returns_df=self.returns,
            **{'label': 'Fatores', 'color': 'Red'}
            )

        ax = self.plotScatterPlot(
            ax = ax,
            returns_df=pd.DataFrame(2*[[self.Rf]]),
            **{'label': 'Risk-Free', 'color': 'Green'}
            )
        
        ax = self.plotScatterPlot(
            ax = ax,
            returns_df = self.getRetornosFronteira(lb = -2, ub = 10, Nstep = 1000),
            **{'label': 'Fronteira de Risky-Assets',
               'color': 'Blue',
               'title': 'Risco vs Retorno',
               's': 20*.05
               }
            )
        return ax
    
# %%

portfolioSet1 = returnSet(returns, riskFree)

if portfolioSet1.debugMode:
    precos = portfolioSet1.precos_dos_componentes

ax = portfolioSet1.plotMeanVariance()

factorSet1 = factorSet(factors, riskFree)
ax = factorSet1.plotMeanVariance(ax)

ax = factorSet1.plotMeanVariance()

# %%

portfolioSet2 = returnSet(returns.iloc[135:], riskFree.iloc[135:])

if portfolioSet2.debugMode:
    precos = portfolioSet2.precos_dos_componentes
      
ax2 = portfolioSet2.plotMeanVariance()

factorSet2 = factorSet(factors.iloc[135:], riskFree.iloc[135:])

factorSet2.plotMeanVariance(ax2)

factorSet2.plotMeanVariance()

# %%


# %% Visualização dos retornos e excessos de retorno levantados
mean_cov = pd.concat([
    factors.mean().to_frame('Mean Excess Return'),
    factors.std().to_frame('Std'),
    ], axis = 1)
if riskFreeAvailable:
    mean_cov.loc['Risk-free'] = 0

ax = mean_cov.drop('Risk-free').plot.scatter(
    y='Mean Excess Return', x='Std', c='DarkBlue',
    label = 'Fatores de Risco',
    fontsize = 12
    )

mean_cov.loc[['Risk-free']].plot.scatter(
    ax = ax,
    y='Mean Excess Return', x='Std', c='Green',
    label = 'Risk-free',
    fontsize = 12
    )

mean_cov_port = pd.concat([
    returns_in_excess.mean().to_frame('Mean Excess Return'),
    returns_in_excess.std().to_frame('Std'),
    ], axis = 1)

mean_cov_port.plot.scatter(
        ax = ax,
        y='Mean Excess Return', x='Std', c='Red',
        label = 'Portfolios (-Rf)',
        fontsize = 12
    )

ax.grid(True)
ax.set_title('Risco x Retorno esperados disponíveis', fontsize = 14)
mp.show()

# %%
"""Para encontrar a fronteira eficiente, decomporemos os retornos por meio da
decomposição ortogonal dos retornos. Todo retorno pode ser decomposto em:
    1) um retorno que seja *múltiplo* do fator de desconto estocástico ($R^*$);
    2) um excesso de retorno que seja a projeção do retorno constante
    contra o espaço de excessos de retornos ($R^e$); e
    3) um excesso de retorno com retorno esperado 0
Esses 3 componentes são ortogonais entre si.
"""

# %% Fator de desconto que precifica todos os portfolios
### Abordagem de Hansen and Jagannathan (1991) apud Cochrane (2005), cap. 4
"""O primeiro passo é estimar um fator de desconto que precifica
todos os retornos observados (como 1, já que E[mR] = 1) e que tenha
um retorno calibrado para o último valor disponível da taxa risk-free observada
ou então para um nível médio de 1%."""

portfolioSet.getDiscountFactor(mode = 'Returns')
portfolioSet.getReturnPrice(portfolioSet.returns)


# %% Creating Re^*
# Re^* = E[Re].T @ E[Re @ Re.T]^(-1) @ Re
"""Em sequência, estimamos a projeção do retorno constante contro os excessos de
retorno para obter ($Re^*$).
"""

mu_ex2 = returns_in_excess2.mean().to_frame('Portfolios menos Port - Mean')

# By default, cov retrieves unbiased estimates
# var_cov = ff25.cov().values
E_ReRe = (
    returns_in_excess2.values.T @ returns_in_excess2.values
    )/returns_in_excess2.shape[0]

ReStar_weights = pd.DataFrame(
    np.linalg.solve(E_ReRe, mu_ex2),
    columns = ['Re_star'],
    index = returns_in_excess2.columns)

ReStar = returns_in_excess2 @ ReStar_weights

ReStar_cov = pd.concat([
    ReStar.mean().to_frame('Mean Excess Return'),
    ReStar.std().to_frame('Std'),
    ], axis = 1)
ReStar

# %% Decomposição dos retornos
RStar = portfolioSet.RStar.copy()
w_i = pd.DataFrame(
    np.linalg.solve(
        ReStar.T @ ReStar,
        ReStar.T @ (returns - RStar.values)
        ),
    columns = returns.columns,
    index = ReStar.columns
    )

xStar.T @ (returns - RStar.values)/xStar.shape[0]
xStar.T @ (ReStar) / xStar.shape[0]
xStar.T @ (returns - RStar.values -  ReStar @ w_i)/xStar.shape[0]

# %% Depicting a fronteira eficiente


pd.concat([
    frontier.mean().to_frame('Mean Excess Return'),
    frontier.std().to_frame('Std'),
    ], axis = 1).plot.scatter(
        y='Mean Excess Return', x='Std', c='Green',
        label = 'Efficient Frontier', s = 20*.05,
        fontsize = 12
        )



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