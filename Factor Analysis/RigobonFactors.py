# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:57:03 2023

@author: vbluhu
"""

import numpy as np
import sys
sys.path.insert(0, r'Y:\Addin\Controle\Codigos Python\libs\Dado_macro')
sys.path.insert(0, r'Y:\Addin\Controle\Codigos Python\libs')
# import My_Data_Functions as mdf
import EconometriaBluhu as eb
import funcoes_auxiliares as faux
# import simulacoes as sm
import basic_functions as bf

from os import path

import pandas as pd
import math
import datetime as dt
import matplotlib.pyplot as mp
import datetime as dt

import os
import base64

import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

# %%
copom = pd.read_excel(r'Y:\Macro\Temp\DatasCopomFinal.xlsx')
copom['REUNIAO'] = 1
copom = copom.loc[copom['Após Mercado'] == 1].set_index(
    # 'Data'
    'Dia Útil Seguinte'
    ).sort_index()[['REUNIAO']]
copom

#%% Puxa os dados de surpresa de copom

data_dict = pd.read_excel(r'Y:\Macro\BRASIL\Dados\Abrir na BBG\SurpresasCopom.xlsx', sheet_name = None)
excelData_dict = pd.read_excel(r'Y:\Macro\BRASIL\Dados\Abrir na BBG\SurpresasCopom.xlsx', sheet_name = None)
data_dict.keys()

# %%
surpresas = []

for name, sheet_name in zip(
    ['s_1', 's_2', 's_3'], ['DI Fut 1st Generic', 'DI Fut 2nd Generic', 'DI Fut 3rd Generic']
):

    temp = data_dict[sheet_name].iloc[6:].copy()
    temp = temp[[temp.columns[0], temp.columns[-1]]]
    temp.columns = ['dtref', name]
    surpresas.append(temp.dropna().set_index('dtref').sort_index()*100)

    
for name, sheet_name in zip(
    ['i_1', 'i_2', 'i_3'], ['DI Fut 1st Generic', 'DI Fut 2nd Generic', 'DI Fut 3rd Generic']
):

    temp = data_dict[sheet_name].iloc[6:].copy()
    temp = temp[[temp.columns[0], temp.columns[-2]]]
    temp.columns = ['dtref', name]
    surpresas.append(temp.dropna().set_index('dtref').sort_index())

# pd.DataFrame.from_dict(surpresas, orient='columns')
surpresas = pd.concat(surpresas, axis = 1).iloc[1:].applymap(float)
surpresas = surpresas.loc[
    surpresas.index>=dt.datetime(2001,4,14)
    ].drop(dt.datetime(2006,12,29))
surpresas.loc[surpresas.index.year == 2012].plot(figsize = (14,7))
surpresas.loc[surpresas.index.year >= 2004].plot(figsize = (14,7))

surpresas.isna().sum()

surpresasCopom = surpresas.loc[
    [x for x in copom.index if x <= dt.datetime.now()]
    # surpresas.loc[surpresas.iloc[:,:1].isna().values]
    ]

surpresasCopom.iloc[:,:3].loc[surpresasCopom['s_3'].isna()]

# %% Retorno do CDI
cdi = bf.retornosDB(dt.datetime(2000,1,1), 'cCDI').set_index('dtref').rename({
    'retorno': 'CDI'
    }, axis = 1).applymap(math.exp)*100-100
cdi.index = [bf.int_date_to_str(x) for x in cdi.index]
cdi.plot()

# %% IBOV
ibov = bf.retornosDB(dt.datetime(2000,1,1), 'IBOV').set_index('dtref').rename({
    'retorno': 'IBOV'
    }, axis = 1).applymap(math.exp)*100-100
ibov.index = [bf.int_date_to_str(x) for x in ibov.index]
ibov = (ibov.loc[ibov.index.isin(cdi.index)] - cdi.loc[cdi.index.isin(ibov.index)].values).rename({
    'IBOV': 'IBOV menos CDI'
    }, axis = 1)
ibov.plot()

# %% USDBRL
usdbrl = bf.fetchTableMacro([4944]).pct_change().dropna()
usdbrl.columns = ['USDBRL']
usdbrl = usdbrl.loc[usdbrl.index >= surpresas.index[0]]*100
usdbrl = (usdbrl.loc[usdbrl.index.isin(cdi.index)] - cdi.loc[cdi.index.isin(usdbrl.index)].values).rename({
    'USDBRL': 'USDBRL menos CDI'
    }, axis = 1)
usdbrl.plot()

# %% Factor Data em pontos percentuais
nefin = bf.fetchTableMacro(
    bf.busca_nome_macro('Nefin').id_serie
    ).drop(
    'Risk Factor - Nefin - Risk_free - Livre de Risco', axis = 1
    )
nefin = 100*nefin.rename({key: key.split('Nefin - ')[-1] for key in nefin}, axis = 1)
nefin

# %% Check dos Dados
"""É fundamental encontrar uma fonte de variação exógena para avaliar o efeito 
de mudanças *inesperadas* nas taxas de juros. O primeiro sanity check é ver se 
identificamos nos dados surpresas sistemáticas nas datas das reuniões de Copom.

Como esperado, mesmo havendo variação expressiva em alguns poucos dias antes ou
depois dos dias imediatamente após a divulgação do comunicado, essa variação não
é sistemática. As reuniões do Copom acontecem a cada ~42-49 dias na maior parte
da amostra, então nenhuma divulgação de dado macroeconômico (IPCA tipicamente) 
acontece sistematicamente antes, depois ou no dia após o comunicado do Copom.
Apesar disso, ao longo de ~20 anos, todos os dados econômicos devem ser divulgados
algumas vezes antes ou logo depois do Copom e veremos surpresas relevantes da curva.

Como os gráficos abaixo sugerem, o grosso da variação acontece logo após o
comunicado do Copom. Ponto a favor do exercício.
"""

s_ = surpresasCopom.iloc[:,:3].copy()

indic = {}
for l in range(-2,2+1):
    if l == 0:
        indic[0] = [x for x in s_.index]
    else:
        indic[l] = []
        for i, x in enumerate(surpresas.index):
            if x in s_.index:
                indic[l] = indic[l] + [surpresas.index[i+l]]

indic = pd.DataFrame.from_dict(indic)

for col in indic:
    ax = surpresas.loc[indic[col].values].iloc[:,:3].plot(
        ylim = (-60,80), **faux.plot_params
        )
    ax.set_title(f'Surpresas de Copom Lagged em {col} dias')
    mp.show()

# %% OLS errado, viés pela endogeneidade
"""Inicialmente, alguém poderia querer encontrar o efeito de mudanças na curva
de juros rodando um OLS dos retornos contra mudanças na curva de juros. A despeito
dos problemas de endogeneidade envolvidos, vamos estimar isso para comparar os efeitos.

É difícil fazer um juízo a priori sobre a magnitude esperada dos efeitos de aumentos
de juros em cada portfolio, mas é razoável esperar efeitos negativos para retornos
de equities (para excessos de retorno de equities contra equities, nem tanto).
Felizmente, isso se confirma. Para o BRL, entretanto, temos um efeito completamente
contrário ao esperado. Aumentos de juros domésticos deveriam apreciar a moeda local,
não sugerir que o dólar fique mais caro em reais.

Aqui é claro o problema de endogeneidade por simultaneidade. Tanto esperamos que
o BRL aprecie (o USDBRL tenha retornos negativos) quando a taxa de juros aumenta
quanto esperamos que a taxa de juros aumente quando o BRL se deprecia (pela política
monetária, que lutará contra a inflação do pass-through). Adicionalmente, o dólar
pode se depreciar em momentos de turbulência política e nesses momentos o prêmio
de risco exigido aumente, abrindo a curva de juros. Claramente a estimativa por
OLS é eivada de vieses.
"""

i_ = surpresas.iloc[:,-3:].diff().dropna()

i_pca_weights = np.linalg.svd(i_.dropna().T @ i_.dropna())[-1].T
i_pca_weights /= i_pca_weights.sum(axis = 0)

i_pca = (i_.dropna() @ i_pca_weights)
i_pca.columns = [f'PCA{i+1} - Delta i_t' for i in range(i_pca.shape[1])]

print("Pesos do PCA1: ", i_pca_weights[:,0])

i_pca[['PCA1 - Delta i_t']].plot()

Covariates = i_pca[['PCA1 - Delta i_t']].copy().dropna()
Covariates = sm.add_constant(Covariates)

Y = pd.concat([ibov, nefin, usdbrl], axis = 1).dropna()

common_ind = [x for x in Covariates.index if x in Y.index]
Y = Y.loc[common_ind]
Covariates = Covariates.loc[common_ind]

OLSModels = {}
OLSResultados = []
OLSp_values = []

for col in Y:
    OLSModels[col] = sm.OLS(Y[[col]], Covariates).fit()
    OLSResultados.append(OLSModels[col].params.to_frame(col))
    OLSp_values.append(OLSModels[col].pvalues.to_frame(col))
    
OLSResultados = pd.concat(OLSResultados, axis = 1).T
OLSp_values = pd.concat(OLSp_values, axis = 1).T

print(80*"=")
print("Resultados:")
print(OLSResultados)
print(80*"=")
print("P-valores:")
print(OLSp_values)


FullResultados = pd.concat([
    OLSResultados[['PCA1 - Delta i_t']].rename({'PCA1 - Delta i_t': 'PCA1 - OLS'}, axis = 1),
    ], axis = 1)
FullResultados.rename({x: x.split(' - ')[-1] for x in FullResultados.index}).plot.bar(
    title = 'Efeito Estimado de aumento de 1 ponto percentual na curva curta',
    figsize = (14,7), fontsize = 14, grid = True, ylabel = '% '
    )



# %% Efeito de Surpresa de Copom nos portfolios de Risco
"""Uma primeira abordagem limpa de fatores exógenos para avaliar os efeitos de 
política monetária nos retornos dos ativos é verificar se os retornos dos portfolios
respondem a surpresas de política monetária. Essas surpresas são observadas nos dias
seguintes aos comunicados do Copom da seguinte forma:
    1) Com os dados de fechamento do dia da divulgação do comunicado do Copom,
    calcula-se o valor esperado para abertura dos contratos de vencimento curtos
    de DI avançando os valores de fechamento pelo CDI do dia do comunicado.
    2) Com o valor de abertura dos futuros e o valor calculado anteriormente,
    calculamos o tamanho da surpresa na taxa média dos futuros.

Essa surpresa é plausivelmente exógena e podemos utilizá-la para avaliar a sensibilidade
dos retornos de ativos/portfolios selecionados a movimentos nas taxas curtas por um
simples OLS.

As unidades dos coeficientes abaixo são "% de retorno por % de Surpresa"
"""

resultados = []
p_values = []
Covariates = surpresasCopom.iloc[:,:3].copy()/100

pca_weights = np.linalg.svd(Covariates.dropna().T @ Covariates.dropna())[-1].T

pca_weights /= pca_weights.sum(axis = 0)

pca = (Covariates.dropna() @ pca_weights)
pca.columns = [f'PCA{i+1}' for i in range(pca.shape[1])]

print("Pesos do PCA1: ", pca_weights[:,0])

Y = pd.concat([ibov, nefin, usdbrl], axis = 1)

for col in Y:

    for colX in Covariates:
        X_reg = sm.add_constant(Covariates[[colX]].dropna())
        model_ols = sm.OLS(
            Y.loc[X_reg.index, [col]],
            X_reg
        )
        model_ols.fit(cov_type='HC0')
        
        resultados.append(model_ols.fit(cov_type='HC0').params.to_frame((col, colX)))
        p_values.append(model_ols.fit(cov_type='HC0').pvalues.to_frame((col, colX)))
    
    X_reg = sm.add_constant(pca.dropna().iloc[:,:1])
    model_ols = sm.OLS(
        Y.loc[X_reg.index, [col]],
        X_reg
    )
    
    resultados.append(model_ols.fit(cov_type='HC0').params.to_frame((col, 'PCA')))
    p_values.append(model_ols.fit(cov_type='HC0').pvalues.to_frame((col, 'PCA')))
    
resultados = pd.concat(resultados, axis = 1)
p_values = pd.concat(p_values, axis = 1)

resultados_selected = round(resultados.reorder_levels([-1,0], axis = 1)['PCA'].dropna(), 2).T
p_values_selected = round(p_values.reorder_levels([-1,0], axis = 1)['PCA'].dropna(), 2).T

print(80*"=")
print("Resultados:")
print(resultados_selected)
print(80*"=")
print("P-valores:")
print(p_values_selected)


FullResultados = pd.concat([
    OLSResultados[['PCA1 - Delta i_t']].rename({'PCA1 - Delta i_t': 'PCA1 - OLS'}, axis = 1),
    resultados_selected[['PCA1']].rename({'PCA1': 'PCA1 - Surpresas'}, axis = 1)
    ], axis = 1)
FullResultados.rename({x: x.split(' - ')[-1] for x in FullResultados.index}).plot.bar(
    title = 'Efeito Estimado de aumento de 1 ponto percentual na curva curta',
    figsize = (14,7), fontsize = 14, grid = True, ylabel = '% '
    )

# %% Primeiros comentários
"""Os efeitos acima estão bem alinhados com o esperado.

Primeiro sobre o câmbio. Pensando em termos da UIP, é razoável que um aumento de
1pp no diferencial de juros a favor do Brasil venha associado de apreciação do
real em magnitude comparável.

Sobre o efeito em ações, esperamos efeitos fortes em ações para mudanças na curva
de juros porque aqui agem dois canais: earnings e fator de desconto. Tanto
descontamos os earnings da empresa a uma taxa maior, reduzindo seu valor presente,
quanto esperamos redução na atividade, diminuindo os earnings esperados. Não é 
surpresa, portanto, encontrarmos 5-6% de efeito negativo para o IBOV ou o retorno
de mercado em excesso ao CDI.

Para os fatores de risco de equities, entretanto, estamos falando de excessos de
retorno comparando equities com equities, então os canais do parágrafo anterior
estão presentes nos dois retornos comparados, se anulando. Não é claro porque um
rankeamento de size, value, momentum ou iliquidez deveria incorporar mais fortemente os
canais de earnings ou fator de desconto no lado long ou no lado short de seus 
portfolios.
    Para o portfolio de Value existe maior discussão sobre sensibilidade a juros
    porque há "competição" entre os efeitos nas duas pontas do fator de Value(-Growth).
    Empresas de Value (alto Book/Mkt) seriam tipicamente empresas mais distressed,
    pras quais o canal de earnings causa mais prejuízo. Empresas de Growth (baixo
    Book/Mkt) seriam pensadas como mais sensíveis a juros, por terem earnings cres-
    centes no tempo.
    Essas interpretações são disputáveis (e bastante disputadas).
"""


# %% Hora do Rigobon
"""O método implementado anteriormente foca completamente nos dias de comunicados
do Copom porque acredita que apenas ali temos uma variável plausivelmente exógena,
mas, com isso, acaba jogando fora muita informação. De ~20 anos de dados, apenas
~170 observações são utilizadas na estimação. Talvez seja possível fazer melhor.
Uma primeira possibilidade é utilizar o método de Rigobon, que também explora dias
de anúncio para identificar os efeitos das variáveis via heteroscedasticidade da
variável relevante no anúncio.

Rigobon lida bastante bem o problema de endogeneidade por simultaneidade, permitindo
limpar da estimação os efeitos de antecipação da taxa de juros e que podem surgir
por outras variáveis relevantes. A hipótese básica é que, nos períodos de divulgação,
a única coisa que acontece é um aumento da variância da variável de interesse,
então os efeitos nas janelas de divulgação podem ser comparados nas janelas de
controle e identificar o efeito verdadeiro da variável.

Para replicar a especificação favorita do exercício anterior e garantir comparabilidade
apresentaremos apenas para o PCA dos futuros curtos como taxa relevante, mas
podemos rodar os exercícios agrupados diariamente ou até semanalmente. A princípio,
rodaremos apenas a estimação diária.

A técnica de estimação é implementada por meio de instrumentos criados com a variação
da taxa de juros e a variação dos próprios ativos que estamos analisando. Para isso,
basta seeparar os períodos de tratamento e controle, 
"""

i_ = surpresas.iloc[:,-3:].diff().dropna()
i_.plot()

i_pca_weights = np.linalg.svd(i_.dropna().T @ i_.dropna())[-1].T
i_pca_weights /= i_pca_weights.sum(axis = 0)

i_pca = (i_.dropna() @ i_pca_weights)
i_pca.columns = [f'PCA{i+1} - Delta i_t' for i in range(i_pca.shape[1])]

print("Pesos do PCA1: ", i_pca_weights[:,0])

i_pca[['PCA1 - Delta i_t']].plot()

treatment_dates = [x for x in surpresasCopom.index]

control_N = 5
control_dates = []
for i, x in enumerate(i_pca.index):
    if x in treatment_dates:
        for l in reversed(range(control_N)):
            control_dates.append(i_pca.index[i-l-1])

ratio_treatment_control = len(treatment_dates)/len(control_dates)

Covariates = i_pca.loc[sorted(control_dates + treatment_dates),['PCA1 - Delta i_t']].copy()
# Covariates = sm.add_constant(Covariates)

Y = pd.concat([ibov, nefin, usdbrl], axis = 1).loc[sorted(control_dates + treatment_dates)]

# %%
variance_comparison = pd.concat([
    pd.concat([100*Covariates, Y], axis = 1).loc[control_dates].std().to_frame('Dias Ordinários'),
    pd.concat([100*Covariates, Y], axis = 1).loc[treatment_dates].std().to_frame('Dias de Comunicado')
    ], axis = 1)**2
variance_comparison

# %%

# number of estimated parameters:
L = Covariates.shape[1]
L = Covariates.shape[1] + Y.shape[1]

Instruments = pd.concat([Covariates, Y], axis = 1)
# Instruments.loc[treatment_dates] *= len(treatment_dates + control_dates)/(len(treatment_dates) - L)
# Instruments.loc[control_dates] *= -len(treatment_dates + control_dates)/(len(control_dates) - L)

Instruments.loc[treatment_dates] *= 1/(len(treatment_dates) - L)
Instruments.loc[control_dates] *= -1/(len(control_dates) - L)

# Se usarmos somente os dias de comunicado do Copom, teremos uma especificação de event-study
EventStudyModels = {}
EventStudyResultados = []
EventStudyp_values = []

for col in Y:
    EventStudyModels[col] = sm.OLS(Y[[col]], Covariates).fit()
    EventStudyResultados.append(EventStudyModels[col].params.to_frame(col))
    EventStudyp_values.append(EventStudyModels[col].pvalues.to_frame(col))
    
EventStudyResultados = pd.concat(EventStudyResultados, axis = 1).T
EventStudyp_values = pd.concat(EventStudyp_values, axis = 1).T

RigobonModels = {}
RigobonResultados = []
Rigobonp_values = []

for col in Y:
    RigobonModels[col] = IV2SLS(Y[[col]], Covariates, Instruments).fit()
    RigobonResultados.append(RigobonModels[col].params.to_frame(col))
    Rigobonp_values.append(RigobonModels[col].pvalues.to_frame(col))
    
RigobonResultados = pd.concat(RigobonResultados, axis = 1).T
Rigobonp_values = pd.concat(Rigobonp_values, axis = 1).T

print(80*"=")
print("Resultados:")
print(RigobonResultados)
print(80*"=")
print("P-valores:")
print(Rigobonp_values)


FullResultados = pd.concat([
    OLSResultados[['PCA1 - Delta i_t']].rename({'PCA1 - Delta i_t': 'PCA1 - OLS'}, axis = 1),
    EventStudyResultados.rename({'PCA1 - Delta i_t': 'PCA1 - EventStudy'}, axis = 1),
    resultados_selected[['PCA1']].rename({'PCA1': 'PCA1 - Surpresas'}, axis = 1),
    RigobonResultados.rename({'PCA1 - Delta i_t': 'PCA1 - Rigobon'}, axis = 1)
    ], axis = 1)
FullResultados.rename({x: x.split(' - ')[-1] for x in FullResultados.index}).plot.bar(
    title = 'Efeito Estimado de aumento de 1 ponto percentual na curva curta',
    figsize = (14,7), fontsize = 14, grid = True, ylabel = '% '
    )


# %% Extensões Possíveis:
"""
Rodar com dados semanais em vez de diários

Teste para instrumentos fracos
 - LEWIS, D. Robust Inference in Models Identified via Heteroskedasticity

External/Internal Instruments / Local Projections para o VAR
 - Mais afeito a modelos low frequency. Semanal já soa overstretch. Cabível,
 é claro, mas overstretch
"""
