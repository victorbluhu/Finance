# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:45:58 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Trading Strategy')

import numpy as np
import pandas as pd
import datetime as dt
import math

import statsmodels.api as sm

# %% Get and organize data
vix = pd.read_csv(r'Data\VIX.csv', index_col = 0, parse_dates=True)
vix.columns = pd.MultiIndex.from_product([vix.columns, ['VIX']])

spx = pd.read_csv(r'Data\SPX.csv', index_col = 0, parse_dates=True)
spx.columns = pd.MultiIndex.from_product([spx.columns, ['SPX']])

data = pd.concat([vix, spx.pct_change()], axis = 1)['CLOSE']

# Checa os dados
print(data.isna().sum())
data.loc[data.isna()['VIX']]
data.loc[data.isna()['SPX']]

# é importante checar o que é missing em cada série. Alguns são feriados, mas nem todos.
# Solução sloppy intermediária: utilizar dados de 2006-06 a 2022-04. É a maior faixa contínua sem missings
# Além disso, pulamos a troca de metodologia do VIX (de 2004)
data = data.loc[
    (data.index >= dt.datetime(2006,6,1)) &
    (data.index < dt.datetime(2022,5,1))
    ]


# %% 
"""O VIX é um índice construído com opções do SPX de tal forma que ele reflita
a raiz da variância esperada implícita do retorno total do SPX pros próximos 30 dias.
Por simplicidade, compararemos os retornos dos 22 pregões seguintes.

Além disso, o VIX é apresentado em termos anuais, então é preciso dividir por sqrt(12).

O ideal seria estimadores da variância realizada do SPX que levassem em conta retornos
intradiários, para se aproximar do processo de difusão suposto pelo VIX, mas não 
dispomos desses dados. Por causa disso, só tomaremos a variância realizada dos 
retornos diários e reescalaremos pelo número de dias utilizado na janela escolhida.
Isso funciona para processos de difusão tradicionais, já que a variância escala
linearmente no tempo.*

Uma importante consideração é que o VIX tem um prêmio médio com respeito à variância
realizada (e, obviamente, com a variância realizada futura). Isso é esperado e
prejudica comparações diretas, mas ainda assim permite comparações entre as séries.

*: neste exercício, desconsideraremos problemas pela concavidade introduzida nas
estimações de volatilidade, já que seus possíveis ajustes são modelo dependentes.
"""

N = 22
compara_vols = pd.concat([
    data['VIX']/100/math.sqrt(12),
    data['VIX'].shift(N-1)/100/math.sqrt(12),
    data['SPX'].rolling(N).std()*math.sqrt(N),
    data['SPX'].rolling(N).std().shift(-N)*math.sqrt(N)
    ], axis = 1).dropna()**2
compara_vols.columns = ['VIX', 'VIX Lagged', 'Var Realizada', 'Var Futura Realizada']

compara_vols[[
    'VIX', 'Var Futura Realizada'
    ]].plot(title = 'VIX e Variância Futura Realizadas')

compara_vols.mean()*100

# %% Será que o VIX prevê bem a variância futura realizada?
"""Um primeiro exercício ordinário para checar se o VIX é um preditor da variância
realizada futura é comparar os dois. Uma regressão linear da Var Futura contra o
VIX é um exercício preliminar em que se esperaria alpha=0 e beta=1, mas limitado:
    1) a primeira limitação é a possibilidade de um grande prêmio de risco para volatilidade.
        O VIX usa preços de mercado, que envolvem a ponderação de risco dos agentes.
        Ele não utiliza diretamente as probabilidades físicas de cada estado do
        mundo, mas sim uma distorção delas (pelo fator de desconto estocástico).
    2) O VIX é pontuado por momentos de intenso, mas breves nervosismos. Isso faz
        com que uma regressão linear acabe dando peso muito forte para seus outliers,
        já que é estimadores de correlação/covariância são muito sensíveis a outliers.

A despeito dos potenciais problemas, os coeficientes ficam dentro do esperado para
a hipótese de que o VIX é um bom preditor (alpha = 0 e beta = 1), que termina não
descartada. Aqui não queremos modelar a variância futura realizada, então por ora
pararemos nos testes de hipótese típicos de OLS.
""" 
# """
#     2) a segunda limitação decorre, por construção, da desigualdade de Jensen.
#         O VIX é construído como um estimador da variância total dos próximos 30 dias.
#         A Var Futura Realizada é a expectativa da
# """

Y = compara_vols[['Var Futura Realizada']].copy()
X = compara_vols[['VIX']].copy()
X = sm.add_constant(X)

model_ols = sm.OLS(Y,X)
results = model_ols.fit(cov_type='HC0')

results.summary()

results.params

hypotheses = 'const = 0, VIX = 1'
t_test = results.t_test(hypotheses)
wald_test = results.wald_test(hypotheses)
print(t_test)
print(wald_test)

# %% Segundo sanity check simples:
"""Um segundo sanity check da relação entre VIX e volatilidade futura que endereça
o problema acima de a regressão ser muito sensível a outliers é checar se o VIX
acerta bem o sinal da variância futura realizada. Isso é possível de ser feito por
diversas modelagens de classificação, mas uma simples tabela de contingência já
aponta bastante potencial.

A conclusão é que o VIX é um excelente preditor para queda na variância realizada.
Adicionalmente, vemos que é raro o VIX ser menor do que a variância realizada.
Isso é sugestivo de que há mesmo um prêmio de volatilidade na medida do VIX.
"""

fut_maior = compara_vols['Var Futura Realizada'] > compara_vols['Var Realizada']
sinal_vix_maior = compara_vols['VIX'] > compara_vols['Var Realizada']

expec_incondicional = fut_maior.value_counts(
    normalize=True).mul(100).to_frame('Incondicional')

vix_maior = fut_maior.loc[
    sinal_vix_maior
].value_counts(normalize=True).mul(100).to_frame('VIX > Var Realizada (%)')

vix_menor = fut_maior.loc[
    ~sinal_vix_maior
].value_counts(normalize=True).mul(100).to_frame('VIX < Var Realizada (%)')

vix_vs_vol_futura = round(
    pd.concat([
        expec_incondicional,
        vix_maior,
        vix_menor], axis = 1),
    1).rename(
        {True: 'Var Futura > Passada',False: 'Var Futura < Passada'}).iloc[::-1]
vix_vs_vol_futura.index.name = 'Resultado'
vix_vs_vol_futura.columns.name = 'Sinal'

vix_vs_vol_futura.loc['T'] = [
    compara_vols.shape[0], 
    (compara_vols['VIX'] > compara_vols['Var Realizada']).sum(),
    compara_vols.shape[0] - (compara_vols['VIX'] > compara_vols['Var Realizada']).sum()
    ]
print(vix_vs_vol_futura)

# %% É possível refinar o sinal para avaliar melhor a volatilidade futura?
"""Uma exploração maior do sinal do VIX vs a variância passada é possível refinando
o sinal. Uma maneira natural de refinar o sinal anterior é incluir o prêmio mediano 
de variância do VIX na comparação. Outra possibilidade é incluir alguma modelagem
para a tendência esperada (local ou global) do VIX e comparar o nível do VIX contra
essa tendência.

Há uma grande margem de arbitrariedade em exercícios como os desenhados acima.
O anterior parece mais limpo do que os outros, mas até ali é possível sofisticar
um pouco. Por exemplo, podemos considerar o ajuste pelo prêmio de volatilidade
que o VIX tem sobre a variância realizada.
O segundo requer uma definição formal para 'tendência esperada' e também uma margem
de tolerância. Essa tendência esperada é extremente sensível à modelagem, é claro,
então tentaremos ser parcimoniosos. Uma média ou mediana móvel para os últimos VIX
é uma candidata inicial simples. Outro é X desvios-padrão acima da média/mediana.
    Uma sofisticação disso seria um modelo para o VIX
    que mimetizasse um modelo de heteroscedasticidade condicional para os retornos,
    a ser estimado em janela móvel. Isso fica para um próximo momento.
    
Por ora, a melhor regra separadora de regimes de mais alta ou mais baixa variância
é a comparação da variância realizada com o prêmio mediano do VIX dos últimos 126 pregões.
"""

sinal_vix_maior = compara_vols['VIX'] > compara_vols['Var Realizada']
sinal_vix_maior.name = 'VIX > Var Realizada'

T_premio_vol = 126
compara_vols['PremioVIX'] = (
    compara_vols['VIX Lagged'] - compara_vols['Var Realizada']
    ).rolling(T_premio_vol).median()
compara_vols['PremioVIX'].plot(title = f'Prêmio mediano móvel do VIX: {T_premio_vol} pregões')

sinal_vix_maior_premio = compara_vols['VIX'] > compara_vols['Var Realizada'] + compara_vols['PremioVIX']
sinal_vix_maior_premio.loc[compara_vols['PremioVIX'].isna()] = np.nan
sinal_vix_maior_premio.name = f'VIX > Var Realizada + Prêmio {T_premio_vol}'

sinal_vix_maior_mediana_VIX = compara_vols['VIX'] > compara_vols['VIX'].rolling(T_premio_vol).median()
sinal_vix_maior_mediana_VIX.loc[compara_vols['VIX'].rolling(T_premio_vol).median().isna()] = np.nan
sinal_vix_maior_mediana_VIX.name = f'VIX > VIX Mediano {T_premio_vol}'

K = 1.5
sinal_vix_maior_mediana_std_VIX = compara_vols['VIX'] > (
    compara_vols['VIX'].rolling(T_premio_vol).median() + K*compara_vols['VIX'].rolling(T_premio_vol).std())
sinal_vix_maior_mediana_std_VIX.loc[compara_vols['VIX'].rolling(T_premio_vol).median().isna()] = np.nan
sinal_vix_maior_mediana_std_VIX.name = f'VIX > VIX Mediano + 2 Std {T_premio_vol}'

sinais = pd.concat([
    sinal_vix_maior,
    sinal_vix_maior_premio,
    sinal_vix_maior_mediana_VIX,
    sinal_vix_maior_mediana_std_VIX
    ], axis = 1).dropna()

fut_maior = fut_maior.loc[sinais.index]

vix_vs_vol_futura = pd.concat([
    fut_maior.value_counts(normalize=True).mul(100).to_frame('Incondicional'),
    pd.concat([pd.concat([
        fut_maior.loc[sinais[col]].value_counts(normalize=True).mul(100).to_frame(col),
        fut_maior.loc[
            sinais[col].apply(lambda x : not x)
            ].value_counts(normalize=True).mul(100).to_frame(col.replace('>','<'))
        ], axis = 1) for col in sinais], axis = 1)
    ], axis = 1).rename(
        {True: 'Var Futura > Passada',False: 'Var Futura < Passada'}
        ).iloc[::-1]
vix_vs_vol_futura.index.name = 'Resultado'
vix_vs_vol_futura.columns.name = 'Sinal'

T_list = [compara_vols.shape[0]]
for col in sinais:
    T_list += [
    sinais[col].sum(),
    sinais[col].apply(lambda x : not x).sum()
    ]

vix_vs_vol_futura.loc['T'] = T_list
vix_vs_vol_futura = round(vix_vs_vol_futura).T
print(round(vix_vs_vol_futura))


# %% Além de previsão de volatilidade, há algum padrão na média dos retornos?
"""Comprada a hipótese de que o VIX mede mesmo a volatilidade dos retornos, uma
pergunta natural é se períodos de aumento do VIX ou alto VIX estão associados a
diferentes níveis de retorno médios futuros. Isso pode ser explorado por diversas
sinais:
    1) VIX maior do que a variância realizada passada
        1.1) VIX maior do que a variância realizada passada + o prêmio mediano de
            volatilidade do VIX
    2) VIX mais alto do que alguma tendência esperada (local ou global)

Há uma grande margem de arbitrariedade em exercícios como os desenhados acima.
O primeiro parece mais limpo do que os outros, mas até ali é possível sofisticar
um pouco. Por exemplo, podemos considerar algum ajuste para o prêmio de volatilidade
que o VIX tem sobre a variância realizada.
O segundo requer uma definição formal para 'tendência esperada' e também uma margem
de tolerância. Essa tendência esperada é extremente sensível à modelagem, é claro,
então tentaremos ser parcimoniosos. Uma média ou mediana móvel para os últimos VIX
é uma candidata inicial simples. Outro é X desvios-padrão acima da média/mediana.
    Uma sofisticação disso seria um modelo para o VIX
    que mimetizasse um modelo de heteroscedasticidade condicional para os retornos,
    a ser estimado em janela móvel. Isso fica para um próximo momento.
"""



nonsinais = sinais.applymap(lambda x : not x).dropna()

retorno_acumulado = pd.concat([
    (1 + data['SPX']).cumprod().pct_change(i).shift(-i) for i in range(1,N+1)
    ], axis = 1)*100
retorno_acumulado.columns = [f'Retorno Acumulado em {i} dias' for i in range(1,N+1)]
retorno_acumulado = retorno_acumulado.loc[sinais.index]


sin = sinais.columns[0]
retorno_medio = pd.DataFrame()
for sin in sinais:
    
    temp = pd.concat([
        retorno_acumulado.loc[sinais[sin]].mean(axis = 0).to_frame('Mean'),
        retorno_acumulado.loc[sinais[sin]].std(axis = 0).to_frame('Std'),
        (retorno_acumulado.loc[sinais[sin]].mean(axis = 0)/
         retorno_acumulado.loc[sinais[sin]].std(axis = 0)).to_frame('SR')
    ], axis = 1)
    temp.columns = pd.MultiIndex.from_product([temp.columns, [sin]])
    
    complementar = pd.concat([
        retorno_acumulado.loc[nonsinais[sin]].mean(axis = 0).to_frame('Mean'),
        retorno_acumulado.loc[nonsinais[sin]].std(axis = 0).to_frame('Std'),
        (retorno_acumulado.loc[nonsinais[sin]].mean(axis = 0)/
         retorno_acumulado.loc[nonsinais[sin]].std(axis = 0)).to_frame('SR')
    ], axis = 1)
    complementar.columns = pd.MultiIndex.from_product([complementar.columns, [sin.replace('>', '<')]])
    
    retorno_medio = pd.concat([retorno_medio, temp, complementar], axis = 1)
    