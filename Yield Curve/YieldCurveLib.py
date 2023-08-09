# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:57:31 2023

@author: Victor
"""

#%% Preâmbulo de importação

import pandas as pd
import datetime as dt
import numpy as np
from math import log, exp, sqrt
import scipy.linalg


# %% Funções Específicas de Curva GSK do Fed e Anbima

def cria_pasta(folder):
    from os.path import exists
    from os import mkdir
    
    if not exists(folder):
        try:
            mkdir(folder)
        except OSError:
            print ("Creation of the directory %s failed" % folder)
        else:
            print ("Successfully created the directory %s " % folder)
    else:
        print("The directory alredy exists.")

def atualizaParametrosGSKdoFED(pasta = 'Data\GSKdata'):
    
    try:
        print(80*"=")
        
        for i, key in enumerate(['nominal', 'tips']):
            
            puxaParametrosGSKdoFED(key , pasta)
            
            print("Tudo certo com os parâmetros da curva", key)
            print(80*"=")
        
        return True
    except:
        return False
    
def puxaParametrosGSKdoFED(tipo = 'nominal', pasta = 'data'):
    
    cria_pasta(pasta)
    
    url_dict = {
            'nominal': r'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv',
            'tips': r'https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv'
            }
    
    import requests
    url = url_dict[tipo]
    r = requests.get(url)
    
    from os.path import join
    file_name = join(pasta, tipo +'-GSK.csv')
    try:
        with open(file_name, 'wb') as outfile:
            outfile.write(r.content)
        return file_name
    except Exception as e:
        print(e)
        print("Não deu certo. Checar")

def montaParametrosGSKdoFED(tipo = 'nominal', pasta = 'Data\GSKdata'):
    
    from os.path import join
    file_name = join(pasta, tipo +'-GSK.csv')
    
    if tipo == 'nominal':
        data = pd.read_csv(file_name, skiprows=range(9), usecols=['Date', 'BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']).dropna()
    elif tipo == 'tips':
        data = pd.read_csv(file_name, skiprows=range(18), usecols=['Date', 'BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2'])
        data['TAU2'] = data['TAU2'].fillna(999999)
        data['BETA3'] = data['BETA3'].fillna(0)
        data = data.dropna()
    
    data['Date'] = data['Date'].apply(lambda x : [int(y) for y in x.split("-")]).apply(lambda x : dt.datetime(x[0],x[1],x[2]))
    data.set_index('Date', inplace = True)

    data.columns = pd.MultiIndex.from_product([['Params'], data.columns])

    return data['Params']

def fetchCurvaGSKParametrizada(
    tipo = 'nominal', pasta = 'Data\GSKdata',
    mensal = False, max_maturity = 360,
    min_date = dt.datetime(1950,1,1),
    max_date = dt.datetime(2100,12,31)
):  
    
    params = montaParametrosGSKdoFED(tipo = tipo)

    return GSK_Curve(
            params.loc[
                    (params.index >= min_date) & (params.index <= max_date)
                    ],
            mensal = mensal, max_maturity = max_maturity, Anbima = 'BR' in tipo
            )


# %% Workhorse básico para Yield Curves

class YieldCurve:
    
    def __init__(self,
                 yields_df = None, modo = 'Yields',
                 mensal = False, fixed_horizon_em_meses = 1,
                 FRA_horizon_em_meses = 12, fixed_rate_horizon = 1,
                 inicializar_tudo = True
                ):
        
        temp = yields_df.copy()
        
        self.mensal = mensal
        
        temp, self.fixed_horizon_em_meses, self.fixed_horizon = self.ajustesDeMensalizacaoYieldsDF(
            temp, fixed_horizon_em_meses)
        
        
        self.fixed_horizon_anual_em_meses = 12*self.fixed_horizon_em_meses
        self.fixed_horizon_anual = 12*self.fixed_horizon
        
        if modo.lower() in ['yields', 'yield']:
            self.Yields = temp.copy()
        elif modo.lower() in ['logyields', 'log_yields', 'logyield', 'log_yield']:
            self.LogYields = temp.copy()
        
        self.dates = temp.index
        self.max_maturity = temp.columns[-1]
        
        self.FRA_horizon = FRA_horizon_em_meses
        self.fixed_rate_horizon = fixed_rate_horizon
        
        self.inicializar_tudo = inicializar_tudo
        
        if inicializar_tudo:
            self.createCurves()
    
    def ajustesDeMensalizacaoYieldsDF(self, YieldsDF, fixed_horizon_em_meses):
        
        if self.mensal:
            print("Mensalizando os dados.")
            fixed_horizon_em_meses = fixed_horizon_em_meses
            fixed_horizon = fixed_horizon_em_meses
        else:
            fixed_horizon_em_meses = fixed_horizon_em_meses
            fixed_horizon = 21*fixed_horizon_em_meses
            
        return (
                YieldsDF.resample('M').last() if self.mensal else YieldsDF
                ), fixed_horizon_em_meses, fixed_horizon
    
    def createCurves(self):
        if hasattr(self, 'Yields'):
            self.createLogYields()
        elif hasattr(self, 'LogYields'):
            self.createYields()
        else:
            # Seria o caso de iniciar a curva com dados de FRA? 
            print("Ainda não criamos uma iniciação para esse tipo de dado. Checar o que está acontecendo.")
       
        self.createFixedRate()
        self.createDFs()
        self.createFRAs()
        self.createHPRs()
        self.createHPRsAnuais()
        self.createRXs()
        self.createRXsCorretos()
        self.createRXsAnuais()
        self.createRXsAnuaisCorretos()
    
    def getLogYieldsWithYields(self, yields_df):
        return (1 + yields_df/100).applymap(log)*100
    
    def getYieldsWithLogYields(self, log_yields_df):
        return (log_yields_df/100).applymap(exp)*100 - 100
    
    def getLoGDFsWithLogYields(self, log_yields_df):
        return (
            -log_yields_df/100 * np.arange(1,1+log_yields_df.columns[-1]).reshape((1,-1))/12
        )
    
    def getLogYieldsWithLogDFs(self, log_DFs_df):
        return - log_DFs_df * 12 / np.array([[x for x in log_DFs_df]]) * 100
    
    def getDFsWithLogDFs(self, log_dfs):
        return log_dfs.applymap(exp)
    
    def getLogFRAsWithLogDFsAndFRA_horizon(self, LogDFs, FRA_horizon):
        return (
            (LogDFs - LogDFs.shift(-FRA_horizon, axis = 1))/(FRA_horizon/12) * 100
        )
    
    # Calcula holding period return anualizados: compra título de maturidade n em t e vende com maturidade n-fixed_horizon_em_meses em t+fixed_horizon
    # Este é um retorno alinhado na linha da montagem do deal, mas só é conhecido em t+1
    # Isso parece escroto, e pode até ser má prática, mas ajuda para conseguir fazer modelagem de retornos em t+1 com dados de t
    def getLogHPRsWithLogDFsAndFixedHorizonMesesAndFixedHorizon(
            self, LogDFs, fixed_horizon_em_meses, fixed_horizon, dados_mensais = True):
        
        lnDF_t1 = LogDFs.shift(fixed_horizon_em_meses, axis = 1).shift(-fixed_horizon, axis = 0)
        lnDF_t1.loc[
            lnDF_t1.index[:-fixed_horizon],fixed_horizon_em_meses
        ] = 0
        
        if dados_mensais:
            return ( lnDF_t1 - LogDFs ) / fixed_horizon * 12 * 100
        else:
            return ( lnDF_t1 - LogDFs ) / (fixed_horizon/21) * 12 * 100
    
    # Calcula Excess Return entre um holding period return e outro holding period return de mesmo fixed horizon
    # Como default, usaremos a taxa mais curta para calcular a fixed rate    
    def getLogRXsWithLogHPRsAndReferenceMonth(self, LogHPRs, ref_month):
        
        return LogHPRs - LogHPRs[[ref_month]].values
    
    def getLogRXsAnuaisWithLogHPRsAnuaisAndLogHPRsMensaisParaReferenceMonth(self, LogHPRsAnuais, LogHPRsMensais, ref_month):
        
        return LogHPRsAnuais - LogHPRsMensais[[ref_month]].rolling(self.fixed_horizon_anual).mean().values
    
    #### Calcula inclinação de alguma das curvas
    def getInclinacaoWithDF(self, df_juros, v_curta, v_longa):
        
        return (df_juros[v_longa] - df_juros[v_curta]).to_frame(str(v_curta) + 'v' + str(v_longa))
    
    def getInclinacao(self, string, v_curta, v_longa):
        
        if not hasattr(self, string):
            print("Não temos nada salvo como atributo em " + string)
            return None
        else:
            return self.getInclinacaoWithDF(getattr(self, string), v_curta, v_longa).rename(
                {str(v_curta) + 'v' + str(v_longa): string + ' - ' + str(v_curta) + 'v' + str(v_longa)}, axis = 1
            )
    
    def getLogDFsInterpoladosWithLogDFs(self, LogDFs, mat):
        
        if mat < 0:
            print('Maturidade negativa. Meça seus parâmetros, parça.')
        
        temp = LogDFs.copy()
        temp[0] = 0
        temp = temp[sorted(temp)]
        
        max_maturity = temp.columns[-1]
        
        if mat in temp.columns:
            return temp[mat]
        elif mat > max_maturity:
            max_v = temp.columns[-1]
            min_v = temp.columns[-2]
        else:
            max_v = [x for x in temp.columns if x >= mat][0]
            min_v = [x for x in temp.columns if x <= mat][-1]
            
        return (LogDFs[min_v] * (max_v - mat) + LogDFs[max_v] * (mat - min_v)) / (max_v - min_v)
    
    # Lida com colunas com missing-data na base da brute-force
    # Uma possibilidade é rodar a função em loop para padrões de missing data diferentes
    # A ver como isso é possível. Talvez seja.
    def getLogDFsInterpoladosWithLogDFsDataAData(self, LogDFs, mat):
        
        return pd.concat([
                self.getLogDFsInterpoladosWithLogDFs(LogDFs.loc[[i]].dropna(axis = 1), mat) for i in LogDFs.index
                ], axis = 0)
        
    
    def getInterpolaLogYields(self, mat, modo = 'ColunasCheias'):
        if not hasattr(self, "LogDFs"):
            print("Não temos nada salvo como Log Discount Factors nos atributos.")
            self.createLogDFs()
        if modo == 'ColunasCheias':
            return self.getLogYieldsWithLogDFs(
                self.getLogDFsInterpoladosWithLogDFs(self.LogDFs, mat).to_frame(mat)
            )
        else:
            return self.getLogYieldsWithLogDFs(
                self.getLogDFsInterpoladosWithLogDFsDataAData(self.LogDFs, mat).to_frame(mat)
            )
    
    def getCurvaJurosDf(self, df_juros, date_indices):
        
        # Checa index
        correct_index = [date for date in date_indices if date in df_juros.index]
        
        return df_juros.loc[correct_index].rename({dt_: dt_.date() for dt_ in correct_index}).T
    
    def getCurva(self, string, date_indices):
        if not hasattr(self, string):
            print("Não temos nada salvo como atributo em " + string)
            return None
        else:
            return self.getCurvaJurosDf(getattr(self, string), date_indices)
    
    def plotCurvaDF(self, df_juros, date_indices, title_string, params = {
        'figsize': (14,7), 'grid': True,
        'lw': 2, 'fontsize': 14
    }, max_vertex = 120):
        
        temp_df = self.getCurvaJurosDf(df_juros, date_indices)
        return temp_df.loc[temp_df.index <= max_vertex].plot(
            title = title_string,
            xlabel = 'Meses',
            **params
            )
    
    def plotCurva(self, string, date_indices, params = {
        'figsize': (14,7), 'grid': True,
        'lw': 2, 'fontsize': 14
    }, max_vertex = 120):
        
        if not hasattr(self, string):
            print("Não temos nada salvo como atributo em " + string)
            return None
        else:
            return self.plotCurvaDF(
                getattr(self, string), date_indices,
                f'Estrutura a Termo de {string}', params, max_vertex = 120)
    
    def createLogYields(self):
        if hasattr(self, 'LogYields'):
            print("Já tinhamos LogYields antes.")
        else:
            if not hasattr(self, 'Yields'):
                print("Não temos LogYields nem Yields. Cheque a inicialização.")
            else:
                self.LogYields = self.getLogYieldsWithYields(self.Yields)
    
    def createYields(self):
        if hasattr(self, 'Yields'):
            print("Já tinhamos Yields antes.")
        else:
            if not hasattr(self, 'LogYields'):
                print("Não temos LogYields nem Yields. Cheque a inicialização.")
            else:
                self.Yields = self.getYieldsWithLogYields(self.LogYields)

    def createLogFixedRate(self):
        if not hasattr(self, 'LogYields'):
            self.createLogYields()
        self.LogFixedRate = self.LogYields[[self.fixed_rate_horizon]]
        
    def createFixedRate(self):
        if not hasattr(self, 'LogFixedRate'):
            self.createLogFixedRate()
        self.FixedRate = self.getYieldsWithLogYields(self.LogFixedRate)
    
    def createLogDFs(self):
        if not hasattr(self, 'LogYields'):
            print("Não tínhamos calculado os LogYields antes. Calcularemos agora.")
            self.createLogYields()
        self.LogDFs = self.getLoGDFsWithLogYields(self.LogYields)
    
    def createDFs(self):
        if not hasattr(self, 'LogDFs'):
            print("Não tínhamos calculado os LogDFs antes. Calcularemos agora.")
            self.createLogDFs()
        self.DFs = self.getDFsWithLogDFs(self.LogDFs)
    
    def createLogFRAs(self):
        if not hasattr(self, 'DFs'):
            print("Não tínhamos calculado os Log Discount Factors antes. Calcularemos agora.")
            self.createDFs()
        self.LogFRAs = self.getLogFRAsWithLogDFsAndFRA_horizon(self.LogDFs, self.FRA_horizon)
    
    def createFRAs(self):
        if not hasattr(self, 'LogFRAs'):
            print("Não tínhamos calculado os LogFRAs antes. Calcularemos agora.")
            self.createLogFRAs()
        self.FRAs = self.getYieldsWithLogYields(self.LogFRAs)
    
    def createLogHPRs(self):
        if not hasattr(self, 'LogDFs'):
            print("Não tínhamos calculado os Log Discount Factors antes. Calcularemos agora.")
            self.createLogDFs()
        self.LogHPRs = self.getLogHPRsWithLogDFsAndFixedHorizonMesesAndFixedHorizon(
            self.LogDFs, self.fixed_horizon_em_meses, self.fixed_horizon
        )
        
    def createHPRs(self):
        if not hasattr(self, 'LogHPRs'):
            print("Não tínhamos calculado os LogHPRs antes. Calcularemos agora.")
            self.createLogHPRs()
        self.HPRs = self.getYieldsWithLogYields(self.LogHPRs)
    
    def createLogHPRsAnuais(self):
        if not hasattr(self, 'LogDFs'):
            print("Não tínhamos calculado os Log Discount Factors antes. Calcularemos agora.")
            self.createLogDFs()
        self.LogHPRsAnuais = self.getLogHPRsWithLogDFsAndFixedHorizonMesesAndFixedHorizon(
            self.LogDFs, self.fixed_horizon_anual_em_meses, self.fixed_horizon_anual
        )
        
    def createHPRsAnuais(self):
        if not hasattr(self, 'LogHPRsAnuais'):
            print("Não tínhamos calculado os LogHPRsAnuais antes. Calcularemos agora.")
            self.createLogHPRsAnuais()
        self.HPRsAnuais = self.getYieldsWithLogYields(self.LogHPRsAnuais)
    
    def createLogRXs(self):
        if not hasattr(self, 'LogHPRs'):
            print("Não tínhamos calculado os LogHPRs antes. Calcularemos agora.")
            self.createLogHPRs()
        self.LogRXs = self.getLogRXsWithLogHPRsAndReferenceMonth(self.LogHPRs, self.fixed_rate_horizon)
    
    def createRXs(self):
        if not hasattr(self, 'LogRXs'):
            print("Não tínhamos calculado os LogRXs antes. Calcularemos agora.")
            self.createLogRXs()
        self.RXs = self.getYieldsWithLogYields(self.LogRXs)
    
    def createRXsCorretos(self):
        if not hasattr(self, 'HPRs'):
            print("Não tínhamos calculado os HPRs antes. Calcularemos agora.")
            self.createHPRs()
        self.RXsCorretos = self.HPRs - self.HPRs[[self.fixed_rate_horizon]].values
        
    def createLogRXsAnuais(self):
        if not hasattr(self, 'LogHPRs'):
            print("Não tínhamos calculado os LogHPRs antes. Calcularemos agora.")
            self.createLogHPRs()
        if not hasattr(self, 'LogHPRsAnuais'):
            print("Não tínhamos calculado os LogHPRsAnuais antes. Calcularemos agora.")
            self.createLogHPRsAnuais()
        self.LogRXsAnuais = self.getLogRXsAnuaisWithLogHPRsAnuaisAndLogHPRsMensaisParaReferenceMonth(self.LogHPRsAnuais, self.LogHPRs, self.fixed_rate_horizon)
    
    def createRXsAnuais(self):
        if not hasattr(self, 'LogRXsAnuais'):
            print("Não tínhamos calculado os LogRXsAnuais antes. Calcularemos agora.")
            self.createLogRXsAnuais()
        self.RXsAnuais = self.getYieldsWithLogYields(self.LogRXsAnuais)
        
    def createRXsAnuaisCorretos(self):
        if not hasattr(self, 'HPRs'):
            print("Não tínhamos calculado os HPRs antes. Calcularemos agora.")
            self.createHPRs()
        if not hasattr(self, 'HPRsAnuais'):
            print("Não tínhamos calculado os HPRsAnuais antes. Calcularemos agora.")
            self.createHPRsAnuais()
        self.RXsAnuaisCorretos = self.HPRsAnuais - self.HPRs[[self.fixed_rate_horizon]].rolling(self.fixed_horizon_anual).mean().values


# %% Agregador de Yield Curves. Serve para comparar Curvas nominais e reais

class YieldCurveSet:
    
    def __init__(self):
        
        pass
    
    def addYieldCurve(self, YieldCurveObject, YieldCurveName):
        
        if hasattr(self, YieldCurveName):
            print("Nome de Atributo existente. Sobrescrevendo.")
            
        setattr(self, YieldCurveName, YieldCurveObject)
    
    def addBreakevenCurve(
            self, NominalYieldCurveName, TipsYieldCurveName, name = None):
        
        if name is None:
            name = 'breakeven'
            
        if hasattr(self, name):
            print("Nome de Atributo existente. Sobrescrevendo.")
        
        if not hasattr(self, NominalYieldCurveName) or not hasattr(self, TipsYieldCurveName):
            print("Nome de Curvas não existentes. Reveja a função")
        else:
            nominal = getattr(self, NominalYieldCurveName)
            tips = getattr(self, TipsYieldCurveName)
            
            setattr(
                self,
                name,
                YieldCurve(
                    yields_df = nominal.LogYields - tips.LogYields,
                    modo = 'LogYields',
                    mensal = nominal.mensal,
                    fixed_horizon_em_meses = nominal.fixed_horizon_em_meses,
                    FRA_horizon_em_meses = nominal.FRA_horizon,
                    fixed_rate_horizon = nominal.fixed_rate_horizon,
                    inicializar_tudo = nominal.inicializar_tudo)
                )
    
    def getCurva(self, string, date_indices, curve_names = []):
        
        for s_ in curve_names:
            if not hasattr(self, s_):
                print(f"Não temos nada salvo como {s_} no YieldCurveSet")
                return None
        else:
            
            Curva = pd.DataFrame()
            
            for s_ in curve_names:
                
                temp = getattr(self, s_).getCurvaJurosDf(
                    getattr(getattr(self, s_), string), date_indices)
                temp.columns = pd.MultiIndex.from_product(
                    [temp.columns, [s_]], names = ['dtref', 'curva']
                    )
                
                Curva = pd.concat([Curva, temp], axis = 1)
            
            return Curva
    
    def plotCurvasDate(self, string, date_indices, title_string, params = {
        'figsize': (14,7), 'grid': True,
        'lw': 2, 'fontsize': 14
    }, max_vertex = 120):
        
        temp_df = self.getCurva(
            string, date_indices,
            curve_names = ['nominal','tips','breakeven'])
        
        cols0 = np.unique([x[0] for x in temp_df])
        
        return {date: temp_df[date].loc[temp_df.index <= max_vertex].plot(
                title = title_string + f' - {date}',
                xlabel = 'Meses',
                **params) for date in cols0
            }
    

#%% Classe de Curvas GSK
        
class GSK_Curve(YieldCurve):
        
    def __init__(
            self, params_df, max_maturity = 360,
        # Parâmetros comuns à classe de YieldCurves
            mensal = False, fixed_horizon_em_meses = 1,
            FRA_horizon_em_meses = 12, fixed_rate_horizon = 1,
            inicializar_tudo = True, Anbima = False
        ):
        
        # O primeiro passo é construir os parâmetros de inicialização da classe YieldCurve
        ### a partir dos parâmetros de uma curva do GSK
        self.raw_params_df = params_df.copy()
        self.max_maturity = max_maturity
        
        temp_LogYields = self.getLogYields(self.raw_params_df, self.max_maturity)
        
        self.mensal = mensal
        self.params_df = self.ajustesDeMensalizacaoYieldsDF(params_df, fixed_horizon_em_meses)[0]
        
        # O primeiro passo é inicializar o GSK_Curve como um objeto do tipo YieldCurve
        super(GSK_Curve, self).__init__(
            yields_df = temp_LogYields, modo = 'Yields' if Anbima else 'LogYields',
            mensal = mensal, fixed_horizon_em_meses = fixed_horizon_em_meses,
            FRA_horizon_em_meses = FRA_horizon_em_meses, fixed_rate_horizon = fixed_rate_horizon,
            inicializar_tudo = inicializar_tudo)
        
        self.InstantLogFRAs = self.getInstantLogFRAs(
            self.params_df.loc[self.dates], self.max_maturity
        )
        self.InstantFRAs = self.getYieldsWithLogYields(
            self.InstantLogFRAs
        )
        
    def getTipsNominalLogReturns(self, LogQReturns):
        
        dates = self.RXs.index
        
        self.LogQReturns = LogQReturns
        self.tipsNominalLogReturns = self.LogRXs + LogQReturns.loc[dates].values

    
    def getLogYield_nWithParams_dfAndn(self, params_df, n, name):
        Beta0, Beta1, Beta2, Beta3, Tau1, Tau2 = params_df[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']].values.T
        
        return pd.DataFrame(
            (Beta0 + Beta1 * (1 - np.exp(-n / Tau1)) / (n / Tau1) + \
               Beta2 * ((1 - np.exp(-n/Tau1))/(n/Tau1) - np.exp(-n/Tau1)) + \
               Beta3 * ((1 - np.exp(-n/Tau2))/(n/Tau2) - np.exp(-n/Tau2)))[:,np.newaxis],
            index = params_df.index,
            columns = [name]
        )
    
    def getInstantLogFRA_nWithParams_dfAndn(self, params_df, n, name):
        Beta0, Beta1, Beta2, Beta3, Tau1, Tau2 = params_df[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']].values.T
        
        return pd.DataFrame(
            (Beta0 + Beta1 * np.exp(-n / Tau1) + \
               Beta2 * np.exp(-n/Tau1) * (n/Tau1) + \
               Beta3 * np.exp(-n/Tau2) * (n/Tau2))[:,np.newaxis],
            columns = [name],
            index = params_df.index
        )
    
    def getLogYields(self, params_df, max_maturity, divisor = 12.0): # Validada com SVENY do FED. Show
        rng_ = range(0, max_maturity)
        return pd.concat([
            self.getLogYield_nWithParams_dfAndn(params_df, (mat+1)/divisor, name = mat) for mat in rng_
        ], axis = 1).rename(
            {k: k+1 for k in rng_}, axis = 1
        )
    
    def getInstantLogFRAs(self, params_df, max_maturity, divisor = 12.0): # Validada com SVENF do FED. Show
        rng_ = range(0, max_maturity)
        return pd.concat([
            self.getInstantLogFRA_nWithParams_dfAndn(params_df, (mat+1)/divisor , name = mat) for mat in rng_
        ], axis = 1).rename(
            {k: k+1 for k in rng_}, axis = 1
        )

            
# %% Classe Mãe dos modelos afim de curva de juros

# Helper functions
def vec(x):
    return np.reshape(x, (-1, 1))
def vec_quad_form(x):
    return vec(np.outer(x, x))

class AffineYieldCurveModel:
    """General Class for Affine Yield Curve Models
    
    Either It receives N x T np.arrays regarding states and shocks or it gets dataframes with data and creates the relevant matrices
    
    States are modeled after X_t+1 = mu + phi X_t + v_t+1
    Indices are organized according to the equations in ACM (2013) / Duffee (2002), so
        X_t is in the t-th column of state_matrix, but v_t+1 is in the t-th column of shock matrix.
    
    This class collects the basic attributes and functions and calculates risk prices.
    
    A and B are arrays with (max_maturities, __, __)-like shapes and are used to calculate LogYields
    """
    # Data is modeled as N x T for states. Makes Var operations easier because X_t is a col vector e X_t = state_matrix[:,t]
    # lambda_vec é 1x(1+N)
    # 
    def __init__(
        self,
        A, B,
        Sigma, lambda_vec,
        A_RF = None, B_RF = None,
        state_matrix = None, risk_free = None,
        ):
        self.A = A
        self.B = B
        self.Sigma = Sigma
        self.lambda_vec = lambda_vec
        self.state_matrix = state_matrix
        self.risk_free = risk_free
        
        self.FittedLogDF = self.getFittedLogDFs(self.A, self.B, self.state_matrix)
        if A_RF is not None and B_RF is not None:
            self.FittedLogDF_RF = self.getFittedLogDFs(self.A_RF, self.B_RF, self.state_matrix)
    
    
    def getSqrtSigma(self):
        self.sqrtmSigma = scipy.linalg.sqrtm(self.Sigma)
    
    def getSqrtSigma_inv(self):
        if not hasattr(self, 'sqrtmSigma'):
            self.getSqrtSigma()
        self.sqrtmSigma_inv = np.linalg.inv(scipy.linalg.sqrtm(self.Sigma))
    
    def getRiskPrice(self):
        if not hasattr(self, 'sqrtmSigma'):
            self.getSqrtSigma()
        self.lambda_t_df = pd.DataFrame(
            np.linalg.solve(
                self.sqrtmSigma,
                self.lambda_vec[:,:1] +  self.lambda_vec[:,1:] @ self.state_matrix.T
                ).T,
            index = self.state_matrix.index,
            columns = self.state_matrix.columns
            )
    
    def getMaximalSharpeRatio(self):
        if not hasattr(self, 'lambda_t_df'):
            self.getRiskPrice()
        self.maxSR = (self.lambda_t_df * self.lambda_t_df).sum(axis = 1).to_frame('Max SR').applymap(sqrt)
    
    # We can get LogKernel Distribution, but then we need to characterize mean and variance for each of the log-normal variables.
    def getLogKernelDistribution(self):
        if not hasattr(self, 'sqrtmSigma'):
            self.getSqrtSigma()
        if not hasattr(self, 'lambda_t_df'):
            self.getRiskPrice()
        
        mean = (- self.risk_free
                - (1/2)*(self.lambda_t_df * self.lambda_t_df).sum(axis = 1).to_frame().values
                ).rename({self.risk_free.columns[0]: 'm_{t+1} - Mean'}, axis = 1)
        variance = (self.lambda_t_df * self.lambda_t_df).sum(axis = 1).to_frame('m_{t+1} - Variance')
        
        self.LogKernelDistribution = pd.concat([
            mean, variance
            ], axis = 1)
    
    def getFittedLogDFs(self, A, B, state_df):
        return (A + state_df @ B).rename({i: i+1 for i in range(B.shape[1])}, axis = 1)
    
    def plotFittedLogYields(self, LogYieldsDF, FittedLogYieldsDF):
        
        plot_dates = LogYieldsDF.index
        rawYields = LogYieldsDF
        fittedYields = FittedLogYieldsDF
        
        import matplotlib.pyplot as mp
        import matplotlib.dates as mdates
        
        mp.figure(dpi=144)
        fig, axes = mp.subplots()
        # axes.plot(plot_dates, rawYields[[12, 60, 120]].values, label="Observed Yields", linewidth=1)
        # axes.plot(plot_dates, fittedYields[[12, 60, 120]].values, label = "Fitted Yields", linewidth=1)
        axes.plot(plot_dates, rawYields[[12, 60, 120]].values, label=[f"Observed {mat}" for mat in [12,60,120]], linewidth=1)
        axes.plot(plot_dates, fittedYields[[12, 60, 120]].values, label = [f"Fitted {mat}" for mat in [12,60,120]], linewidth=1)
        axes.xaxis.set_major_locator(mdates.AutoDateLocator())
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        axes.set_xlabel("date")
        axes.set_ylabel("Yields")
        axes.set_title("Yield fit")
        axes.legend()
        mp.show()


# %% Modelo do ACM
class ACMcomClasse(AffineYieldCurveModel):
    
    def __init__(
        self,
        YieldCurveObject, risk_free_maturity = 1, min_maturity_pca = 3,
        rx_maturities = np.array([6*i for i in range(1,10+1)] + [84,120]), K_states = 5,
        max_maturity_pca = 120, state_matrix_mode = 'PCA5',
        estimate = True
        ):
        
        if not YieldCurveObject.mensal:
            print("""Precisamos trabalhar o PCA e regressão nos dados mensalizados. Precisamos:
- Criar os pesos dos PCAs a partir dos dados mensalizados
- Caso queira trabalhar com o modelo com mu = 0, demean a curva. Acho que não vale a pena. Desenvolvi o modelo original sem fazer isso.
- Adaptar o cálculo dos retornos para 21 dias úteis (ou algum outro período que o valha).
    Seria o caso de incluir uma estimativa da matriz de covariância que leve em conta as autocovariâncias dos excessos de retornos sobrepostos?
Minha intuição é que sim, mas ainda preciso pensar nisso.""")
            
        self.YieldCurveObject = YieldCurveObject
        
        self.AffineYieldCurveModelType = f'ACM-{state_matrix_mode}'
        self.mensal = self.YieldCurveObject.mensal
        
        self.risk_free = self.YieldCurveObject.LogYields[[risk_free_maturity]]
        self.rx_data = self.YieldCurveObject.LogRXs[rx_maturities]
        
        self.t = self.YieldCurveObject.LogYields.shape[0] -\
            self.YieldCurveObject.fixed_horizon
        self.state_matrix_mode = state_matrix_mode
        
        self.K_states = K_states
        
        if estimate:
            self.orYields = self.YieldCurveObject.LogYields.iloc[
                :, min_maturity_pca:max_maturity_pca
                ].copy()
            
            self.state_df = self.getStateDF(
                self.orYields, self.K_states, mode = self.state_matrix_mode)
            self.X_lhs, self.X_rhs, self.mu, self.phi, self.v_df, self.v, self.Sigma_df, self.Sigma = self.estimateStep1(
                self.state_df)
            self.N, self.Z, self.abc, self.E, self.sigmasq_ret, self.a, self.beta, self.c = self.estimateStep2(
                YieldCurveObject, self.rx_data, self.v, self.state_df, self.K_states)
            
            self.E_df = pd.DataFrame(
                self.E.T,
                columns = self.rx_data.columns,
                index = self.rx_data.index[:-1]
                )
            
            self.BStar, self.lambda1, self.lambda0 = self.estimateStep3(
                self.beta, self.c, self.a, self.Sigma, self.sigmasq_ret)
            self.A, self.B = self.buildAandB(
                YieldCurveObject, self.K_states, self.risk_free, self.state_df,
                self.mu, self.lambda0, self.lambda1,
                self.Sigma, self.sigmasq_ret, self.phi,
                divisor_retornos_mensais_centesimais = True)
            self.A_RF, self.B_RF = self.buildAandB_RF(
                YieldCurveObject, self.K_states, self.risk_free, self.state_df,
                self.mu, self.lambda0, self.lambda1,
                self.Sigma, self.sigmasq_ret, self.phi,
                divisor_retornos_mensais_centesimais = True)
        
        super(ACMcomClasse, self).__init__(
            A = self.A, B = self.B, Sigma = self.Sigma, lambda_vec = np.hstack(
                [self.lambda0,self.lambda1]),
            A_RF = self.A_RF, B_RF = self.B_RF,
            state_matrix = self.state_df,
            risk_free = self.risk_free)
        
        self.getRiskPrice()
        self.getLogKernelDistribution()
        self.getMaximalSharpeRatio()
        
        # self.FittedLogDF = self.getFittedLogDFs(self.A, self.B, self.state_df)
        self.FittedLogYields = self.YieldCurveObject.getLogYieldsWithLogDFs(self.FittedLogDF)
        self.FittedLogYields_RF = self.YieldCurveObject.getLogYieldsWithLogDFs(self.FittedLogDF_RF)
        self.BRP = self.YieldCurveObject.LogYields - self.FittedLogYields_RF
        
        # self.plotFittedLogYields(YieldCurveObject.LogYields, self.FittedLogYields)
        
        self.YieldPricingError = self.YieldCurveObject.LogYields - self.FittedLogYields
        
        self.avaliaResiduos(
            smpl= self.YieldCurveObject.LogYields.index[:-1],
            columns = [12,24,36,60,84,120])

    def getStateDF(self, orYields, K_states, mode = 'PCA5'):
        
        if mode == 'PCA5':
            [eigenvalues, eigenvectors] = np.linalg.eig(orYields.resample('M').last().corr())
            yieldPCs = orYields @ np.real(eigenvectors)
            state_matrix = pd.DataFrame(
                yieldPCs.values[:, 0:K_states],
                index = orYields.index,
                columns = range(K_states)
                )
            
        return state_matrix
    
    def estimateNextExpectedReturn(self):
        self.NextExpectedReturn = pd.DataFrame(
            (np.hstack([
                self.a,
                self.beta.T
                ]) @ np.vstack([
                self.getAuxColOnes(1).T,
                self.state_df.iloc[
                    self.t:].T
                ])).T,
            columns = self.rx_data.columns,
            index = self.YieldCurveObject.dates[-1:]
            )
    
    def estimateStep1(self, X_df):
        """Estima o VAR relacionado à transição dos estados.
        Para fazer isso, usamos o dataframe original dos dados.
        Isso é feito para usar a infra de VAR do python.
        """
        
        X_lhs = X_df.iloc[1:].T.values  #X_t+1. Left hand side of VAR.
        t = X_lhs.shape[1]
        X_rhs = np.vstack([
            np.ones((1, t)),
            X_df.iloc[:-1].T.values
            ]) #X_t and a constant.
        var_coeffs = (X_lhs @ np.linalg.pinv(X_rhs))
        mu = var_coeffs[:, [0]]
        phi = var_coeffs[:, 1:]
        
        # O índice do v_df se refere ao dia em que ele entra no estado X_lhs.
        # Isso significa que ele é conhecido quando o estado é conhecido.
        v_df = pd.DataFrame(
            (X_lhs - var_coeffs @ X_rhs).T,
            index = X_df.index[1:],
            columns = X_df.columns
            )
        v = v_df.values.T
        Sigma_df = v_df.cov(ddof=0)
        Sigma = Sigma_df.values
        
        return X_lhs, X_rhs, mu, phi, v_df, v, Sigma_df, Sigma
    
    def estimateStep2(
            self,
            YieldCurveObject, rx_data, v, state_df, K_states, 
            divisor_retornos_mensais_centesimais = True,
            SURmode = False
            ):
        
        if not SURmode:
            selected_rx = rx_data.iloc[
                :self.t
                ].T.values / (12*100 if divisor_retornos_mensais_centesimais else 1)   # Exclui a última entrada, já que não conhecemos os excessos de retorno do último período
            N = selected_rx.shape[0]
            # print("Shapes dos inputs:", np.ones((1, self.t)).shape, v[:,:self.t].shape, state_df.iloc[:self.t].T.values.shape)
            Z = np.vstack([
                self.getAuxColOnes(self.t).T,
                # np.ones((1, self.t)),
                v[:,:self.t],
                state_df.iloc[
                    :self.t
                    ].T.values
                ])  #Innovations and lagged X
            abc = selected_rx @ np.linalg.pinv(Z)
            E = selected_rx - abc @ Z
            sigmasq_ret = np.sum(E * E) / E.size
            
            a = abc[:, [0]]
            beta = abc[:, 1:K_states+1].T
            c = abc[:, K_states+1:]
        
        return N, Z, abc, E, sigmasq_ret, a, beta, c
    
    def estimateStep3(self, beta, c, a, Sigma, sigmasq_ret):
        
        # Step (3) of the three-step procedure: Run cross-sectional regressions
        BStar = np.squeeze(np.apply_along_axis(vec_quad_form, 1, beta.T))
        betapinv = np.linalg.pinv(beta.T)
        lambda1 = betapinv @ c
        lambda0 = betapinv @ (a + 1/2 * (BStar @ vec(Sigma) + sigmasq_ret))
        
        return BStar, lambda1, lambda0
        
    def buildAandB(self, YieldCurveObject, K_states, risk_free, state_df,
                   mu, lambda0, lambda1, Sigma, sigmasq_ret, phi,
                   divisor_retornos_mensais_centesimais = True):
        
        # Run bond pricing recursions
        A = np.zeros((1, YieldCurveObject.max_maturity))
        B = np.zeros((K_states, YieldCurveObject.max_maturity))
        
        delta = (
            risk_free.iloc[:self.t].T.values / (12*100 if divisor_retornos_mensais_centesimais else 1)
            ) @ np.linalg.pinv(
                np.vstack([
                    np.ones((1, self.t)), state_df.iloc[:self.t].T.values
                    ])
                )
        delta0 = delta[[0], [0]]
        delta1 = delta[[0], 1:]
        
        A[0, 0] = - delta0
        B[:, 0] = - delta1
        
        for i in range(0, YieldCurveObject.max_maturity - 1):
            A[0, i+1] = A[0, i] + B[:, i].T @ (mu - lambda0) + 1/2 * (B[:, i].T @ Sigma @ B[:, i] + 0 * sigmasq_ret) - delta0
            B[:, i+1] = B[:, i] @ (phi - lambda1) - delta1
            
        return A, B
    
    def buildAandB_RF(self, YieldCurveObject, K_states, risk_free, state_df,
                   mu, lambda0, lambda1, Sigma, sigmasq_ret, phi,
                   divisor_retornos_mensais_centesimais = True):
        return self.buildAandB(
            YieldCurveObject, K_states, risk_free, state_df,
            mu, np.zeros(lambda0.shape), np.zeros(lambda1.shape), Sigma,
            sigmasq_ret, phi, divisor_retornos_mensais_centesimais = True)
    
    # Aux Functions para chegar às matrizes de variância e covariância
    def getGeradoraResiduos(self, X):
        """Input: matriz X TxK"""
        T, K = X.shape
        return np.eye(T) - X @ np.linalg.solve(X.T @ X, X.T)
    
    def getAuxM_V(self, V):
        return self.getGeradoraResiduos(V.T)
    
    def getAuxf(self, a, c):
        return np.hstack((a,c))
    
    def getAuxColOnes(self, l):
        return np.ones((l,1))
    
    def getAuxZ_(self, i_T, X_):
        return np.hstack([i_T, X_.T]).T
    
    def getLambdaHat(self, beta, rx_matrix, Bstar, Sigma, i_T, sigma2, i_N, M_V, Z_):
        return np.linalg.solve(beta @ beta.T, beta) @ (
            rx_matrix + 1/2 * Bstar @ vec(Sigma) @ i_T.T + 1/2 * sigma2 * i_N @ i_T.T
            ) @ M_V @ np.linalg.solve( Z_ @ M_V @ Z_.T, Z_.T ).T
    
    def getVarVecBeta(sigma2, N, Sigma):
        return sigma2 * np.kron(np.eye(N), np.linalg.inv(Sigma))
    
    def avaliaResiduos(self, smpl, columns = [12,24,36,60,84,120]):
        
        rerror = self.E_df.loc[smpl,columns]
        perror = self.YieldPricingError.loc[smpl, columns]

        metrics = pd.DataFrame()
        for df, name in zip([perror, rerror],
                            ['Panel A: Yield pricing errors', 'Panel B: Return pricing errors']):
            
            rho_df = pd.concat([df, df.shift(1), df.shift(6)], axis = 1)
            
            rho_df.columns = pd.MultiIndex.from_product([
                ['t', 't-1', 't-6'], df.columns
                ])
            
            cov = rho_df.cov()
            
            name_df = pd.DataFrame(columns = [name], index = df.columns)
            metrics = pd.concat([
                metrics,
                name_df,
                round(pd.concat([
                    df.mean(axis = 0).to_frame('Mean'),
                    df.std(axis = 0).to_frame('Std. Deviation'),
                    df.skew(axis = 0).to_frame('Skewness'),
                    df.kurtosis(axis = 0).to_frame('Kurtosis'),
                    pd.DataFrame(
                        np.diag(cov.loc['t','t-1']/cov.loc['t','t']).reshape(-1,1),
                        columns = [r'$\rho(1)$'],
                        index = df.columns
                        ),
                    pd.DataFrame(
                        np.diag(cov.loc['t','t-6']/cov.loc['t','t']).reshape(-1,1),
                        columns = [r'$\rho(6)$'],
                        index = df.columns
                        ),
                    ], axis = 1),3),
                ], axis = 1)

        self.diagnosticsTable = metrics.T.replace(np.nan, '').to_markdown()

    def plotDecomposicao(self, path):
        
        last_date = self.BRP.index[-1]
    
        last_curve = pd.concat([
            self.YieldCurveObject.LogYields.loc[[last_date]],
            self.FittedLogYields_RF.loc[[last_date]],
            self.BRP.loc[[last_date]]
            ], axis = 0).T
        last_curve.columns = ['Última Curva', 'Expectativa Risco-Neutra', 'BRP']
    
        neg_brp = last_curve.loc[last_curve.BRP < 0]
        pos_brp = last_curve.loc[last_curve.BRP >= 0]
    
        ax = last_curve.iloc[:,:-1].plot(
            figsize = (14,7),
            grid = True, lw = 4,
            title = r'Decomposição da Última Curva'
            )
    
        # quebra em segmentos para não haver conexão entre pontos estranhos
        for df, alpha, color, label in zip(
                [neg_brp, pos_brp],
                [.2, .2],
                ['Red', 'Green'],
                ['Negative BRP', 'Positive BRP']
                ):
            seg_list = []
            
            for i, ind in enumerate(df.index):
                if i == 0:
                    start = ind
                    last = start
                elif i == df.shape[0]-1:
                    seg_list.append((start,last))
                else:
                    if df.index[i] == last + 1:
                        last = ind
                    else:
                        seg_list.append((start, last))
                        start = ind
                        last = start
            
            for l_ in seg_list:
                temp = df.loc[(df.index >= l_[0]) & (df.index <= l_[1])]
                ax.fill_between(
                    temp.index,
                    temp['Última Curva'],
                    temp['Expectativa Risco-Neutra'],
                    alpha = alpha,
                    color = color, label = label
                    )
    
        ax.get_figure().savefig(
            path,
            bbox_inches = 'tight'
            )

#%%  Modelo do ACM diário
class ACMDaily(AffineYieldCurveModel):
    
    def __init__(self, ACMMensal, YieldCurveObjectDaily):
        
        self.AffineYieldCurveModelType = f'ACM-{ACMMensal.state_matrix_mode}-Daily'
        self.mensal = False
        
        self.risk_free = YieldCurveObjectDaily.LogYields[ACMMensal.risk_free.columns]
        self.rx_data = YieldCurveObjectDaily.LogRXs[ACMMensal.rx_data.columns]
        
        # self.t = YieldCurveObjectDaily.LogYields.shape[0] - YieldCurveObjectDaily.fixed_horizon
        self.state_matrix_mode = ACMMensal.state_matrix_mode
        
        self.K_states = ACMMensal.K_states
        
        # Aqui recuperamos tudo que dá da estimação mensal e organizamos os dados diários.
        self.orYields = YieldCurveObjectDaily.LogYields[ACMMensal.orYields.columns].copy()
        self.state_df = ACMMensal.getStateDF(
            self.orYields, self.K_states, mode = self.state_matrix_mode)
        self.mu, self.phi, self.Sigma_df, self.Sigma = ACMMensal.mu, ACMMensal.phi, ACMMensal.Sigma_df, ACMMensal.Sigma
        self.N, self.abc, self.E, self.sigmasq_ret, self.a, self.beta, self.c =\
            ACMMensal.N, ACMMensal.abc, ACMMensal.E, ACMMensal.sigmasq_ret, ACMMensal.a, ACMMensal.beta, ACMMensal.c
        self.BStar, self.lambda1, self.lambda0 = ACMMensal.BStar, ACMMensal.lambda1, ACMMensal.lambda0
        self.A, self.B = ACMMensal.A, ACMMensal.B
        self.A_RF, self.B_RF = ACMMensal.A_RF, ACMMensal.B_RF
        
        
        super(ACMDaily, self).__init__(
            A = self.A, B = self.B, Sigma = ACMMensal.Sigma,
            lambda_vec = ACMMensal.lambda_vec,
            A_RF = self.A_RF, B_RF = self.B_RF,
            state_matrix = self.state_df,
            risk_free = self.risk_free)
        
        self.getRiskPrice()
        self.getLogKernelDistribution()
        self.getMaximalSharpeRatio()
        
        # self.FittedLogDF = self.getFittedLogDFs(self.A, self.B, self.state_df)
        self.FittedLogYields = YieldCurveObjectDaily.getLogYieldsWithLogDFs(self.FittedLogDF)
        self.FittedLogYields_RF = YieldCurveObjectDaily.getLogYieldsWithLogDFs(self.FittedLogDF_RF)
        
        self.plotFittedLogYields(YieldCurveObjectDaily.LogYields, self.FittedLogYields)
        
        
# %% Modelo do AACM, modelo de decomposição de curvas nominal e reals
class AACMcomClasse(AffineYieldCurveModel):
    
    def __init__(
        self,
        YieldCurveSetObject, risk_free_maturity = 1,
        maturity_pca_nominal = np.arange(6,126+6,6), K_states_nominal = 3,
        maturity_pca_tips = np.arange(24,126+6,6), K_states_tips = 2,
        cpi_pct_change_annualized_bps = None,
        rx_maturities_nominal = np.array([6] + [12*i for i in range(1,10+1)]),
        rx_maturities_tips = np.array([12*i for i in range(2,10+1)]),
        SURmode = True,
        # max_maturity_pca = 120, state_matrix_mode = 'PCA5',
        estimate = True
        ):
        
        self.nominalCurve = YieldCurveSetObject.nominal
        self.tipsCurve = YieldCurveSetObject.tips
        self.LogQReturns = YieldCurveSetObject.tips.LogQReturns
        
        self.KN = K_states_nominal 
        self.KR = K_states_tips
        
        self.SURmode = SURmode
        
        self.rx_maturities_nominal = rx_maturities_nominal
        self.rx_maturities_tips = rx_maturities_tips
        
        self.rx_data_nominal = self.nominalCurve.LogRXs[rx_maturities_nominal]
        # self.rx_data_tips = self.tipsCurve.LogRXs[rx_maturities_tips]
        self.rx_data_tips_nominal = self.tipsCurve.tipsNominalLogReturns[
            rx_maturities_tips]
        
        # Produz o vetor todo de rx_data para estimação
        self.getCompiledReturns()
        
        # Produz os vetores de estado individuais para estimação do modelo
        self.getIndividualStateDF(K_states_nominal, K_states_tips)
        # Agrega os Estados em um único estado. Essa agregação está aqui para
        ## postergarmos a inclusão de medida de uma liquidez
        self.getStateDF()
        
        # Estima os parâmetros
        if estimate:
            self.estimate()
            print("Ainda falta estimar os parâmetros da inflação")
            
        
        
    def getCompiledReturns(self):
        
        columns = [f'Nominal - {x}' for x in self.rx_data_nominal] + [
            f'Tips - {x}' for x in self.rx_data_tips_nominal]
        
        self.rx_data = pd.concat([
            self.rx_data_nominal, self.rx_data_tips_nominal
            ], axis = 1)
        self.rx_data.columns = columns
        
    
    def getIndividualStateDF(self, K_states_nominal, K_states_tips):
        
        [eigenvalues, eigenvectors] = np.linalg.eig(
            self.nominalCurve.LogYields.resample('M').last().corr()
            )
        yieldPCs = self.nominalCurve.LogYields @ np.real(eigenvectors)
        self.state_matrix_nominal = pd.DataFrame(
            yieldPCs.values[:, 0:K_states_nominal],
            index = self.nominalCurve.LogYields.index,
            columns = [f'PCA Nominal {x}' for x in range(K_states_nominal)]
            )
        
        [eigenvalues, eigenvectors] = np.linalg.eig(
            self.tipsCurve.LogYields.resample('M').last().corr()
            )
        yieldPCs = self.tipsCurve.LogYields @ np.real(eigenvectors)
        self.state_matrix_tips = pd.DataFrame(
            yieldPCs.values[:, 0:K_states_tips],
            index = self.tipsCurve.LogYields.index,
            columns = [f'PCA Tips {x}' for x in range(K_states_tips)]
            )
        
    def getStateDF(self):
        
        if not hasattr(self, 'LiquidityState'):
            print("Não temos nada salvo como atributo em LiquidityState ainda")
            self.state_matrix = pd.concat([
                self.state_matrix_nominal, self.state_matrix_tips
                ], axis = 1)
        else:
            print("""É necessário unir o dataframe de estados com a liquidez.
                  Vou deixar o algoritmo quebrar.""")
        
        self.states_dim = self.state_matrix.shape[1]
    
    def getAuxColOnes(self, l):
        return np.ones((l,1))
    
    def getRegressand(self):
        return self.rx_data.loc[self.estimateStep1Dates].values.T
    
    def estimate(self):
        
        regressor = self.estimateAuxRegressor().T
        regressand = self.getRegressand()/12
        
        abc = regressand @ np.linalg.pinv(regressor)
        
        self.regressor = regressor
        self.regressand = regressand
        self.abc = abc
        
        E_ols = regressand - abc @ regressor
        sigmasq_ret_ols = E_ols @ E_ols.T / E_ols.shape[1]
        
        self.E = E_ols
        self.sigmasq_ret = sigmasq_ret_ols
        
        BPhi_ols = abc[:, 1:self.states_dim+1]
        B_ols = abc[:, self.states_dim+1:]
        
        self.BPhi_ols = BPhi_ols
        self.B_ols = B_ols
        
        if self.SURmode:
            
            PhiTil_gls = - np.linalg.solve(
                B_ols.T @ np.linalg.solve(sigmasq_ret_ols, B_ols ),
                B_ols.T @ np.linalg.solve(sigmasq_ret_ols, BPhi_ols )
                )
            
            self.PhiTil = PhiTil_gls
            
            SURregressor = np.vstack([
                regressor[:1,:],
                - PhiTil_gls @ regressor[1:self.states_dim+1,:] +\
                    regressor[self.states_dim+1:,:]
                ])
            aB = regressand @ np.linalg.pinv(SURregressor)
            alpha_gls = aB[:, :1]
            B_gls = aB[:, 1:]
            
            self.alpha = alpha_gls
            self.B = B_gls
            
            mu_X = self.state_matrix.mean(axis = 0).values[:,np.newaxis]
            
            var_regressand = (regressor[self.states_dim+1:] - mu_X)
            var_regressor = (regressor[1:self.states_dim+1] - mu_X)
            Phi = var_regressand @ np.linalg.pinv(var_regressor)
            var_resid = var_regressand - Phi @ var_regressor
            var_sigmasq = var_resid @ var_resid.T / var_resid.shape[1]
            
            self.mu_X = mu_X
            self.Phi = Phi
            self.Sigma = var_sigmasq
            
            gamma_gls = np.diag(self.B @ self.Sigma @ self.B.T).reshape((-1,1))
            muTil_gls = - np.linalg.solve(
                B_gls.T @ np.linalg.solve(sigmasq_ret_ols, B_gls), B_gls.T
                ) @ np.linalg.solve(
                sigmasq_ret_ols, alpha_gls + 1/2*gamma_gls)
            
            self.gamma = gamma_gls
            self.muTil = muTil_gls        
            
            self.lambda_0 = (np.eye(self.states_dim) - Phi) @ mu_X - muTil_gls
            self.lambda_1 = Phi - PhiTil_gls
        
        Y = self.nominalCurve.FixedRate.loc[self.estimateStep1Dates].values
        X = np.hstack([
            self.getAuxColOnes(len(self.estimateStep1Dates)),
            self.state_matrix.loc[self.estimateStep1Dates].values
            ])

        deltas = Y.T @ np.linalg.pinv(X.T)
        self.delta_0, self.delta_1 = deltas[:,:1], deltas[:,1:].T
        
    def estimateAuxRegressor(self):
        
        self.estimateStep1FullDates = self.state_matrix.dropna().index
        self.estimateStep1Dates = self.estimateStep1FullDates[:-1]
        
        
        self.estimateStep1_state_matrix_t = self.state_matrix.loc[
            self.estimateStep1Dates]
        self.estimateStep1_state_matrix_t1 = self.state_matrix.shift(-1).loc[
            self.estimateStep1Dates]
        
        self.i_T = self.getAuxColOnes(self.estimateStep1_state_matrix_t.shape[0])
        
        return np.hstack([
            self.i_T, 
            self.estimateStep1_state_matrix_t.values,
            self.estimateStep1_state_matrix_t1.values
            ])    
    
    
    def buildNominalAandB(self, YieldCurveObject, K_states, risk_free, state_df,
                    mu, lambda0, lambda1, Sigma, sigmasq_ret, phi,
                    divisor_retornos_mensais_centesimais = True):
        
        # Run bond pricing recursions
        A = np.zeros((1, self.nominalCurve.max_maturity))
        B = np.zeros((self.state_matrix.shape[1], self.nominalCurve.max_maturity))
        
        A[0, 0] = - self.delta0
        B[:, 0] = - self.delta_1
        
        for i in range(0, B.shape[1] - 1):
            A[0, i+1] = A[0, i] + B[:, i].T @ (self.muTil - self.lambda_0) +\
                1/2 * (B[:, i].T @ self.Sigma @ B[:, i] + 0 * self.sigmasq_ret) - self.delta0
            B[:, i+1] = B[:, i] @ self.PhiTil - self.delta_1
        
        self.nominalA, self.nominalB = A, B
    
    def estimatePi(self):
        
        self.iterPi = 0
        self.getPhiArray()
        # auxGetABfromPi(self, pi)
        
        # Find the minimum value of func
        from scipy.optimize import minimize
        res = minimize(
            self.lossFromFitTipsYieldsFromPi,
            # bounds = (),
            x0 = tuple([1] + self.state_matrix.shape[1]*[1]))
        
        self.pi_0 = np.array([[res.x[0]]])
        self.pi_1 = np.array([[x for x in res.x[1:]]])
        
        print(res)
    
    def getPhiArray(self):
        
        phiArray = [np.eye(self.PhiTil.shape[0])]
        for i in range(self.tipsCurve.max_maturity):
            phiArray.append(self.PhiTil @ phiArray[-1])
        
        self.phiArray = np.array(phiArray).cumsum(axis = 0)
    
    def auxGetABfromPi(self, pi):
        
        pi_0 = np.array([[pi[0]]])
        pi_1 = np.array([[x for x in pi[1:]]]).T
        
        B = np.hstack([
            (- self.delta_1.T @ self.phiArray[i] +\
                pi_1.T @ (self.phiArray[i+1] - np.eye(self.states_dim))).T
                for i in range(self.tipsCurve.max_maturity)
            ])
        
        A = [0]
        for i in range(self.tipsCurve.max_maturity):
            if i == 0:
                B_minus = np.zeros((self.states_dim,1)) + pi_1
            else:
                B_minus = B[:,i-1:i] + pi_1
            A.append(float(
                A[-1] + B_minus.T @ self.muTil + 1/2 * B_minus.T @ 
                    self.Sigma @ B_minus - self.delta_0 + pi_0
                    ))
        A = np.array([A[1:]])
        
        return A, B
    
    def fitTipsYieldsFromPi(self, pi):
        
        A, B = self.auxGetABfromPi(pi)
        
        fit = (A + self.state_matrix.loc[self.estimateStep1Dates] @ B).rename(
            {i: i+1 for i in range(self.state_matrix.shape[1])}, axis = 1
            )[self.rx_maturities_tips]
        
        # Multiplica por -12/n, sendo n a maturidade
        return fit * (-12) / np.array([[x for x in fit]])
        # return self.tipsCurve.getLogYieldsWithLogDFs(fit)
    
    def lossFromFitTipsYieldsFromPi(self, pi):
        
        fit = self.fitTipsYieldsFromPi(pi)
        error = self.tipsCurve.LogYields.loc[
            self.estimateStep1Dates, self.rx_maturities_tips
            ] - fit
        
        loss = (error**2).sum().sum()
        self.iterPi += 1
        print(f'{self.iterPi}: {loss}')
        print(self.iterPi*' ' + f': {fit.iloc[-1:]}')
        
        return loss
    
    # def buildTipsAandB(self, YieldCurveObject, K_states, risk_free, state_df,
    #                 mu, lambda0, lambda1, Sigma, sigmasq_ret, phi,
    #                 divisor_retornos_mensais_centesimais = True):
        
    #     # Run bond pricing recursions
    #     A = np.zeros((1, self.nominalCurve.max_maturity))
    #     B = np.zeros((self.state_matrix.shape[1], self.nominalCurve.max_maturity))
        
    #     A[0, 0] = - self.delta0
    #     B[:, 0] = - self.delta_1
        
    #     for i in range(0, B.shape[1] - 1):
    #         A[0, i+1] = A[0, i] + B[:, i].T @ (self.muTil - self.lambda_0) +\
    #             1/2 * (B[:, i].T @ self.Sigma @ B[:, i] + 0 * self.sigmasq_ret) - self.delta0
    #         B[:, i+1] = B[:, i] @ self.PhiTil - self.delta_1
        
    #     self.A, self.B = A, B
    
    
    # Aux Functions para chegar às matrizes de variância e covariância
    def getGeradoraResiduos(self, X):
        """Input: matriz X TxK"""
        T, K = X.shape
        return np.eye(T) - X @ np.linalg.solve(X.T @ X, X.T)
    
    def getAuxM_V(self, V):
        return self.getGeradoraResiduos(V.T)
    
    
    # def buildTipsNominalReturn(self):
        
    #     self.TipsNominalReturn = 
        
#         if not YieldCurveSetObject.mensal:
#             print("""Precisamos trabalhar o PCA e regressão nos dados mensalizados. Precisamos:
# - Criar os pesos dos PCAs a partir dos dados mensalizados
# - Caso queira trabalhar com o modelo com mu = 0, demean a curva. Acho que não vale a pena. Desenvolvi o modelo original sem fazer isso.
# - Adaptar o cálculo dos retornos para 21 dias úteis (ou algum outro período que o valha).
#     Seria o caso de incluir uma estimativa da matriz de covariância que leve em conta as autocovariâncias dos excessos de retornos sobrepostos?
# Minha intuição é que sim, mas ainda preciso pensar nisso.""")
            
        
#         self.AffineYieldCurveModelType = f'ACM-{state_matrix_mode}'
#         self.mensal = YieldCurveSetObject.mensal
        
#         self.risk_free = YieldCurveObject.LogYields[[risk_free_maturity]]
#         self.rx_data = YieldCurveObject.LogRXs[rx_maturities]
        
#         self.t = YieldCurveObject.LogYields.shape[0] - YieldCurveObject.fixed_horizon
#         self.state_matrix_mode = state_matrix_mode
        
#         self.K_states = K_states
        
#         if estimate:
#             self.orYields = YieldCurveObject.LogYields.iloc[
#                 :, min_maturity_pca:max_maturity_pca
#                 ].copy()
            
#             self.state_df = self.getStateDF(
#                 self.orYields, self.K_states, mode = self.state_matrix_mode)
#             self.X_lhs, self.X_rhs, self.mu, self.phi, self.v_df, self.v, self.Sigma_df, self.Sigma = self.estimateStep1(
#                 self.state_df)
#             self.N, self.Z, self.abc, self.E, self.sigmasq_ret, self.a, self.beta, self.c = self.estimateStep2(
#                 YieldCurveObject, self.rx_data, self.v, self.state_df, self.K_states)
#             self.BStar, self.lambda1, self.lambda0 = self.estimateStep3(
#                 self.beta, self.c, self.a, self.Sigma, self.sigmasq_ret)
#             self.A, self.B = self.buildAandB(
#                 YieldCurveObject, self.K_states, self.risk_free, self.state_df,
#                 self.mu, self.lambda0, self.lambda1,
#                 self.Sigma, self.sigmasq_ret, self.phi,
#                 divisor_retornos_mensais_centesimais = True)
#             self.A_RF, self.B_RF = self.buildAandB_RF(
#                 YieldCurveObject, self.K_states, self.risk_free, self.state_df,
#                 self.mu, self.lambda0, self.lambda1,
#                 self.Sigma, self.sigmasq_ret, self.phi,
#                 divisor_retornos_mensais_centesimais = True)
        
#         super(ACMcomClasse, self).__init__(
#             A = self.A, B = self.B, Sigma = self.Sigma, lambda_vec = np.hstack([self.lambda0,self.lambda1]),
#             A_RF = self.A_RF, B_RF = self.B_RF,
#             state_matrix = self.state_df,
#             risk_free = self.risk_free)
        
#         self.getRiskPrice()
#         self.getLogKernelDistribution()
#         self.getMaximalSharpeRatio()
        
#         # self.FittedLogDF = self.getFittedLogDFs(self.A, self.B, self.state_df)
#         self.FittedLogYields = YieldCurveObject.getLogYieldsWithLogDFs(self.FittedLogDF)
#         self.FittedLogYields_RF = YieldCurveObject.getLogYieldsWithLogDFs(self.FittedLogDF_RF)
        
#         self.plotFittedLogYields(YieldCurveObject.LogYields, self.FittedLogYields)

#     def estimateStep1(self, X_df):
#         """Estima o VAR relacionado à transição dos estados.
#         Para fazer isso, usamos o dataframe original dos dados.
#         Isso é feito para usar a infra de VAR do python.
#         """
        
#         X_lhs = X_df.iloc[1:].T.values  #X_t+1. Left hand side of VAR.
#         t = X_lhs.shape[1]
#         X_rhs = np.vstack([
#             np.ones((1, t)),
#             X_df.iloc[:-1].T.values
#             ]) #X_t and a constant.
#         var_coeffs = (X_lhs @ np.linalg.pinv(X_rhs))
#         mu = var_coeffs[:, [0]]
#         phi = var_coeffs[:, 1:]
        
#         # O índice do v_df se refere ao dia em que ele entra no estado X_lhs.
#         # Isso significa que ele é conhecido quando o estado é conhecido.
#         v_df = pd.DataFrame(
#             (X_lhs - var_coeffs @ X_rhs).T,
#             index = X_df.index[1:],
#             columns = X_df.columns
#             )
#         v = v_df.values.T
#         Sigma_df = v_df.cov(ddof=0)
#         Sigma = Sigma_df.values
        
#         return X_lhs, X_rhs, mu, phi, v_df, v, Sigma_df, Sigma
    
#     def estimateStep2(
#             self,
#             YieldCurveObject, rx_data, v, state_df, K_states, 
#             divisor_retornos_mensais_centesimais = True,
#             SURmode = False
#             ):
        
#         if not SURmode:
#             selected_rx = rx_data.iloc[
#                 :self.t
#                 ].T.values / (12*100 if divisor_retornos_mensais_centesimais else 1)   # Exclui a última entrada, já que não conhecemos os excessos de retorno do último período
#             N = selected_rx.shape[0]
#             # print("Shapes dos inputs:", np.ones((1, self.t)).shape, v[:,:self.t].shape, state_df.iloc[:self.t].T.values.shape)
#             Z = np.vstack([
#                 self.getAuxColOnes(self.t).T,
#                 # np.ones((1, self.t)),
#                 v[:,:self.t],
#                 state_df.iloc[
#                     :self.t
#                     ].T.values
#                 ])  #Innovations and lagged X
#             abc = selected_rx @ np.linalg.pinv(Z)
#             E = selected_rx - abc @ Z
#             sigmasq_ret = np.sum(E * E) / E.size
            
#             a = abc[:, [0]]
#             beta = abc[:, 1:K_states+1].T
#             c = abc[:, K_states+1:]
        
#         return N, Z, abc, E, sigmasq_ret, a, beta, c
    
#     def estimateStep3(self, beta, c, a, Sigma, sigmasq_ret):
        
#         # Step (3) of the three-step procedure: Run cross-sectional regressions
#         BStar = np.squeeze(np.apply_along_axis(vec_quad_form, 1, beta.T))
#         betapinv = np.linalg.pinv(beta.T)
#         lambda1 = betapinv @ c
#         lambda0 = betapinv @ (a + 1/2 * (BStar @ vec(Sigma) + sigmasq_ret))
        
#         return BStar, lambda1, lambda0
    
#     def buildAandB_RF(self, YieldCurveObject, K_states, risk_free, state_df,
#                    mu, lambda0, lambda1, Sigma, sigmasq_ret, phi,
#                    divisor_retornos_mensais_centesimais = True):
#         return self.buildAandB(
#             YieldCurveObject, K_states, risk_free, state_df,
#             mu, np.zeros(lambda0.shape), np.zeros(lambda1.shape), Sigma,
#             sigmasq_ret, phi, divisor_retornos_mensais_centesimais = True)

    
#     def getAuxf(self, a, c):
#         return np.hstack((a,c))
    

    
#     def getAuxZ_(self, i_T, X_):
#         return np.hstack([i_T, X_.T]).T
    
#     def getLambdaHat(self, beta, rx_matrix, Bstar, Sigma, i_T, sigma2, i_N, M_V, Z_):
#         return np.linalg.solve(beta @ beta.T, beta) @ (
#             rx_matrix + 1/2 * Bstar @ vec(Sigma) @ i_T.T + 1/2 * sigma2 * i_N @ i_T.T
#             ) @ M_V @ np.linalg.solve( Z_ @ M_V @ Z_.T, Z_.T ).T
    
#     def getVarVecBeta(sigma2, N, Sigma):
#         return sigma2 * np.kron(np.eye(N), np.linalg.inv(Sigma))

