# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:43:52 2023

@author: Victor
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import math

# %% Generate dataset

# generate approximately correlated Uniform Variables
def generateCorrelatedUniformVariables(rho, N):
    
    U1 = np.random.normal(0, 1, size=N)
    U2 = np.random.normal(0, 1, size=N)
    
    # Create correlation matrix
    corr_matrix = np.array([[1, rho], [rho, 1]])
    
    # Perform Cholesky decomposition
    cholesky_matrix = np.linalg.cholesky(corr_matrix)
    
    # Generate correlated Gaussian variables
    Z = np.dot(cholesky_matrix, np.vstack((U1, U2))).T
    
    return norm.cdf(Z)

def generateCharacteristics(N = 800, T = 600, rho = .7):
    
    corrUniforms = np.hstack(
        [generateCorrelatedUniformVariables(0.7, N) for t in range(T)]
    )
    
    lead = math.ceil(math.log(N, 10))
    
    indexAssets = [
        'Asset #' + f"%0{lead}d" % i
        for i in range(1, N+1)]
    
    return pd.DataFrame(
        corrUniforms,
        columns = pd.MultiIndex.from_product([range(1,1+T),['C1', 'C2']]),
        index = indexAssets
        )
    
def generateFactorReturnsAndBetas(
        muF = 1, sigmaF = 2, sigmaE = 8,
        T = 600, characteristics_df = pd.DataFrame(),
        betaMode = 'Additively Linear'
        ):
    
    # Implied sharpe ratio muF/sigmaF
    factor = pd.DataFrame(
        np.random.normal(muF, sigmaF, size = (T, 1)),
        columns = ['Factor'],
        )
    
    C1 = characteristics_df.reorder_levels([-1,0], axis = 1)['C1']
    C2 = characteristics_df.reorder_levels([-1,0], axis = 1)['C2']
    
    if betaMode == 'Additively Linear':
        betas = (C1 + C2).T
    elif betaMode == 'Nonlinear':
        betas = (1/2 - (1 - C1**2)*(1 - C2**2)).T
    
    returns = betas * factor.values + np.random.normal(
        0, sigmaE, size = (T, N)
        )
    
    return factor, returns, betas

# %% generate portfolios and AP-tree

# Each tree has a different halfing order for the characteristics
# The order defines, at each date, the new portfolio weights for each node
# We define

def getAllHalfingOrders(characteristics, depth = 3, dropSingle = True):
    
    from itertools import product
    # Get the column names of the DataFrame
    column_names = list(np.unique([x[1] for x in characteristics.columns]))
    
    def all_elements_repeated(tup):
        return all(element == tup[0] for element in tup)
    
    if dropSingle:
        return [x for x in product(column_names, repeat=depth)
                if not all_elements_repeated(x)]
    else:
        return list(product(column_names, repeat=depth))

# Given a parent node name and the char_series, create new names
def generateNodeNames(parentName = '1L', char_name = 'C2'):
    
    char_number = char_name[1:]
    
    return parentName + char_number + 'L', parentName + char_number + 'H'

# Given a characteristic series for a given date, separate portfolios in half
def getHalfingWeights(
    char_series,
    weights_series, 
    low_name = '', high_name = ''# both series with the same assets in indices
        ):

    sort_char = char_series.sort_values()
    sort_weights = weights_series.loc[sort_char.index]
    
    low_index = []
    low_weights = []
    
    high_index = []
    high_weights = []
    
    total_low = 0
    for asset, x in sort_weights.items():
        if total_low < 0.5:
            low_index.append(asset)
            if total_low + x <=.5:
                low_weights.append(x)
                total_low += x
            else:
                low_weights.append(.5 - total_low)
                remainder = x - (.5 - total_low)
                total_low += .5
                
                high_index.append(asset)
                high_weights.append(remainder)
        else:
            high_index.append(asset)
            high_weights.append(x)
    
    
    l = pd.Series(low_weights, index = low_index, name = low_name)
    h = pd.Series(high_weights, index = high_index, name = high_name)
    
    return l/l.sum(), h/h.sum()

# given a sequence of splits and characteristics, give node weights for a dtref
def getTreeWeightsForDtref(
    characteristics, sequence = ('C1','C1','C2'),
    modeWeights = 'EW', dtref = 400
        ):
    
    masterNode = 'Main'
    if modeWeights == 'EW':
        masterWeight = pd.Series(
            1/characteristics.shape[0],
            index = characteristics.index,
            name = masterNode
            )
    
    treeWeights = [masterWeight]
    
    maxDepth = len(sequence)
    d = 0
    oldGen = treeWeights
    
    while d < maxDepth:
        
        char_da_vez = characteristics[dtref][sequence[d]]
        newGen = []
        
        for series in oldGen:
            
            if d == 0:
                parentName = ''
            else:
                parentName = series.name
                
            l_name, h_name = generateNodeNames(
                parentName=parentName,
                char_name=sequence[d]
                )
            l_weights, h_weights = getHalfingWeights(
                char_da_vez.loc[series.index],
                series, 
                l_name, h_name# both series with the same assets in indices
                    )
            newGen.append(l_weights)
            newGen.append(h_weights)
        
        treeWeights += newGen
        oldGen = newGen
        d += 1
        
    return pd.concat(treeWeights, axis = 1).fillna(0)

# Given returns and characteristics data, define resampling dates for each portfolio
def assignRebalanceDate(characteristics, rebalanceSkip = 12):
    
    rebalanceDates = characteristics.columns.levels[0][::rebalanceSkip]
    
    assignment = {date: rebalanceDates[rebalanceDates <= date][-1]
        for date in characteristics.columns.levels[0]
        }
    
    datesPerRebalance = {
        date: [key for key, val in assignment.items() if val == date]
            for date in rebalanceDates
            }
    
    return rebalanceDates, assignment, datesPerRebalance

def getPortfoliosReturn(returns_df, weights_df):
    
    return returns_df @ weights_df
 
def getTreeReturns(
    characteristics, returns,
    sequence = ('C1','C1','C2'), modeWeights = 'EW',
    rebalanceSkip = 12
        ):

    rebalanceDates, rebalancePerDate, datesPerRebalance = assignRebalanceDate(
        characteristics, rebalanceSkip = rebalanceSkip)
    
    TreeReturns = pd.DataFrame()
    for date in rebalanceDates:
        tempWeights = getTreeWeightsForDtref(
           characteristics, sequence = ('C1','C1','C2'),
           modeWeights = 'EW', dtref = date
               )
        TreeReturns = pd.concat([
            TreeReturns,
            returns.loc[datesPerRebalance[date]] @ tempWeights
            ], axis = 0)
        
    return TreeReturns

def getNodeDepth(nodeName):
    if nodeName == 'Main':
        return 0
    else:
        return len(nodeName)//2

def getAdjustedTreeReturns(treeReturns):
    
    depth_list = [0,1,1,2,3]
    depth_list = [getNodeDepth(x) for x in treeReturns]
    adjustmentFactor = 2**(-np.array([depth_list], dtype=float))
    
    return treeReturns * adjustmentFactor

# %% SDF Estimation Functions

from scipy.linalg import eigh
from numpy.linalg import svd


# Given a tree of adjusted returns, a test sample and the elastic net parameters,
# get the SRWeights

# We follow the authors notation as close as possible
def getLassoRegressors(
        treeReturns, testSample, lambdaVec = np.array([1,1,1]),
        minEigenValue = 1e-14
        ):
    
    lambda0, lambda1, lambda2 = lambdaVec
    
    muHat = treeReturns.mean().values.reshape(-1,1)
    muBar = muHat.mean()
    
    # Robust Estimators for Mean and Variance
    # sorted eigenvalues (asc) and corresponding eigenvectors (in the columns)
    s, rE = eigh(treeReturns.cov())
    
    eigenPositive = abs(s) > minEigenValue
    # delta = sum(eigenPositive) # How many positive eigen
    
    # eigenPositive[:3] = False
    VTilde = rE[:,eigenPositive]
    DTilde = np.diag(s[eigenPositive])
    
    sigmaTilde = VTilde @ DTilde**(.5) @ VTilde.T
    muTilde = VTilde @ DTilde**(-.5) @ VTilde.T @ (
        muHat + lambda0 * muBar @ np.ones((VTilde.shape[0],1)) 
        )
    
    y = np.hstack([muTilde, np.zeros((VTilde.shape[0],1)) ])
    X = np.hstack([sigmaTilde, math.sqrt(lambda2) * np.eye(VTilde.shape[0]) ])
    
    return y, X


# %% Define cortes válidos de uma AP-tree, caracterizada apenas pelos retornos
### de seus portfolios

def pruneTree(nodeToBeCut = '1L2H', nodesList = []):
    
    numLetters = len(nodeToBeCut)
    return [x for x in nodesList if x[:numLetters] != nodeToBeCut]

def getValidSubtreeSets(returnTree):
    
    max_depth = max([len(x)//2 for x in returnTree])
    
    


# %% get


# %% Generate Characteristics betas

N = 800
T = 600
rho = .7

# Generate random samples
np.random.seed(22111990)  # For reproducibility
characteristics = generateCharacteristics(N, T, rho)

# %% Generate assets returns

np.random.seed(19042021)  # For reproducibility

factor, returns, betas = generateFactorReturnsAndBetas(
        muF = 1, sigmaF = 2, sigmaE = 8,
        T = T, characteristics_df = characteristics,
        betaMode = 'Additively Linear'
        # betaMode = 'Nonlinear'
        )


# %% Define the tree portfolio function

possibleTrees = getAllHalfingOrders(characteristics, depth = 3, dropSingle = True)

baseSequence = possibleTrees

treeReturns = getTreeReturns(
    characteristics, returns,
    sequence = baseSequence, modeWeights = 'EW',
    rebalanceSkip = 1
        )

treeReturns.mean() / treeReturns.std()

adjustedTreeReturns = getAdjustedTreeReturns(treeReturns)

# %%

from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)

# Establish unique IDs to conform with package
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(),np.arange(1,N+1)))
data.firm = data.firm.apply(lambda x: ID[x])

# use multi-index for panel groups
data = data.set_index(['firm', 'year'])
y = data['invest']
X = data.drop('invest', axis=1)

from ipca import InstrumentedPCA
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=X, y=y)
Gamma, Factors = regr.get_factors(label_ind=True)


