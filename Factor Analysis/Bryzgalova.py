# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:43:52 2023

@author: Victor
"""

import numpy as np
from scipy.stats import norm
import pandas as pd

# %% Aux functions

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


# %% Generate Characteristics betas

# Generate random samples
np.random.seed(22111990)  # For reproducibility

# Set the desired correlation coefficient
# correlation_coefficient = 0.7
N, T = 800, 600

corrUniforms = np.hstack(
    [generateCorrelatedUniformVariables(0.7, N) for t in range(T)]
)
characteristics = pd.DataFrame(
    corrUniforms,
    columns = pd.MultiIndex.from_product([range(1,1+T),['C1', 'C2']])
    )

# %% Generate assets returns

def generateFactorReturnsAndBetas(
        muF = 1, sigmaF = 2, sigmaE = 8,
        T = 600, characteristics_df = characteristics,
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

np.random.seed(19042021)  # For reproducibility

factor, returns, betas = generateFactorReturnsAndBetas(
        muF = 1, sigmaF = 2, sigmaE = 8,
        T = T,
        characteristics_df = characteristics,
        # betaMode = 'Additively Linear'
        betaMode = 'Nonlinear'
        )





