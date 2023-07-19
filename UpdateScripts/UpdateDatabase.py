# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:11:42 2023

@author: Victor
"""

import os
os.chdir(r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance\Data')

import sys
sys.path.insert(0, r'C:\Users\Victor\Dropbox\Python Scripts\Repositories\Finance\Yield Curve')
import YieldCurveLib as yc

if len(sys.argv) > 1:
    pasta = sys.argv[1]
else:
    pasta = 'GSKdata'
if isinstance(pasta, str):
    try:
        yc.atualizaParametrosGSKdoFED(pasta = pasta)
    except Exception as e:
        print('Por favor, cheque se seu argumento é uma string válida para nome de pasta.')
        print(e)