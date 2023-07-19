# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:11:42 2023

@author: Victor
"""

import sys
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