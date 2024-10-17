"""
Gabo Bernardino, Niccolò Fabbri

Utils functions for MTH9897 Stat Arb project
"""


import pandas as pd
import polars as pl
import numpy as np


# Factors
industry_factors = ['AERODEF', 'AIRLINES', 'ALUMSTEL', 'APPAREL', 'AUTO', 'BANKS','BEVTOB', 'BIOLIFE', 'BLDGPROD','CHEM', 'CNSTENG', 'CNSTMACH', 'CNSTMATL', 'COMMEQP', 'COMPELEC', 'COMSVCS', 'CONGLOM', 'CONTAINR', 'DISTRIB', 'DIVFIN', 'ELECEQP', 'ELECUTIL', 'FOODPROD', 'FOODRET', 'GASUTIL', 'HLTHEQP', 'HLTHSVCS', 'HOMEBLDG', 'HOUSEDUR','INDMACH', 'INSURNCE', 'INTERNET',  'LEISPROD', 'LEISSVCS', 'LIFEINS', 'MEDIA', 'MGDHLTH','MULTUTIL', 'OILGSCON', 'OILGSDRL', 'OILGSEQP', 'OILGSEXP', 'PAPER', 'PHARMA', 'PRECMTLS','PSNLPROD','REALEST', 'RESTAUR', 'ROADRAIL','SEMICOND', 'SEMIEQP','SOFTWARE', 'SPLTYRET', 'SPTYCHEM', 'SPTYSTOR', 'TELECOM', 'TRADECO', 'TRANSPRT', 'WIRELESS']

style_factors = ['BETA', 'SIZE', 'MOMENTUM', 'VALUE', 'GROWTH', 'LEVERAGE', 'LIQUIDTY',  'DIVYILD', 'LTREVRSL', 'EARNQLTY', 'STREVRSL']

other_factors = ['1DREVRSL',
 'DWNRISK',
 'EARNYILD',
 'INDMOM',
 'MGMTQLTY',
 'MIDCAP',
 'PROFIT',
 'PROSPECT',
 'RESVOL',
 'SEASON',
 'SENTMT',
]


all_factors = style_factors + other_factors + industry_factors


def get_estu(df, min_cap=10e9):
    """
    Only keep stocks with Market Cap above `min_cap`
    """
    estu = df.filter(pl.col('IssuerMarketCap') > min_cap).clone()
    return estu


def gordon_to_matrix(cov: pl.DataFrame) -> pl.DataFrame:
    """
    Given covariance data in Gordon's form, pivot it to an actual
    covariance matrix
    """
    omega = cov.drop('DataDate').pivot(
        values='VarCovar', index='Factor1', columns='Factor2'
    )

    factors = omega.columns[1:]

    omega_lower = omega[:, 1:].transpose().rename(
        {f'column_{i}': factors[i] for i in range(len(factors))}
    )

    omega = omega.with_columns(
        [pl.col(f).fill_null(omega_lower[f]) for f in factors]
    )
    return omega


def SMW_invert_solver(a, u, v, is_a_inverted=True):
    """
    Use Sherman-Morrison-Woodbury formula to invert a matrix of the form:
    (A + UV')^{-1} - Most efficient version available
    """
    if not is_a_inverted:
        a = np.diag(1 / np.diag(a))

    identity = np.eye(v.shape[0])
    system = identity + v @ a @ u
    M = np.linalg.solve(system, v)

    return a - a @ u @ M @ a
