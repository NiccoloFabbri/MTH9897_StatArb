import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle
import bz2
import scipy
plt.style.use('seaborn')


figsize = (12, 6)


# Data cleaning
def open_bz2(file_name):
    """
    Function to open a bz2 file of dates: pd.DataFrame's
    and turn the df's into Polars
    """
    pandas_dict = pd.read_pickle(bz2.open(file_name, 'rb'))
    polars_dict = {}
    polars_dict.update({k: pl.from_pandas(v) for k, v in pandas_dict.items()})
    return polars_dict


def wins(x, a, b):
    return np.where(x <= a, a, np.where(x >= b, b, x))

def wins_pl(series, lower, upper):
    """
    Winsorize the series by clipping extreme values.

    Args:
        series (pl.Series): Input series to winsorize.
        lower (float): Lower quantile to clip.
        upper (float): Upper quantile to clip.

    Returns:
        pl.Series: Winsorized series.
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)


def clean_nas(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for numeric_column in numeric_columns:
        df[numeric_column] = np.nan_to_num(df[numeric_column])

    return df


# plotting
def density_plot(data, title):
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.plot(xs,density(xs))
    plt.title(title)
    plt.show()


# Factors
industry_factors = ['AERODEF', 'AIRLINES', 'ALUMSTEL', 'APPAREL', 'AUTO',
       'BANKS','BEVTOB', 'BIOLIFE', 'BLDGPROD','CHEM', 'CNSTENG', 'CNSTMACH', 'CNSTMATL', 'COMMEQP', 'COMPELEC',
       'COMSVCS', 'CONGLOM', 'CONTAINR', 'DISTRIB',
       'DIVFIN', 'ELECEQP', 'ELECUTIL', 'FOODPROD', 'FOODRET', 'GASUTIL',
       'HLTHEQP', 'HLTHSVCS', 'HOMEBLDG', 'HOUSEDUR','INDMACH', 'INSURNCE', 'INTERNET',
        'LEISPROD', 'LEISSVCS', 'LIFEINS', 'MEDIA', 'MGDHLTH','MULTUTIL',
       'OILGSCON', 'OILGSDRL', 'OILGSEQP', 'OILGSEXP', 'PAPER', 'PHARMA',
       'PRECMTLS','PSNLPROD','REALEST',
       'RESTAUR', 'ROADRAIL','SEMICOND', 'SEMIEQP','SOFTWARE', 'SPLTYRET', 'SPTYCHEM', 'SPTYSTOR',
       'TELECOM', 'TRADECO', 'TRANSPRT', 'WIRELESS']


style_factors = ['BETA', 'SIZE', 'MOMENTUM', 'VALUE', 'GROWTH', 'LEVERAGE',
                 'LIQUIDTY', 'DIVYILD', 'LTREVRSL', 'EARNQLTY']

alpha_factors = ['STREVRSL', 'MGMTQLTY', 'SENTMT',
                 'EARNYILD', 'SEASON', 'INDMOM']


all_factors = ['Intercept'] + style_factors + industry_factors# + alpha_factors
n_factors = len(all_factors)


# an R-style formula which can be used to construct a cross sectional regression
def get_formula(alphas, Y):
    L = ["0"]
    L.extend(alphas)
    L.extend(style_factors)
    L.extend(industry_factors)
    return Y + " ~ " + " + ".join(L)


# The term 'estu' is short for estimation universe
def get_estu(df, min_cap=1e9):
    estu = df.filter(pl.col('IssuerMarketCap') > min_cap).clone()
    return estu


# Sherman-Morrison-Woodbury matrix inversion
# I have experimented with different versions as the SMW formula to deal with
# the smaller matrix inversion which appears in the formula
# The function `SMW_invert_basic` simply uses `np.linalg.inv` to invert the matrix;
# The function `SMW_invert_solver` uses `np.linalg.solve` to compute
# the inverse multiplied by the next term as the solution to multiple
# consecutive linear systems. This gets the best performance;
# The function `SMW_invert_lu` uses an LU solver which I implemented as
# efficiently as I could, but it does not improve on numpy's own solver

def SMW_invert_basic(a, u, v, is_a_inverted=True):
    """
    Use Sherman-Morrison-Woodbury formula to invert a matrix of the form:
    (A + UV')^{-1}
    """
    if not is_a_inverted:
        a = np.diag(1 / np.diag(a))
    identity = np.eye(v.shape[0])
    return a - a @ u @ np.linalg.inv(identity + v @ a @ u) @ v @ a


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


def solve_system(A, X):
    """
    A of size dxd, X of size dxn;
    output is M of size dxn and solves `AM = X`
    """
    _, L, U = scipy.linalg.lu(A)

    y = np.linalg.solve(L, X)
    out = np.linalg.solve(U, y)

    return out


def SMW_invert_lu(a, u, v, is_a_inverted=True):
    """
    Use Sherman-Morrison-Woodbury formula to invert a matrix of the form:
    (A + UV')^{-1}
    """
    if not is_a_inverted:
        a = np.diag(1 / np.diag(a))

    identity = np.eye(v.shape[0])
    system = identity + v @ a @ u
    M = solve_system(system, v)

    return a - a @ u @ M @ a


# visualization 
def visualize_results(_data: dict):
    """
    *** Using `pandas` instead of `polars` because it's easier to plot ***

    Based on the project description, this function will do the following:

    * Plot long/short/net in dollars
    * Number of holdings
    * Factor model's predicted volatility of the portfolio
    * Percent of variance from idio, style, industry
    * Predicted beta of the portfolio, which is the dot product of holdings with the PredBeta attribute.
    * Dot product of your portfolio with the return for each date in the sample and plot the cumulative sum of the results
    """
    # make a dataframe from the input
    data = pd.DataFrame(_data).T
    # print(data)
    data.index = pd.to_datetime(data.index, format='%Y%m%d')

    # Plot long/short/net in dollars
    plt.figure(figsize=figsize)
    plt.bar(data.index, data['Long'], label='Long')
    plt.bar(data.index, data['Short'], label='Short')
    plt.title("Long and Short exposures")
    plt.xlabel('Time')
    plt.ylabel('$')
    plt.legend()
    plt.show()

    # Number of holdings
    plt.figure(figsize=figsize)
    plt.bar(data.index, data['Nhold'])
    plt.title("Number of holdings per day")
    plt.xlabel('Time')
    plt.show()

    # Factor model's predicted volatility of the portfolio
    plt.figure(figsize=figsize)
    plt.plot(data.index, data['Volatility'])
    plt.title("Predicted volatility")
    plt.xlabel('Time')
    plt.ylabel('$')
    plt.show()

    # Percent of variance from idio, style, industry
    btm = np.zeros(len(data))
    plt.figure(figsize=figsize)
    for prop in ['Idio', 'Style', 'Industry']:
        plt.bar(data.index, data[prop] * data['variance'], bottom=btm, label=prop)
        btm += data[prop] * data['variance']
    plt.title('Variance decomposition by day')
    plt.xlabel('Time')
    plt.ylabel(r'$\$^2$')
    plt.legend()
    plt.show()

    # Predicted beta of the portfolio
    plt.figure(figsize=figsize)
    plt.plot(data.index, data['Beta'])
    plt.title("Predicted portfolio beta")
    plt.xlabel('Time')
    plt.ylabel(r'$\beta$')
    plt.show()

    # Dot product of your portfolio with the return
    plt.figure(figsize=figsize)
    plt.plot(data.index, data['PnL'].cumsum())
    plt.title("Cumulative portfolio P&L")
    plt.xlabel("Time")
    plt.ylabel("$")
    plt.show()
    
    mu = data['PnL'].cumsum().iloc[-1]
    sig = np.sqrt(sum(data['Volatility'] ** 2))
    sharpe = mu / sig / np.sqrt(len(data))
    print(f"Daily Sharpe Ratio = {sharpe:.6f}")


