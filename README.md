# Portfolio_Optim
 Markowitz Portfolio Optimizer

# Installation

This module is installed via pip:

pip install --upgrade --no-cache-dir git+https://github.com/ekoshv/Portfolio_Optim.git

# ekoptim class documentation
The ekoptim class is used to optimize a portfolio based on a number of inputs. It uses optimization functions to calculate the weights of assets to maximize certain criteria such as return or Sharpe ratio.

# Dependencies
numpy
math
scipy
sklearn
# Parameters
returns: An array of asset returns. It should have shape (days, assets).
risk_free_rate: A float representing the risk-free rate.
target_SR: A float representing the target Sharpe ratio.
target_Return: A float representing the target return.
target_Volat: A float representing the target volatility.
max_weight: A float representing the maximum weight of any asset.
toler: A float representing the optimization tolerance.
# Class Functions
__init__(self, returns, risk_free_rate, target_SR, target_Return, target_Volat, max_weight, toler)
This function initializes the class instance with input parameters.

risk_cnt(self, w)
This function calculates the risk (volatility) of a portfolio given the weights of each asset in the portfolio.

sharpe_ratio_cnt(self, w)
This function calculates the Sharpe ratio of a portfolio given the weights of each asset in the portfolio.

return_cnt(self, w)
This function calculates the return of a portfolio given the weights of each asset in the portfolio.

sortino_ratio_cnt(self, w)
This function calculates the Sortino ratio of a portfolio given the weights of each asset in the portfolio.

Optim_return_nl(self)
This function runs an optimization algorithm to maximize the return of a portfolio subject to the constraints defined in the constraints attribute of the class.

Optim_return_cnt_sharpe(self)
This function runs an optimization algorithm to maximize the return of a portfolio subject to the constraints defined in the constraints attribute of the class, as well as a constraint on the Sharpe ratio.

Optim_return_cnt_volat(self)
This function runs an optimization algorithm to maximize the return of a portfolio subject to the constraints defined in the constraints attribute of the class, as well as a constraint on the volatility.

# Attributes
returns: An array of asset returns. It should have shape (days, assets).
target_SR: A float representing the target Sharpe ratio.
target_Return: A float representing the target return.
target_Volat: A float representing the target volatility.
max_weight: A float representing the maximum weight of any asset.
n: An integer representing the number of assets.
days: An integer representing the number of days.
w0: An array representing the initial weights of each asset.
durc: A float representing the duration.
risk_free_rate: A float representing the risk-free rate.
toler: A float representing the optimization tolerance.
bounds: A list of tuples representing the upper and lower bounds of each asset weight.
constraints: A list of dictionaries representing the constraints on the optimization (x+-15%). The constraints include a sum-to-one constraint, constraints on the return, Sharpe ratio, volatility, and maximum weight of each asset.
