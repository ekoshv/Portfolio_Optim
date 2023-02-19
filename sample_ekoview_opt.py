import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from ekoptim import ekoptim
import pickle as pkl
import datetime

#retrieve historical data and calculate returns
print("----Parameteres----")
total_equity = float(input("Enter total equity: "))
History_Days = int(input("How many Trading Days(Trading Year eq 252 days): "))
target_SR = float(input("Target Sharpe Ratio: "))
target_Volat = float(input("Target Volatility: "))
target_Return = float(input("Target Return(%): "))/100
max_weight = float(input("Maximum Weight Allocation(0-1): "))
min_tresh  = float(input("Minimum Threshold Weight Allocation(0-1): "))
max_numb = int(input("Number of weights greater than threshold: "))


Exg = 'FWB'
rates_lists = []
with open('rates_list_{}_{}_V01.pkl'.format(Exg,History_Days), 'rb') as inpt:
    rates_lists = pkl.load(inpt)
    inpt.close()


returns_list = []
begindate = (datetime.datetime.today()-
             datetime.timedelta(days=(int(History_Days*365/252))))
for rate in rates_lists:
    if ((rate.index[0] > begindate) and
        (True)):
        returnx = pd.Series(rate['returns'], name=rate['symbol'][0])
        returns_list.append(returnx)
returns = pd.concat(returns_list, axis=1)
returns.drop(returns.columns[returns.apply(lambda col: col.isna().sum() >
                                           (int(0.05*History_Days*365/252)))],
             axis=1,inplace=True)
returns.fillna(0,inplace=True)
risk_free_rate = 0.03
xreturn = returns.iloc[:,300:500]
tolerance = None
#-----------Optimization---------------------------------
optimizer = ekoptim(xreturn, risk_free_rate, target_SR,
                    target_Return, target_Volat, max_weight,min_tresh,max_numb,tolerance)
optimized_weights = optimizer.markowitz_optimization_risk_sharpe()
#-----------Optimization---------------------------------
print("Sum of the weights: ", optimized_weights.sum())
threshold = 0.005
portfolio_return = optimized_weights.T @ xreturn.mean() * History_Days - risk_free_rate*(History_Days/252)
portfolio_volatility = (optimized_weights.T @ LedoitWolf().fit(xreturn).covariance_ @ optimized_weights)**0.5 * np.sqrt(History_Days)
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
print("***********************")
print("Sharpe Ratio: ", sharpe_ratio)
print("Return: ", portfolio_return)
print("Volatility: ", portfolio_volatility)
print("***********************")

equity_div = []
for i, weight in enumerate(optimized_weights):    
    if (weight>threshold):
        equity_div_x = {"symbol": xreturn.iloc[:,i].name,
                        "Weight":round(weight*10000)/100,
                        "Allocation": total_equity*weight}
        equity_div.append(equity_div_x)
equity_div_df = pd.DataFrame(equity_div)
print("------------------")
print(equity_div_df)
print("------------------")
print("All the equity needed: ",equity_div_df["Allocation"].sum())
# use Monte Carlo simulation to generate multiple sets of random weights
num_portfolios = 500
returns_listx = []
sharpe_ratios_listx = []
volatilities_listx = []
for i in range(num_portfolios):
    weights = np.random.random(xreturn.shape[1])
    weights /= np.sum(weights)
    portfolio_returnx = weights.T @ xreturn.mean() * History_Days - risk_free_rate*(History_Days/252)
    portfolio_volatilityx = (weights.T @ LedoitWolf().fit(xreturn).covariance_ @ weights)**0.5 * np.sqrt(History_Days)
    sharpe_ratiox = (portfolio_returnx - risk_free_rate) / portfolio_volatilityx
    returns_listx.append(portfolio_returnx)
    volatilities_listx.append(portfolio_volatilityx)
    sharpe_ratios_listx.append(sharpe_ratiox)
# plot the efficient frontier
plt.scatter(volatilities_listx, returns_listx, c=sharpe_ratios_listx)
plt.scatter(portfolio_volatility, portfolio_return, c='red', marker='D', s=200)
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()