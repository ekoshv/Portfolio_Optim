import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from concurrent.futures import ThreadPoolExecutor
from ekoptim import ekoptim

def connect_to_metatrader(path, username, password, server):
    # initialize connection to the MetaTrader 5 terminal
    if not mt5.initialize(path=path,
                          login=username, server=server, password=password):
        print("initialize() failed, error code =",mt5.last_error())
        quit()


if __name__ == "__main__":
    # username = input("Enter your username: ")
    # username = int(username)
    # password = input("Enter your password: ")
    # server = input("Enter the server name: ")
    path = "C:\\Program Files\\JFD MetaTrader 5\\terminal64.exe"
    connect_to_metatrader(path, 555855, "?XRyrR2#", "JFD-Live")
    #connect_to_metatrader(path, username, password, server)
    
    print("-------------------------")
    # display data on connection status, server name and trading account
    print(mt5.terminal_info())
    # display data on MetaTrader 5 version
    print(mt5.version())
    # display the account info
    print(mt5.account_info())
    print("-------------------------")
    
    #retrieve historical data and calculate returns
    print("----Parameteres----")
    total_equity = float(input("Enter total equity: "))
    Group_Name = input("Enter your Group name: e.g.(*.D.EX): ")
    History_Days = int(input("How many historical days: "))
    target_SR = float(input("Target Sharpe Ratio: "))
    target_Volat = float(input("Target Volatility: "))
    target_Return = float(input("Target Return(%): "))/100
    max_weight = float(input("Maximum Weight Allocation(0-1): "))
    min_tresh  = float(input("Minimum Threshold Weight Allocation(0-1): "))
    print("-------------------------")
    print("1. Optimum Return with no limit.")
    print("2. Optimum Return with constraint Sharpe Ratio.")
    print("3. Optimum Return with constraint Volatility.")
    print("4. Optimum Risk with no limit.")
    print("5. Optimum Risk with constraint Sharpe Ratio.")
    print("6. Optimum Risk with constraint Return.")
    print("7. Markowitz Min Risk+Max Sharp.")
    print("-------------------------")
    otp_sel = int(input("Which type of opt you wish: "))
    # min_tresh  = float(input("Minimum Threshold Weight Allocation(0-1): "))
    # max_numb = int(input("Number of weights greater than threshold: "))

    
    symbols=mt5.symbols_get(group=Group_Name)
    returns_list = []
    returns_symbols = []    
    def process_symbol(s):
        rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, History_Days)
        if(len(rates)==History_Days):
            data = pd.DataFrame(data=rates, columns=["time", "open", "high", "low",
                                                     "close", "tick_volume", "spread", "real_volume"])
            # convert time in seconds into the datetime format
            data['time']=pd.to_datetime(data['time'], unit='s')
            data["returns"] = data["close"].pct_change()
            data["returns"] = data["returns"].fillna(0)
            returns_list.append(data["returns"])
            returns_symbols.append(s.name)
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_symbol, symbols)
    
    returns = pd.concat(returns_list, axis=1)
    risk_free_rate = 0.03
    tol = None
    #-----------Optimization---------------------------------
    optimizer = ekoptim(returns, risk_free_rate, target_SR,
                        target_Return, target_Volat, max_weight,tol)
    optimized_weights = optimizer.otpiselect(otp_sel)
    #-----------Optimization---------------------------------
    print("Sum of the weights: ", optimized_weights.sum())
    threshold = min_tresh
    portfolio_return = optimized_weights.T @ returns.mean() * History_Days - risk_free_rate*(History_Days/252)
    portfolio_volatility = (optimized_weights.T @ LedoitWolf().fit(returns).covariance_ @ optimized_weights)**0.5 * np.sqrt(History_Days)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    print("***********************")
    print("Sharpe Ratio: ", sharpe_ratio)
    print("Return: ", portfolio_return)
    print("Volatility: ", portfolio_volatility)
    print("***********************")
    # shut down connection to the MetaTrader 5 terminal
    # mt5.shutdown()
    # connect_to_metatrader(555855, "?XRyrR2#", "JFD-Live")
    # symbols=mt5.symbols_get(group=Group_Name)
    equity_div = []
    for i, weight in enumerate(optimized_weights):
        symbol_info = mt5.symbol_info(returns_symbols[i])
        if (weight>threshold and not(symbol_info.volume_min*symbol_info.bid>1.1*total_equity*weight)):#
            print("Name: ",symbol_info.name,", Bid: ",symbol_info.bid,", step: ",symbol_info.volume_step)
            equity_div_x = {"symbol": symbol_info.name,"Weight":round(weight*10000)/100,
                            "Allocation": round(max(symbol_info.volume_min,
                                              round(total_equity*weight/symbol_info.bid,
                                                    -int(np.floor(np.log10(symbol_info.volume_step))+1)+1))*symbol_info.bid),
                            "Volume": max(symbol_info.volume_min,round(total_equity*weight/symbol_info.bid,
                                                                      -int(np.floor(np.log10(symbol_info.volume_step))+1)+1))}
            equity_div.append(equity_div_x)
    equity_div_df = pd.DataFrame(equity_div)
    equity_div_df.sort_values(by="Weight")
    print("------------------")
    print(equity_div_df)
    print("------------------")
    print("All the equity needed: ",equity_div_df["Allocation"].sum())
    # use Monte Carlo simulation to generate multiple sets of random weights
    num_portfolios = 5000
    returns_listx = []
    sharpe_ratios_listx = []
    volatilities_listx = []
    for i in range(num_portfolios):
        weights = np.random.random(returns.shape[1])
        weights /= np.sum(weights)
        portfolio_returnx = weights.T @ returns.mean() * History_Days - risk_free_rate*(History_Days/252)
        portfolio_volatilityx = (weights.T @ LedoitWolf().fit(returns).covariance_ @ weights)**0.5 * np.sqrt(History_Days)
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
    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
