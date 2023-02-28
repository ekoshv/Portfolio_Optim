import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
#from concurrent.futures import ThreadPoolExecutor
from ekoptim import ekoptim
from ekoView import TradingViewfeed, Interval
import datetime

def connect_to_metatrader(path, username, password, server):
    # initialize connection to the MetaTrader 5 terminal
    if not mt5.initialize(path=path,
                          login=username, server=server, password=password):
        print("initialize() failed, error code =",mt5.last_error())
        quit()

def filter_symbols_by_path(symbols, path_name):
    filtered_symbols = []
    for s in symbols:
        if path_name in s.path:
            filtered_symbols.append(s)
    return filtered_symbols

if __name__ == "__main__":
    
    Exg = "FWB"
    #tv = TradingViewfeed(username, password)
    tv = TradingViewfeed()
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
    filt_type = input("Type of filter(Group or Path: G/P): ")
    if(filt_type == 'G' or filt_type == 'g'):
        Group_Name = input("Enter your Group name: e.g.(*.D.EX): ")
        filtered_symbols=mt5.symbols_get(group=Group_Name)
    elif(filt_type == 'P' or filt_type == 'p'):
        Path_Name = input("Enter your Path: e.g.('Stocks.DE\\'): ")
        symbols = mt5.symbols_get()
        filtered_symbols = filter_symbols_by_path(symbols, Path_Name)
    History_Days = int(input("How many historical days: "))
    #is_weighted = bool(input("Do you want it to be weighted(True/False): "))
    target_SR = float(input("Target Sharpe Ratio: "))
    target_Volat = float(input("Target Volatility: "))
    target_Return = float(input("Target Return(%): "))/100
    max_weight = float(input("Maximum Weight Allocation(0-1): "))
    min_tresh  = float(input("Minimum Threshold Weight Allocation(0-1): "))
    print("-------------------------")
    print("1.  Optimum Return with no limit.")
    print("2.  Optimum Return with constraint Sharpe Ratio.")
    print("3.  Optimum Return with constraint Volatility.")
    print("4.  Optimum Risk with no limit.")
    print("5.  Optimum Risk with constraint Sharpe Ratio.")
    print("6.  Optimum Risk with constraint Return.")
    print("7.  Markowitz Min-Risk + Max-Sharp.")
    print("8.  Markowitz Min-Risk + Max-Sortino.")
    print("9.  EKO Min-Surprise + Max-Sharp.")
    print("10. EKO Min-Surprise + Max-Sortino.")
    print("-------------------------")
    otp_sel = int(input("Which type of opt you wish: "))
    # min_tresh  = float(input("Minimum Threshold Weight Allocation(0-1): "))
    # max_numb = int(input("Number of weights greater than threshold: "))
    
    returns_list = []
    rates_list = []
    begindate = (datetime.datetime.today()-
                 datetime.timedelta(days=(int(History_Days*365/252))))
    i=0
    for s in filtered_symbols:
        try:
            sym_list = tv.search_symbol(s.isin,exchange="FWB")
            print("Symbol is: ", s.name,": ",sym_list[0]["symbol"])
            rates = tv.get_hist(symbol=sym_list[0]["symbol"], 
                                exchange=Exg, 
                                interval= Interval.in_daily, 
                                n_bars=History_Days,
                                ctype="dividends")
            #print(rates)
            if(len(rates)>=round(0.75*History_Days) and
               (rates.index[0] > begindate)):
                rates[s.name] = ((rates['close']).
                                 interpolate(method='polynomial', order=2)).pct_change()
                rates[s.name] = rates[s.name].interpolate(method='polynomial', order=2)
                rates[s.name] = rates[s.name].fillna(0)
                rates_list.append(rates)
                returns_list.append(rates[s.name])
            #returns_symbols.append(s.name)
            i+=1
            print("---------")
            print(round(10000*i/len(filtered_symbols))/100,"%")
            print("---------")
        except:
            print("---------")
            print("An exception occurred: ", s.name)
            print("---------")
    
    returns = pd.concat(returns_list, axis=1)
    returns.fillna(0,inplace=True)
    
    # if is_weighted:
    #     maxsec = int(returns.index.max().timestamp())
        
    risk_free_rate = 0.03
    tol = None
    #-----------Optimization---------------------------------
    optimizer = ekoptim(returns, risk_free_rate, target_SR,
                        target_Return, target_Volat, max_weight,tol)
    optimized_weights = optimizer.optiselect(otp_sel)
    #-----------Optimization---------------------------------
    print("Sum of the weights: ", optimized_weights.sum())
    threshold = min_tresh
    
    print("***********************")
    metrics = optimizer.calculate_metrics(optimized_weights)    
    print("Sharpe Ratio: ", metrics['Sharpe'])
    print("Return: ", round(metrics['Return']*1000)/10, "%")
    print("Volatility: ", metrics['Risk'])
    print("Sortino: ", metrics['Sortino'])
    print("Surprise: ", metrics['Surprise'])
    print("***********************")
    # shut down connection to the MetaTrader 5 terminal
    # mt5.shutdown()
    # connect_to_metatrader(555855, "?XRyrR2#", "JFD-Live")
    # symbols=mt5.symbols_get(group=Group_Name)
    equity_div = []
    
    # for i, weight in enumerate(optimized_weights):
    #     try:
    #         symbol_info = mt5.symbol_info_tick(returns.columns[i])
    #         if (weight>threshold and not(symbol_info.volume_min*symbol_info.bid>
    #                                      1.1*total_equity*weight)):#
    #             print("Name: ",symbol_info.name,", Bid: ",
    #                   symbol_info.bid,", step: ",symbol_info.volume_step)
    #             equity_div_x = {"symbol": symbol_info.name,"Weight":round(weight*10000)/100,
    #                             "Allocation": round(max(symbol_info.volume_min,
    #                                               round(total_equity*weight/symbol_info.bid,
    #                                                     -int(np.floor(np.log10(symbol_info.volume_step))+
    #                                                          1)+1))*symbol_info.bid),
    #                             "Volume": max(symbol_info.volume_min,
    #                                           round(total_equity*weight/symbol_info.bid,
    #                                                                       -int(np.floor(np.log10(symbol_info.volume_step))+
    #                                                                            1)+1))}
    #             equity_div.append(equity_div_x)
    #     except:
    #         print("An exception happened!")
    
    returns_selected = []
    for i, weight in enumerate(optimized_weights):    
        if (weight>threshold):
            equity_div_x = {"symbol": returns.columns[i],
                            "Weight":round(weight*10000)/100,
                            "Allocation": total_equity*weight}
            equity_div.append(equity_div_x)
            returns_selected.append(returns[returns.columns[i]])
    
    equity_div_df = pd.DataFrame(equity_div)
    equity_div_df.sort_values(by="Weight",inplace=True, ignore_index=True)
    returns_selected = pd.concat(returns_selected, axis=1)
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
        weights = np.random.random(returns.shape[1])
        weights /= np.sum(weights)
        portfolio_returnx = (weights.T @ returns.mean() * History_Days -
                             risk_free_rate*(History_Days/252))
        portfolio_volatilityx = (weights.T @ LedoitWolf().fit(returns).
                                 covariance_ @ weights)**0.5 * np.sqrt(History_Days)
        sharpe_ratiox = (portfolio_returnx - risk_free_rate) / portfolio_volatilityx
        returns_listx.append(portfolio_returnx)
        volatilities_listx.append(portfolio_volatilityx)
        sharpe_ratios_listx.append(sharpe_ratiox)
    # plot the efficient frontier
    plt.scatter(volatilities_listx, returns_listx, c=sharpe_ratios_listx)
    plt.scatter(metrics['Risk'], metrics['Return'], c='red', marker='D', s=200)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()
    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
