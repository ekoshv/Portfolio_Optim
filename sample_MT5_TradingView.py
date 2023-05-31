#%%
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from concurrent.futures import ThreadPoolExecutor
from ekoptim import ekoptim
from ekoView import TradingViewfeed, Interval
import datetime
from tqdm import tqdm
import traceback
from sklearn.covariance import LedoitWolf
import dill
import types
from pathlib import Path

def can_be_pickled(obj):
    try:
        dill.dumps(obj)
        return True
    except:
        return False

def connect_to_metatrader(path, username, password, server):
    # initialize connection to the MetaTrader 5 terminal
    if not mt5.initialize(path=path,
                          login=username, server=server, password=password):
        print("initialize() failed, error code =",mt5.last_error())
        quit()

def filter_symbols_by_path(symbols, path_names_str):
    path_names = path_names_str.split(",")
    filtered_symbols = []
    for s in symbols:
        for pn in path_names:
           if pn in s.path:
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
#%%
    path = "C:\\Program Files\\JFD MetaTrader 5\\terminal64.exe"
    # connect_to_metatrader(path, username, password, server)
    connect_to_metatrader(path, 555855, "?XRyrR2#", "JFD-Live")
    
    print("-------------------------")
    # display data on connection status, server name and trading account
    print(mt5.terminal_info())
    # display data on MetaTrader 5 version
    print(mt5.version())
    # display the account info
    print(mt5.account_info())
    print("-------------------------")
#%%    
    #retrieve historical data and calculate returns
    print("----Parameteres----")
    total_equity = float(input("Enter total equity: "))
    filt_type = input("Type of filter(Group or Path: G/P): ")
    if(filt_type == 'G' or filt_type == 'g'):
        Group_Name = input("Enter your Group name: e.g.(*.D.EX): ")
        filtered_symbols=mt5.symbols_get(group=Group_Name)
        allsymb = [s.name for s in filtered_symbols]
    elif(filt_type == 'P' or filt_type == 'p'):
        Path_Name = input("Enter your Path: e.g.('Stocks.DE\\'): ")
        symbols = mt5.symbols_get()
        filtered_symbols = filter_symbols_by_path(symbols, Path_Name)
        allsymb = [s.name for s in filtered_symbols]
    oil = input("Enter the Oil name: e.g.(.BrentCrude):")
    oil_smb = mt5.symbol_info(oil)
    gold = input("Enter the Gold name: e.g.(XAUUSD):")
    gold_smb = mt5.symbol_info(gold)
    History_Days = int(input("How many historical days: "))
    mt5same = input("Length of MT5 as of TradingView?(True/False): ")
    mt5same = True if mt5same.lower() == "true" else False
#%%    
    returns_TV = []
    returns_MT5 = []
    rates_TV = []
    rates_MT5 = []
    begindate = (datetime.datetime.today()-
                 datetime.timedelta(days=(int(1.1*History_Days*365/252))))
    i=0
    for s in tqdm(filtered_symbols):
        try:
            sym_list = tv.search_symbol(s.isin,exchange="FWB")
            print("Symbol is: ", s.name,": ",sym_list[0]["symbol"])
            rates = tv.get_hist(symbol=sym_list[0]["symbol"], 
                                exchange=Exg, 
                                interval= Interval.in_daily, 
                                n_bars=History_Days,
                                ctype="splits")
            datamt5 = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, History_Days)
            #print(rates)
            if(len(rates)>=round(0.75*History_Days) and
               (len(datamt5)>=round(0.75*History_Days) or not mt5same) and
               (rates.index[0] >= begindate)):
                print('***',len(rates),', ',rates.index[0],', ',s.name)
                rates[s.name] = ((rates['close']).
                                 interpolate(method='polynomial', order=2)).pct_change()
                rates[s.name] = rates[s.name].interpolate(method='polynomial', order=2)
                rates[s.name] = rates[s.name].fillna(0)
                rates_TV.append(rates)
                returns_TV.append(rates[s.name])
                
                data = pd.DataFrame(data=datamt5, columns=["time", "open", "high", "low",
                                                         "close", "tick_volume", "spread", "real_volume"])
                # convert time in seconds into the datetime format
                data['time']=pd.to_datetime(data['time'], unit='s')
                data.set_index('time', inplace=True)
                data[s.name] = data["close"].pct_change()
                data[s.name] = data[s.name].fillna(0)
                rates_MT5.append(data)
                returns_MT5.append(data[s.name])
        except Exception as e:
            print("---------")
            print(f"An error occurred in downloading {s.name}: {e}")
            traceback.print_exc()
            print("---------")
    
    for s in tqdm([gold_smb, oil_smb]):
        try:
            datamt5 = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, History_Days)
            data = pd.DataFrame(data=datamt5, columns=["time", "open", "high", "low",
                                                     "close", "tick_volume", "spread", "real_volume"])
            # convert time in seconds into the datetime format
            data['time']=pd.to_datetime(data['time'], unit='s')
            data.set_index('time', inplace=True)
            data[s.name] = data["close"].pct_change()
            data[s.name] = data[s.name].fillna(0)
            rates_MT5.append(data)
            returns_MT5.append(data[s.name])
        except Exception as e:
            print("---------")
            print(f"An error occurred in downloading {s.name}: {e}")
            traceback.print_exc()
            print("---------")
            
    returnsTV = pd.concat(returns_TV, axis=1)
    returnsMT5= pd.concat(returns_MT5, axis=1)
    returnsTV.fillna(0,inplace=True)
    returnsMT5.fillna(0,inplace=True)
    
    # rates_MT5_cnt = pd.concat(rates_MT5, axis=1)
    # # Find the first non-NaN row
    # first_valid_index = rates_MT5_cnt.apply(lambda x: x.first_valid_index()).max()

    # # Find the last non-NaN row
    # last_valid_index = rates_MT5_cnt.apply(lambda x: x.last_valid_index()).min()

    # # Slice the DataFrame from the first non-NaN row to the last non-NaN row
    # clean_rates_MT5_cnt = rates_MT5_cnt.loc[first_valid_index:last_valid_index]
    # #clean_rates_MT5_cnt = clean_rates_MT5_cnt.interpolate(method='polynomial', order=3)
        
    risk_free_rate = 0.03
    tol = None
#%%    
    #-----------Optimization---------------------------------
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
    print("7.  Markowitz Min-Risk + Max-Sharpe.")
    print("8.  Markowitz Min-Risk + Max-Sortino.")
    print("9.  Markowitz Min-MXDDP + Max-Sharpe.")
    print("10. EKO Min-Surprise + Max-Sharpe.")
    print("11. EKO Min-Surprise + Max-Sortino.")
    print("-------------------------")
    otp_sel = int(input("Which type of opt you wish: "))
#%%
    optimizerTV = ekoptim(returnsTV, risk_free_rate, target_SR,
                        target_Return, target_Volat, max_weight,tol,
                        full_rates = rates_MT5)

#%%
    print("Optimization started, please wait...")
    optimized_weights_TV = optimizerTV.optiselect(otp_sel)
    #-----------Optimization---------------------------------
    print("Sum of the weights: ", optimized_weights_TV.sum())
    threshold = min_tresh
    
    print("***********************")
    metrics = optimizerTV.calculate_metrics(optimized_weights_TV,0.85)    
    print("Sharpe Ratio: ", metrics['Sharpe'])
    print("Return: ", round(metrics['Return']*1000)/10, "%")
    print("Volatility: ", metrics['Risk'])
    print("Sortino: ", metrics['Sortino'])
    print("Surprise: ", metrics['Surprise'])
    print("CVAR: ", metrics['CVAR'])
    print("Maximum DrawDown%: ", 100*metrics['MXDDP'])
    print("***********************")

    equity_div = []
    returns_selected = []
    for i, weight in enumerate(optimized_weights_TV):
        try:
            symbol_info = mt5.symbol_info(returnsTV.columns[i])
            if symbol_info.bid<=0:
                kapla=rates_MT5[i]['close'][-1]
            else:
                kapla=symbol_info.bid
            if (weight>threshold and not(symbol_info.volume_min*kapla>
                                          1.1*total_equity*weight)):#
                print("Name: ",symbol_info.name,", Bid: ",
                      kapla,", step: ",symbol_info.volume_step)
                equity_div_x = {"symbol": symbol_info.name,"Weight":round(weight*10000)/100,
                                "Allocation": round(max(symbol_info.volume_min,
                                                  round(total_equity*weight/kapla,
                                                        -int(np.floor(np.log10(symbol_info.volume_step))+
                                                              1)+1))*kapla),
                                "Volume": max(symbol_info.volume_min,
                                              round(total_equity*weight/kapla,
                                                                          -int(np.floor(np.log10(symbol_info.volume_step))+
                                                                                1)+1))}
                returns_selected.append(returnsTV[returnsTV.columns[i]])
                equity_div.append(equity_div_x)
        except:
            print("An exception happened!")
    
    # returns_selected = []
    # for i, weight in enumerate(optimized_weights_TV):    
    #     if (weight>threshold):
    #         equity_div_x = {"symbol": returnsTV.columns[i],
    #                         "Weight":round(weight*10000)/100,
    #                         "Allocation": total_equity*weight}
    #         equity_div.append(equity_div_x)
    #         returns_selected.append(returnsTV[returnsTV.columns[i]])
    
    equity_div_df = pd.DataFrame(equity_div)
    equity_div_df.sort_values(by="Weight",inplace=True, ignore_index=True)
    returns_selected = pd.concat(returns_selected, axis=1)
    xyz = optimizerTV.cov2corr(100*LedoitWolf().fit(returns_selected).covariance_)
    selected_symb = equity_div_df['symbol'].tolist()
    print("------------------")
    print(equity_div_df)
    print("------------------")
    print("All the equity needed: ",equity_div_df["Allocation"].sum())
    # use Monte Carlo simulation to generate multiple sets of random weights
    optimizerTV.frontPlot(optimized_weights_TV, save=False)
    # shut down connection to the MetaTrader 5 terminal
#%%
    Dqp = 64 # past days for deep learning    
    Dyp = 16 # past days
    Dyf = 8 # future days
    n_t = 2 # tile size
    xhh = 0.01
    xhl = 0.5*xhh    
    xlh = -xhl
    xll = -xhh
    optimizerTV.Prepare_Data(tile_size=(n_t,int(n_t*Dqp/4)),xrnd=1e-3,#(n*Dqp->m=n*Dqp/4)
                             Selected_symbols=selected_symb[-1],
                             Dqp=Dqp, Dyp=Dyp, Dyf=Dyf, Thi=1,
                             hh=xhh,#
                             hl=xhl,#
                             lh=xlh,#
                             ll=xll) #None
    alphax = optimizerTV.HNrates[0]
    cetax = optimizerTV.selected_rates
    optimizerTV.draw_states(cetax[0])
#%%    
    # Put the data you need in a dictionary
    data = {
        'optimizerTV': optimizerTV,
        'rates_TV': rates_TV,
        'rates_MT5': rates_MT5,
        'selected_symb': selected_symb,
        # Add all the variables you need
    }
    
    # Get the current date and time
    now = datetime.datetime.now()
    
    # Format the date and time as a string
    date_time = now.strftime("%Y%m%d_%H%M%S")
    
    # Save the data to a file
    with open(f'data_{date_time}.pkl', 'wb') as f:
        dill.dump(data, f)
#%%
    optimizerTV.NNmake(num_inps = 1, learning_rate=0.001, epochs=1000, batch_size=32,
                       k_n=None, f1_method='macro', f1_w=False, mcc_w=False, filters=64,
                       load_train=False)
#%%
    optimizerTV.load_model_fit()
    optimizerTV.predict_all('close')
    betax = optimizerTV.Predicted_Rates
#%%
    mt5.shutdown()
