import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from sklearn.covariance import LedoitWolf
import traceback
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ekoptim():
    def __init__(self, returns, risk_free_rate,
                               target_SR, target_Return, target_Volat,
                               max_weight,toler,
                               full_rates,Dyp=120, Dyf=30, Thi=5):
        self.returns = returns
        self.target_SR = target_SR
        self.target_Return = target_Return
        self.target_Volat = target_Volat
        self.max_weight = max_weight
        self.n = returns.shape[1]
        self.days = returns.shape[0]
        #initialize starting point
        self.w0 = [1/self.n] * self.n
        self.optimized_weights = [1/self.n] * self.n
        self.durc = self.days/252
        self.risk_free_rate = risk_free_rate*self.durc
        self.toler = toler
        
        #define constraints
        self.bounds = [(0,1) for i in range(self.n)]
        
        self.constraints = [{"type":"eq",#0 sum = 1
                             "fun":lambda x: x.sum() - 1},
                            {"type":"ineq",#1 lb<return<ub @15%
                             "fun":lambda x: self.return_cnt(x)-0.85*self.target_Return},
                            {"type":"ineq",#2
                             "fun":lambda x: -self.return_cnt(x)+1.15*self.target_Return},
                            {"type":"ineq",#3 lb<sharpe<ub @15%
                             "fun":lambda x: self.sharpe_ratio_cnt(x)-0.85*self.target_SR},
                            {"type":"ineq",#4
                             "fun":lambda x: -self.sharpe_ratio_cnt(x)+1.15*self.target_SR},
                            {"type":"ineq",#5 lb<volat<ub @15%
                             "fun":lambda x: 1.15*self.target_Volat-self.risk_cnt(x)},
                            {"type":"ineq",#6
                             "fun":lambda x: -0.85*self.target_Volat+self.risk_cnt(x)},
                            {"type":"ineq",#7 max weight
                             "fun":lambda x: self.max_weight-x}]
    
        self.Dyp = Dyp   # Number of past days to consider in the moving horizon
        self.Dyf = Dyf    # Number of future days to predict in the moving horizon
        self.Thi = Thi   # Time horizon interval (in days)
        self.full_rates = full_rates
        self.nnmodel = tf.keras.Sequential()
        
    #define the optimization functions    
    def __initial_weight(self, w0):
        self.w0 = w0

    def cov2corr(self, cov):
        corr = np.zeros_like(cov)
        n = cov.shape[0]
        
        for i in range(n):
            corr[i, i] = 1.0
            
            for j in range(i + 1, n):
                if cov[i, i] == 0.0 or cov[j, j] == 0.0:
                    corr[i, j] = corr[j, i] = 0.0
                else:
                    corr[i, j] = corr[j, i] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        
        return corr
    
    # Define a function that applies the moving horizon to a given dataframe
    def apply_moving_horizon(self,df,smb):
        new_df = pd.DataFrame()
        for i in range(self.Dyp, len(df)-self.Dyf+1, self.Thi):
            past_data = df[smb].iloc[i-self.Dyp:i]
            future_data = df[smb].iloc[i:i+self.Dyf]
            new_row = {
                'past_data': past_data,
                'future_data': future_data,
            }
            new_df = new_df.append(new_row, ignore_index=True)
        return new_df

    def normalize(self,data):
        #Normalize a pandas series by scaling its values to the range [0, 1].
        return (data - data.min()) / (data.max() - data.min())

    def apply_moving_horizon_norm(self,df,smb):
        new_df = []
        for i in range(self.Dyp, len(df)-self.Dyf+1, self.Thi):
            past_data = df[smb].iloc[i-self.Dyp:i]
            past_data_normalized = self.normalize(past_data)
            future_data = df[smb].iloc[i:i+self.Dyf]
            future_data_rescaled = ((future_data - past_data.min()) /
                                    (past_data.max() - past_data.min()))
            new_row = {
                'past_data': past_data_normalized.values,
                'future_data': future_data_rescaled.values,
            }
            #print(new_row)
            new_df.append(new_row)
        return new_df
    
    #---------------------------------------------------
    #---Neural Network ---------------------------------
    #---------------------------------------------------     
    def Hrz_Nrm(self,smb):
        # Apply the moving horizon to each dataframe in rates_lists
        return [self.apply_moving_horizon_norm(df,smb) for df in self.full_rates]
    
    def NNmake(self):
        
        HNrates = self.Hrz_Nrm('close')
        # Define the neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.Dyp, 1)),
            
            tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(self.Dyf)
        ])

        # Compile the model with mean squared error loss
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        #model.compile(optimizer=opt, loss='mse')
        model.compile(optimizer=opt, loss='mse')

        # Train the model on the data in new_rates_lists
        X = np.array([d['past_data'] for lst in HNrates for d in lst])
        X = np.expand_dims(X, axis=-1)  # add a new axis for the input feature
        #X_in = X.reshape(X.shape[0],1,X.shape[1])
        y = np.array([d['future_data'] for lst in HNrates for d in lst])
        #y_in = y.reshape(y.shape[0],1,y.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        model.fit(X_train, y_train, epochs=100, batch_size=8,
                  validation_split=0.33, shuffle=True)
        self.nnmodel = model
        
    #---------------------------------------------------
    #---Risk, Sharpe, Sortino, Return, Surprise --------
    #---------------------------------------------------        
    def risk_cnt(self, w):
        portfolio_volatility = (w.T @ LedoitWolf().fit(self.returns).
                                covariance_ @ w)**0.5 * np.sqrt(self.days)
        return portfolio_volatility

    def cvar_cnt(self, w, alpha):
        # Calculate the conditional value-at-risk (CVaR) of the portfolio
        # with confidence level alpha
        
        # Calculate the portfolio return and volatility
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        portfolio_volatility = (w.T @ LedoitWolf().fit(self.returns).
                                covariance_ @ w)**0.5 * np.sqrt(self.days)
        
        # Calculate the VaR of the portfolio using the normal distribution
        #z_alpha = norm.ppf(alpha)
        #portfolio_var = portfolio_return - z_alpha * portfolio_volatility
        
        # Calculate the expected shortfall (ES) of the portfolio
        portfolio_es = -1/alpha * (1 - alpha) * \
            norm.pdf(norm.ppf(alpha)) * portfolio_volatility
        portfolio_cvar = portfolio_return - portfolio_es
        
        return portfolio_cvar

    def surprise_cnt(self, w):
        # Calculate the percentage change between consecutive returns
        delta_returns = self.returns.pct_change().fillna(0)
        
        # Calculate the absolute percentage change between consecutive returns
        aks = abs(delta_returns.replace([np.inf, -np.inf], 0))
        
        # Calculate the log of the absolute percentage change plus one
        #aks_log = np.log(aks + 1)
        aks_sqrt = np.sqrt(aks+1)
        
        # Calculate the correlation matrix of the log-returns adjusted for the absolute percentage change between consecutive returns
        covar = LedoitWolf().fit(self.returns*aks_sqrt).covariance_
        #corr = self.cov2corr(covar)
        
        # Calculate the portfolio surprise
        portfolio_surprise = (w.T @ covar @ w)**0.5 * np.sqrt(self.days)
        
        return portfolio_surprise
    
    def sharpe_ratio_cnt(self, w):
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        portfolio_volatility = ((w.T @ LedoitWolf().fit(self.returns).covariance_ @ w)**0.5 *
                                np.sqrt(self.days))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe_ratio
    
    def return_cnt(self, w):
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        return portfolio_return
    
    def sortino_ratio_cnt(self, w):
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        downside_returns = self.returns[self.returns.dot(w) < 0].dot(w)
        downside_volatility = np.sqrt((downside_returns**2).mean()) * np.sqrt(self.days)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_volatility
        return sortino_ratio
    
    def maximum_drawdown_cnt(self, w):
        # Calculate the maximum drawdown of the portfolio
        
        portfolio_return = (self.returns @ w).cumsum()
        portfolio_peak = np.maximum.accumulate(portfolio_return)
        drawdown = portfolio_peak - portfolio_return
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown / portfolio_peak.max()
        
        return max_drawdown_pct

    #-------------------------------
    #---Optimizations---------------
    #-------------------------------
    
    #---Return---
    def Optim_return_nl(self):#1
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights
    
    def Optim_return_cnt_sharpe(self):#2
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,3,4,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def Optim_return_cnt_volat(self):#3
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,5,6,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights
    #---Risk---
    def Optim_risk_nl(self):#4
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def Optim_risk_cnt_sharpe(self):#5
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,3,4,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def Optim_risk_cnt_return(self):#6
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,1,2,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights
    #---Markowitz Original---
    def markowitz_optimization_risk_sharpe(self):#7
        #run the optimization
        fn = lambda x:  (math.exp(self.risk_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def markowitz_optimization_risk_sortino(self):#8
        #run the optimization
        fn = lambda x:  (math.exp(self.risk_cnt(x))+
                         math.exp(-self.sortino_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def markowitz_optimization_mxddp_sharpe(self):#9
        #run the optimization
        fn = lambda x:  (math.exp(self.maximum_drawdown_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights
    #---Surprise---
    def surprise_sharpe_optimization(self):#10
        #run the optimization
        fn = lambda x:  (math.exp(self.surprise_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights

    def surprise_sortino_optimization(self):#11
        #run the optimization
        fn = lambda x:  (math.exp(self.surprise_cnt(x))+
                         math.exp(-self.sortino_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        self.optimized_weights = result.x
        return self.optimized_weights
    #-------------------------------
    #---selections of Stocks--------
    #-------------------------------

    def optiselect(self, sel=7):
        try:
            if sel==1:
                return self.Optim_return_nl()
            elif sel==2:
                return self.Optim_return_cnt_sharpe()
            elif sel==3:
                return self.Optim_return_cnt_volat()
            elif sel==4:
                return self.Optim_risk_nl()
            elif sel==5:
                return self.Optim_risk_cnt_sharpe()
            elif sel==6:
                return self.Optim_risk_cnt_return()
            elif sel==7:
                return self.markowitz_optimization_risk_sharpe()
            elif sel==8:
                return self.markowitz_optimization_risk_sortino()
            elif sel==9:
                return self.markowitz_optimization_mxddp_sharpe()
            elif sel==10:
                return self.surprise_sharpe_optimization()
            elif sel==11:
                return self.surprise_sortino_optimization()
            else:
                return -1
        except Exception as e:
            print("Caught an exception:")
            traceback.print_exc()
    
    def calculate_metrics(self,w, alpha):
        return {'Risk': self.risk_cnt(w),
                'Return': self.return_cnt(w),
                'Sharpe': self.sharpe_ratio_cnt(w),
                'Sortino': self.sortino_ratio_cnt(w),
                'Surprise': self.surprise_cnt(w),
                'CVAR': self.cvar_cnt(w, alpha),
                'MXDDP': self.maximum_drawdown_cnt(w)}

    def frontPlot(self, w, alpha=0.95, save=False):
        # use Monte Carlo simulation to generate multiple sets of random weights
        num_portfolios = 500
        returns_listx = []
        sharpe_ratios_listx = []
        volatilities_listx = []
        for i in range(num_portfolios):
            weights = w+(2*np.random.random(self.n)-1)/(3*np.sqrt(self.n)+1)
            weights /= np.sum(weights)
            portfolio_returnx = self.return_cnt(weights)
            portfolio_volatilityx = self.risk_cnt(weights)
            sharpe_ratiox = self.sharpe_ratio_cnt(weights)
            returns_listx.append(portfolio_returnx)
            volatilities_listx.append(portfolio_volatilityx)
            sharpe_ratios_listx.append(sharpe_ratiox)
        # plot the efficient frontier
        metrics = self.calculate_metrics(w,alpha)#self.optimized_weights
        data = {'Volatility': volatilities_listx, 'Return': returns_listx, 'Sharpe Ratio': sharpe_ratios_listx}
        data = pd.DataFrame(data)
        sns.scatterplot(data=data, x='Volatility', y='Return', hue='Sharpe Ratio', palette='viridis')
        plt.scatter(metrics['Risk'], metrics['Return'], c='red', marker='D', s=200)
        plt.xlabel(f'Volatility\nOptimun found at Sharpe Ratio: {metrics["Sharpe"]:.2f}, Risk: {metrics["Risk"]:.2f}')
        plt.ylabel('Return')
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'efficient_frontier_{current_time}.png'
        if save:
            plt.savefig(file_name, dpi=300)
        plt.show()
# end of class ekoptim