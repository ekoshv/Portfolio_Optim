import numpy as np
#import pandas as pd
import math
#import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
#from concurrent.futures import ThreadPoolExecutor

class ekoptim():
    def __init__(self, returns, risk_free_rate,
                               target_SR, target_Return, target_Volat,
                               max_weight,toler):
        self.returns = returns
        self.target_SR = target_SR
        self.target_Return = target_Return
        self.target_Volat = target_Volat
        self.max_weight = max_weight
        self.n = returns.shape[1]
        self.days = returns.shape[0]
        #initialize starting point
        self.w0 = [1/self.n] * self.n
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
        
        #self.constraints_weight = [{"type":"eq","fun":lambda x: x.sum() - 1},
        #{"type":"ineq","fun":lambda x: self.max_weight-x}]
        #,{"type":"ineq","fun":lambda x: (self.max_numb-len([sx for sx in x 
        #if sx>=self.tresh]))}
    
    
    #define the optimization functions    
    def __initial_weight(self, w0):
        self.w0 = w0
    #-------------------------------
    #---Risk, Sharpe,Sortino, Return--------
    #-------------------------------        
    def risk_cnt(self, w):
        portfolio_volatility = (w.T @ LedoitWolf().fit(self.returns).
                                covariance_ @ w)**0.5 * np.sqrt(self.days)
        return portfolio_volatility
    
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
        optimized_weights = result.x
        return optimized_weights
    
    def Optim_return_cnt_sharpe(self):#2
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,3,4,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights

    def Optim_return_cnt_volat(self):#3
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,5,6,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights
    #---Risk---
    def Optim_risk_nl(self):#4
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights

    def Optim_risk_cnt_sharpe(self):#5
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,3,4,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights

    def Optim_risk_cnt_return(self):#6
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,1,2,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights
    #---Markowitz Original---
    def markowitz_optimization_risk_sharpe(self):#7
        #run the optimization
        fn = lambda x:  (math.exp(self.risk_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]],
                          tol = self.toler)
        optimized_weights = result.x
        return optimized_weights

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
            else:
                return -1
        except:
            print("An exception occurred in Optimization")
# end of class ekoptim
