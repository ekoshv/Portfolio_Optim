import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from concurrent.futures import ThreadPoolExecutor

class ekoptim():
    def __init__(self, returns, risk_free_rate,
                               target_SR, target_Return, target_Volat,
                               max_weight):
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
        
        #define constraints
        self.bounds = [(0,1) for i in range(self.n)]
        
        self.constraints = [{"type":"eq","fun":lambda x: x.sum() - 1},
                            {"type":"ineq",
                             "fun":lambda x: self.return_cnt(x)-0.85*self.target_Return},
                            {"type":"ineq",
                             "fun":lambda x: -self.return_cnt(x)+1.15*self.target_Return},
                            {"type":"ineq",
                             "fun":lambda x: self.sharpe_ratio_cnt(x)-0.85*self.target_SR},
                            {"type":"ineq",
                             "fun":lambda x: -self.sharpe_ratio_cnt(x)+1.15*self.target_SR},
                            {"type":"ineq",
                             "fun":lambda x: 1.15*self.target_Volat-self.risk_cnt(x)},
                            {"type":"ineq",
                             "fun":lambda x: -0.85*self.target_Volat+self.risk_cnt(x)},
                            {"type":"ineq",
                             "fun":lambda x: self.max_weight-x}]
        
        #self.constraints_weight = [{"type":"eq","fun":lambda x: x.sum() - 1},
                              #{"type":"ineq","fun":lambda x: self.max_weight-x}]
    
    
    #define the optimization functions    
    def initial_weight(self, w0):
        self.w0 = w0
    
        
    def risk_cnt(self, w):
        portfolio_volatility = (w.T @ LedoitWolf().fit(self.returns).covariance_ @ w)**0.5 * np.sqrt(self.days)
        return portfolio_volatility
    
    def sharpe_ratio_cnt(self, w):
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        portfolio_volatility = (w.T @ LedoitWolf().fit(self.returns).covariance_ @ w)**0.5 * np.sqrt(self.days)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe_ratio
    
    def return_cnt(self, w):
        portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
        return portfolio_return

    def markowitz_optimization_return(self):
        #run the optimization
        fn = lambda x:  (math.exp(-(self.return_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,5,6,7]])
        optimized_weights = result.x
        return optimized_weights

    def markowitz_optimization_risk(self):
        #run the optimization
        fn = lambda x:  (math.exp((self.risk_cnt(x))))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]])
        optimized_weights = result.x
        return optimized_weights

    def markowitz_optimization_risk_sharpe(self):
        #run the optimization
        fn = lambda x:  (math.exp(self.risk_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=[self.constraints[i] for i in [0,7]])
        optimized_weights = result.x
        return optimized_weights

    def markowitz_optimization_risk_sharpe_cnt(self):
        #run the optimization
        fn = lambda x:  (math.exp(self.risk_cnt(x))+
                         math.exp(-self.sharpe_ratio_cnt(x)))
        result = minimize(fn, self.w0,
                          method='SLSQP', bounds=self.bounds,
                          constraints=self.constraints)
        optimized_weights = result.x
        return optimized_weights
# end of class ekoptim