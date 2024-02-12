import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import math
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet
import traceback

class ekoptim():
    def __init__(self, returns, risk_free_rate,
                 target_SR, target_Return,
                 target_Volat, cvar_alpha,
                 max_weight, toler):
        try:
            self.returns = returns  # Set returns
            self.target_SR = target_SR  # Set target Sharpe Ratio
            self.target_Return = target_Return  # Set target return
            self.target_Volat = target_Volat  # Set target volatility
            self.cvar_alpha = cvar_alpha # Set Alpha tail of CVaR
            self.max_weight = max_weight  # Set maximum weight
            self.n = returns.shape[1]  # Number of assets
            self.days = returns.shape[0]  # Number of days
            # Initialize starting point (equal weight)
            self.w0 = [1 / self.n] * self.n
            # Initialize optimized_weights (equal weight)
            self.optimized_weights = [1 / self.n] * self.n
            self.durc = self.days / 252  # Calculate duration
            # Calculate risk-free rate adjusted for duration
            self.risk_free_rate = risk_free_rate * self.durc
            self.toler = toler  # Set tolerance
    
            # Define constraints
            self.bounds = [(0, 1) for i in range(self.n)]  # Set bounds for weights
    
            self.constraints = [
                {"type": "eq",  # Constraint 1: Sum of weights equals 1
                 "fun": lambda x: x.sum() - 1},
                {"type": "ineq",  # Constraint 2: Lower bound on return
                 "fun": lambda x: self.return_cnt(x) - 0.85 * self.target_Return},
                {"type": "ineq",  # Constraint 3: Upper bound on return
                 "fun": lambda x: -self.return_cnt(x) + 1.15 * self.target_Return},
                {"type": "ineq",  # Constraint 4: Lower bound on Sharpe Ratio
                 "fun": lambda x: self.sharpe_ratio_cnt(x) - 0.85 * self.target_SR},
                {"type": "ineq",  # Constraint 5: Upper bound on Sharpe Ratio
                 "fun": lambda x: -self.sharpe_ratio_cnt(x) + 1.15 * self.target_SR},
                {"type": "ineq",  # Constraint 6: Lower bound on volatility
                 "fun": lambda x: 1.15 * self.target_Volat - self.risk_cnt(x)},
                {"type": "ineq",  # Constraint 7: Upper bound on volatility
                 "fun": lambda x: -0.85 * self.target_Volat + self.risk_cnt(x)},
                {"type": "ineq",  # Constraint 8: Maximum weight constraint
                 "fun": lambda x: self.max_weight - x}
            ]
    
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    def __initial_weight(self, w0):
        self.w0 = w0

    def cov2corr(self, cov):
        # Initialize an empty correlation matrix with the same shape as the covariance matrix
        corr = np.zeros_like(cov)
        # Get the number of assets
        n = cov.shape[0]
    
        # Iterate through the rows of the matrix
        for i in range(n):
            # Set the diagonal elements to 1.0 (correlation of an asset with itself)
            corr[i, i] = 1.0
    
            # Iterate through the columns of the matrix, starting from the next element in the row
            for j in range(i + 1, n):
                # Check if the variance of the assets i and j is zero
                if cov[i, i] == 0.0 or cov[j, j] == 0.0:
                    # Set the correlation between assets i and j to zero if either variance is zero
                    corr[i, j] = corr[j, i] = 0.0
                else:
                    # Calculate and set the correlation between assets i and j
                    corr[i, j] = corr[j, i] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    
        # Return the correlation matrix
        return corr
 
    #---------------------------------------------------
    #---Risk, Sharpe, Sortino, Return, Surprise --------
    #---------------------------------------------------        
    def risk_cnt(self, w):
        try:
            # Calculate the covariance matrix using the LedoitWolf method
            covariance_matrix = LedoitWolf().fit(self.returns).covariance_
    
            # Compute the portfolio volatility by multiplying the weights,
            # covariance matrix, and weights transposed, then taking the square root
            portfolio_volatility = np.sqrt(w.T @ covariance_matrix @ w) * np.sqrt(self.days)
    
            # Return the portfolio volatility
            return portfolio_volatility
        except Exception as e:
            print(f"An error occurred in risk_cnt: {e}")
            traceback.print_exc()
            return None

    # Custom distribution based on Beta distribution
    def custom_distribution(self, x, a=2, b=5):
        """
        Calculate the PDF and CDF of the custom distribution for a given x.
        
        Parameters:
        - x: The value at which to calculate the PDF and CDF.
        - a: The alpha parameter for the Beta distribution.
        - b: The beta parameter for the Beta distribution.
        
        Returns:
        - pdf: The probability density function value at x.
        - cdf: The cumulative distribution function value at x.
        """
        pdf = beta.pdf(x, a, b)
        cdf = beta.cdf(x, a, b)
        return pdf, cdf    
    
    def find_var_point_custom(self, alpha, a=2, b=5):
        """
        Find the Value at Risk (VaR) point for a given confidence level alpha using the custom distribution.
        
        Parameters:
        - alpha: The confidence level (e.g., 0.95 for 95% confidence).
        - a, b: Parameters of the beta distribution.
        
        Returns:
        - var_point: The Value at Risk point from the custom distribution.
        """
        # Calculate the inverse CDF (quantile) at 1 - alpha for the custom distribution
        var_point = beta.ppf(1 - alpha, a, b)
        return var_point
    
    def normalize_data(self, data):
        """
        Normalize data to the [0, 1] interval.
        
        Parameters:
        - data: The data to normalize, as a numpy array.
        
        Returns:
        - normalized_data: The normalized data.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        return normalized_data, data_min, data_max
    
    def fit_beta_distribution(self, data):
        """
        Fit a beta distribution to the data and return the estimated parameters.
        
        Parameters:
        - data: The data to fit, normalized to the [0, 1] interval.
        
        Returns:
        - a, b: The estimated parameters of the beta distribution.
        """
        a, b, _, _ = beta.fit(data, floc=0, fscale=1)  # Fix the location to 0 and scale to 1
        return a, b

    
    def cvar_ctm_pdf_cnt(self, w, alpha):
        try:
            # Calculate the portfolio return
            portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
    
            # Calculate the portfolio volatility using the LedoitWolf covariance matrix
            covariance_matrix = LedoitWolf().fit(self.returns).covariance_
            portfolio_volatility = np.sqrt(w.T @ covariance_matrix @ w) * np.sqrt(self.days)
            
            # self.returns is a numpy array of portfolio returns
            normalized_returns, data_min, data_max = self.normalize_data(self.returns)
            a, b = self.fit_beta_distribution(normalized_returns)
            
            # Find the VaR point using the custom distribution
            var_point = self.find_var_point_custom(alpha, a, b)
    
            # Calculate the PDF and CDF at the VaR point for the custom distribution
            pdf, cdf = self.custom_distribution(var_point, a, b)
            portfolio_es = -1 / alpha * (1 - cdf) * pdf * portfolio_volatility
    
            # Calculate the conditional value-at-risk (CVaR) of the portfolio
            portfolio_cvar = portfolio_return - portfolio_es
    
            return portfolio_cvar
        except Exception as e:
            print(f"An error occurred in cvar_cnt_custom: {e}")
            return None
    
    def cvar_cnt(self, w, alpha):
        try:
            # Calculate the conditional value-at-risk (CVaR) of the portfolio
            # with confidence level alpha
    
            # Calculate the portfolio return
            portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
    
            # Calculate the portfolio volatility using the LedoitWolf covariance matrix
            covariance_matrix = LedoitWolf().fit(self.returns).covariance_
            portfolio_volatility = np.sqrt(w.T @ covariance_matrix @ w) * np.sqrt(self.days)
    
            # Calculate the expected shortfall (ES) of the portfolio
            portfolio_es = (
                -1 / alpha
                * (1 - alpha)
                * norm.pdf(norm.ppf(alpha))
                * portfolio_volatility
            )
    
            # Calculate the conditional value-at-risk (CVaR) of the portfolio
            portfolio_cvar = portfolio_return - portfolio_es
    
            return portfolio_cvar
        except Exception as e:
            print(f"An error occurred in cvar_cnt: {e}")
            traceback.print_exc()
            return None

    def surprise_cnt(self, w):
        try:
            # Calculate the percentage change between consecutive returns
            delta_returns = self.returns.pct_change().fillna(0)
    
            # Calculate the absolute percentage change between consecutive returns
            aks = abs(delta_returns.replace([np.inf, -np.inf], 0))
    
            # Calculate the log of the absolute percentage change plus one
            aks_log = np.log(aks + 1)
    
            # Calculate the covariance matrix of the returns adjusted for the log of the absolute percentage change
            covar = MinCovDet().fit(self.returns * aks_log).covariance_
    
            # Calculate the portfolio surprise using the covariance matrix
            portfolio_surprise = (w.T @ covar @ w)**0.5 * np.sqrt(self.days)
    
            return portfolio_surprise
        except Exception as e:
            print(f"An error occurred in surprise_cnt: {e}")
            traceback.print_exc()
            return None

    def sharpe_ratio_cnt(self, w):
        try:
            # Calculate the portfolio return using the weighted average of asset returns and risk-free rate
            portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
    
            # Calculate the portfolio volatility using the weighted covariance matrix of asset returns
            portfolio_volatility = (
                (w.T @ LedoitWolf().fit(self.returns).covariance_ @ w)**0.5 * np.sqrt(self.days)
            )
    
            # Calculate the Sharpe ratio by dividing the excess return by the portfolio volatility
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
            return sharpe_ratio
        except Exception as e:
            print(f"An error occurred in sharpe_ratio_cnt: {e}")
            traceback.print_exc()
            return None
 
    def return_cnt(self, w):
        try:
            # Calculate the portfolio return using the weighted average of asset returns and risk-free rate
            portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
    
            return portfolio_return
        except Exception as e:
            print(f"An error occurred in return_cnt: {e}")
            traceback.print_exc()
            return None
    
    def sortino_ratio_cnt(self, w):
        try:
            # Calculate the portfolio return
            portfolio_return = w.T @ self.returns.mean() * self.days - self.risk_free_rate
            
            # Filter the returns to only include negative returns
            downside_returns = self.returns[self.returns.dot(w) < 0].dot(w)
            
            # Calculate the downside volatility by taking the square root of the mean of squared negative returns
            downside_volatility = np.sqrt((downside_returns**2).mean()) * np.sqrt(self.days)
            
            # Calculate the Sortino ratio using portfolio return, risk-free rate, and downside volatility
            sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_volatility
            
            return sortino_ratio
        except Exception as e:
            print(f"An error occurred in sortino_ratio_cnt: {e}")
            traceback.print_exc()
            return None
    
    def maximum_drawdown_cnt(self, w):
        try:
            # Calculate the cumulative returns of the portfolio
            portfolio_return = (self.returns @ w).cumsum()
    
            # Identify the running maximum value of the portfolio returns
            portfolio_peak = np.maximum.accumulate(portfolio_return)
    
            # Calculate drawdowns by subtracting the current return from the peak value
            drawdown = portfolio_peak - portfolio_return
    
            # Find the maximum drawdown value
            max_drawdown = np.max(drawdown)
    
            # Calculate the maximum drawdown percentage
            max_drawdown_pct = max_drawdown / portfolio_peak.max()
    
            return max_drawdown_pct
        except Exception as e:
            print(f"An error occurred in maximum_drawdown_cnt: {e}")
            traceback.print_exc()
            return None

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
            print(f"Caught an exception: {e}")
            traceback.print_exc()
            return None
    
    def calculate_metrics(self,w, alpha):
        return {'Risk': self.risk_cnt(w),
                'Return': self.return_cnt(w),
                'Sharpe': self.sharpe_ratio_cnt(w),
                'Sortino': self.sortino_ratio_cnt(w),
                'Surprise': self.surprise_cnt(w),
                'CVAR': self.cvar_cnt(w, alpha),
                'MXDDP': self.maximum_drawdown_cnt(w)}

    def frontPlot(self, w, alpha_cvar=0.95, save=False):
        try:
            # Use Monte Carlo simulation to generate multiple sets of random weights
            num_portfolios = 500
            returns_listx = []
            sharpe_ratios_listx = []
            volatilities_listx = []
    
            for i in range(num_portfolios):
                # Generate random weights and normalize them
                weights = w + (2 * np.random.random(self.n) - 1) / (3 * np.sqrt(self.n) + 1)
                weights /= np.sum(weights)
    
                # Calculate portfolio return, volatility, and Sharpe ratio
                portfolio_returnx = self.return_cnt(weights)
                portfolio_volatilityx = self.risk_cnt(weights)
                sharpe_ratiox = self.sharpe_ratio_cnt(weights)
    
                # Store the results in lists
                returns_listx.append(portfolio_returnx)
                volatilities_listx.append(portfolio_volatilityx)
                sharpe_ratios_listx.append(sharpe_ratiox)
    
            # Calculate the optimized metrics
            metrics = self.calculate_metrics(w, alpha_cvar)
    
            # Prepare data for plotting
            data = {'Volatility': volatilities_listx, 'Return': returns_listx, 'Sharpe Ratio': sharpe_ratios_listx}
            data = pd.DataFrame(data)
    
            # Create a scatterplot of the efficient frontier
            sns.scatterplot(data=data, x='Volatility', y='Return', hue='Sharpe Ratio', palette='viridis')
            # Plot the optimized point on the graph
            plt.scatter(metrics['Risk'], metrics['Return'], c='red', marker='D', s=200)
    
            # Set the labels for the axes
            plt.xlabel(f'Volatility\nOptimum found at Sharpe Ratio: {metrics["Sharpe"]:.2f}, Risk: {metrics["Risk"]:.2f}')
            plt.ylabel('Return')
    
            # Save the graph to a file if specified
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'efficient_frontier_{current_time}.png'
            if save:
                plt.savefig(file_name, dpi=300)
    
            # Display the graph
            plt.show()
    
        except Exception as e:
            print(f"An error occurred in frontPlot: {e}")
            traceback.print_exc()

# end of class ekoptim
