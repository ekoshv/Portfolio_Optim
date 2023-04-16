import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import math
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import traceback
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pywt
from tqdm import tqdm


class ekoptim():
    def __init__(self, returns, risk_free_rate,
                 target_SR, target_Return, target_Volat,
                 max_weight, toler,
                 full_rates, Dyp=120, Dyf=16, Thi=3):
        try:
            self.returns = returns  # Set returns
            self.target_SR = target_SR  # Set target Sharpe Ratio
            self.target_Return = target_Return  # Set target return
            self.target_Volat = target_Volat  # Set target volatility
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
            
            self.HNrates = []
            self.Predicted_Rates=[]
            self.Dyp = Dyp   # Number of past days to consider in the moving horizon
            self.Dyf = Dyf    # Number of future days to predict in the moving horizon
            self.Thi = Thi   # Time horizon interval (in days)
            self.full_rates = full_rates
            self.new_full_rates = []
            self.nnmodel = tf.keras.Sequential()
    
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
    #--- Neural Networks -----------
    #-------------------------------
    def r_squared(self, y_true, y_pred):
        # Calculate the residual sum of squares (numerator)
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        
        # Calculate the total sum of squares (denominator)
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        
        # Calculate R-squared
        r2 =  tf.divide(tf.exp(tf.subtract(1.0, tf.divide(residual, total))),tf.exp(1.0))
        
        return r2

    def reshape_nm(self, L):
    
        # Find the factors of L
        factors = [i for i in range(1, L+1) if L % i == 0]
    
        # Find the factor pair closest to a square shape
        mid = len(factors) // 2
        if len(factors) % 2 == 0:
            m, n = factors[mid-1], factors[mid]
        else:
            m = n = factors[mid]
    
        return n, m
    #--- Wavelets ------
    def decompose_and_flatten(self, data, wavelet):
        flattened_coeffs_total = []
        data_transposed = data.T  # Transpose the input data to iterate over columns
        for dx in data_transposed:
            coeffs = pywt.wavedec(dx, wavelet)
            lengths = [len(c) for c in coeffs]
            flattened_coeffs = np.concatenate(coeffs)
            flattened_coeffs_total.append(flattened_coeffs)
        return np.vstack(flattened_coeffs_total), lengths    
    
    def reconstruct_from_flattened(self, flattened_coeffs, wavelet, lengths):
        coeffs = []
        start = 0
        for length in lengths:
            coeffs.append(flattened_coeffs[start:start + length])
            start += length
        reconstructed_data = pywt.waverec(coeffs, wavelet)
        return reconstructed_data

    def create_2d_image(self, data, wavelet):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet)
        final_size = len(data)
        # Repeat coefficients to the same length as the original data
        repeated_coeffs = []
        for c in coeffs:
            repeated = np.repeat(c, len(data) // len(c))
            repeated_coeffs.append(repeated)
        
        # Stack the repeated coefficients to create a 2D image
        image = np.stack(repeated_coeffs)
        
        # Repeat the image to create a final square image
        repeat_x = final_size // image.shape[1]
        repeat_y = final_size // image.shape[0]
        final_image = np.tile(image, (repeat_y, repeat_x))
        
        return final_image

    # Define a function that applies the moving horizon to a given dataframe
    def apply_moving_horizon(self, df, smb):
        new_df = pd.DataFrame()
        for i in range(self.Dyp, len(df)-self.Dyf+1, self.Thi):
            past_data = df[smb].iloc[i-self.Dyp:i]
            future_data = df[smb].iloc[i:i+self.Dyf]
            flattened_coeffs, lengths = self.decompose_and_flatten(future_data.values, 'db1')
            new_row = {
                'past_data': past_data,
                'future_data': flattened_coeffs,
                'cLength': lengths
            }
            new_df = new_df.append(new_row, ignore_index=True)
        return new_df

    def normalize(self,data, dmn, dmx):
        #Normalize a pandas series by scaling its values to the range [0, 1].
        return (data - dmn) / (dmx - dmn), data.min(axis=0)['low'], data.max(axis=0)['high']

    def apply_moving_horizon_norm(self,df,smb,spn):
        new_df = []
        if isinstance(smb, str):
            smb_col = smb
        elif isinstance(smb, int):
            smb_col = df.columns[smb]
        else:
            raise ValueError("smb should be either a string or an integer.")

        for i in range(self.Dyp, len(df)-self.Dyf+1, self.Thi):
            past_data = df[['open','high','low','close']].iloc[i-self.Dyp:i]
            psdt_HH = past_data.max(axis=0)['high']
            psdt_LL = past_data.min(axis=0)['low']
            past_data_normalized, mindf, maxdf = self.normalize(past_data, psdt_LL, psdt_HH)
            past_data_normalized_w, lng = self.decompose_and_flatten(past_data_normalized.values,'db1')
            pst_dt_tiled = np.tile(past_data_normalized_w, (8,2))
            future_data = df[smb_col].iloc[i:i+self.Dyf]
            future_data_rescaled, fdmn, fdmx = self.normalize(future_data, psdt_LL, psdt_HH)
            signal = ((2 if future_data_rescaled.max() > 1.5 else 1 if 1.03 <=
                       future_data_rescaled.max() <= 1.5 else 0) +
                      (-2 if future_data_rescaled.min() < -1.5 else -1 if -1.5 <=
                       future_data_rescaled.min() <= -1.03 else 0))
            new_row = {
                'past_data': pst_dt_tiled,
                'future_data': future_data_rescaled,
                'signal': signal,
                'minmax': [psdt_LL,psdt_HH]
            }
            #print(new_row)
            new_df.append(new_row)
        return new_df        

    def Hrz_Nrm(self, smb, spn):
        # Apply the moving horizon to each dataframe in rates_lists
        return [self.apply_moving_horizon_norm(df, smb, spn) for 
                df in tqdm(self.full_rates, desc='Processing DataFrames')]

    def Prepare_Data(self, symb, spn):
        print("Preparing Data...")
        self.spn = spn
        self.HNrates = self.Hrz_Nrm(symb, spn)
        self.mz = self.HNrates[0][0]['past_data'].shape[0]
        self.nz = self.HNrates[0][0]['past_data'].shape[1]

    def create_model(self, image_height, image_width):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(image_height, image_width, 1)),
    
            tf.keras.layers.SeparableConv2D(filters=64,
                                   kernel_size=9, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
    
            tf.keras.layers.SeparableConv2D(filters=128,
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
    
            tf.keras.layers.SeparableConv2D(filters=256,
                                   kernel_size=5, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
    
            tf.keras.layers.GlobalAveragePooling2D(),
    
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
    
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
    
            tf.keras.layers.Dense(5, activation="softmax")
        ])
        return model

    def NNmake(self,
               learning_rate=0.001, epochs=100, batch_size=32,
               load_train=False):
       # Define the neural network
       model = self.create_model(self.mz, self.nz) 
       
       # Compile the model with mean squared error loss
       opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
       model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy', self.r_squared])

       #model.compile(optimizer=opt, loss='mape')
   
       # Set up the callback to save the best model weights
       checkpoint_dir = './checkpoints'
       if not os.path.exists(checkpoint_dir):
           os.makedirs(checkpoint_dir)
       filepath = checkpoint_dir + '/best_weights.hdf5'
       checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
   
       # Train the model on the data in new_rates_lists
       X = np.array([d['past_data'] for lst in self.HNrates for d in lst])
       X = np.expand_dims(X, axis=-1)  # add a new axis for the input feature
       y = np.array([d['signal'] for lst in self.HNrates for d in lst])
       
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
       y_train = pd.Series(y_train).map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}).to_numpy()
       y_test = pd.Series(y_test).map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}).to_numpy()
       # Create a label encoder for mapping the class labels
       label_encoder = LabelEncoder()
       unique_labels = np.unique(y_train)
       encoded_labels = label_encoder.fit_transform(unique_labels)
       # Compute the class weights
       class_weights = class_weight.compute_class_weight('balanced',
                                                         classes=unique_labels,
                                                         y=y_train)
       class_weight_dict = dict(zip(encoded_labels, class_weights))
       print(f"The class weights: {class_weight_dict}")
       tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
       model.summary()
       print("Training Model...")
       if(load_train):
            model.load_weights(filepath)
        
       model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                 validation_split=0.33, shuffle=True ,
                 callbacks=[tensorboard_callback, checkpoint_callback],
                 class_weight=class_weight_dict)
       # Load the best model weights
       model.load_weights(filepath)
       
       # Evaluate the model on the test set
       score = model.evaluate(X_test, y_test)
       print(score)
       self.nnmodel = model

    def load_model_fit(self):
        model = self.create_model(self.mz, self.nz)
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = checkpoint_dir + '/best_weights.hdf5'
        model.load_weights(filepath)
        self.nnmodel = model

    def predict_next(self, rate, smb):
        
        if isinstance(smb, str):
            smb_col = smb
        elif isinstance(smb, int):
            smb_col = rate.columns[smb]
        else:
            raise ValueError("smb should be either a string or an integer.")
        # Get the last Dyp rows of full_rates for the given symbol
        past_data = rate[['open','high','low','close']].tail(self.Dyp)
        psdt_HH = past_data.max(axis=0)['high']
        psdt_LL = past_data.min(axis=0)['low']
        past_data_normalized, mindf, maxdf = self.normalize(past_data, psdt_LL, psdt_HH)
        past_data_normalized_w, lng = self.decompose_and_flatten(past_data_normalized.values,'db1')
        # Reshape the past data for input to the neural network
        X = np.expand_dims(past_data_normalized_w, axis=(0, -1))
        # Use the trained neural network model to predict the future data
        y_pred = np.array(self.nnmodel.predict(X))
        y_pred = y_pred.squeeze()
        y_pred = [0 if x<1e-3 else round(100*x)/100 for x in y_pred] 
        return y_pred

    
    def predict_all(self, smb):
        # Loop through all dataframes in full_rates
        for df in self.full_rates:
            # Predict the next values for the given symbol using the predict_next method
            y_pred = self.predict_next(df, smb)
            self.Predicted_Rates.append(y_pred)
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

    def frontPlot(self, w, alpha=0.95, save=False):
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
            metrics = self.calculate_metrics(w, alpha)
    
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
