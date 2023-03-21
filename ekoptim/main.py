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
import os
from tensorflow.keras.callbacks import ModelCheckpoint
#from sklearn.metrics import accuracy_score
import pywt


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
        self.HNrates = []
        self.Predicted_Rates=[]
        
        
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
        self.new_full_rates = []
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

    def decompose_and_flatten(self, data, wavelet):
        coeffs = pywt.wavedec(data, wavelet)
        lengths = [len(c) for c in coeffs]
        flattened_coeffs = np.concatenate(coeffs)
        return flattened_coeffs, lengths
    
    def reconstruct_from_flattenedx(self, flattened_coeffs, wavelet, lengths):
        coeffs = []
        start = 0
        for length in lengths:
            coeffs.append(flattened_coeffs[start:start + length])
            start += length
        reconstructed_data = pywt.waverec(coeffs, wavelet)
        return reconstructed_data

    def reconstruct_from_flattened(self, flattened_coeffs, wavelet, lengths):
        coeffs = []
        start = 0
        for length in lengths:
            coeffs.append(flattened_coeffs[start:start + length])
            start += length
        
        # Add print statements to inspect shapes
        print("Flattened coeffs shape:", flattened_coeffs.shape)
        print("Reconstructed coeffs shapes:", [c.shape for c in coeffs])
    
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
            past_data_im =  self.create_2d_image(past_data.values,'db1')
            future_data = df[smb].iloc[i:i+self.Dyf]
            flattened_coeffs, lengths = self.decompose_and_flatten(future_data.values, 'db1')
            new_row = {
                'past_data': past_data_im,
                'future_data': flattened_coeffs,
                'cLength': lengths
            }
            new_df = new_df.append(new_row, ignore_index=True)
        return new_df

    def r_squared(self, y_true, y_pred):
        # Calculate the residual sum of squares (numerator)
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        
        # Calculate the total sum of squares (denominator)
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        
        # Calculate R-squared
        r2 =  tf.divide(tf.exp(tf.subtract(1.0, tf.divide(residual, total))),tf.exp(1))
        
        return r2

    def normalize(self,data):
        #Normalize a pandas series by scaling its values to the range [0, 1].
        return (data - data.min()) / (data.max() - data.min()), data.min(), data.max()

    def apply_moving_horizon_norm(self,df,smb):
        new_df = []
        if isinstance(smb, str):
            smb_col = smb
        elif isinstance(smb, int):
            smb_col = df.columns[smb]
        else:
            raise ValueError("smb should be either a string or an integer.")
        for i in range(self.Dyp, len(df)-self.Dyf+1, self.Thi):
            past_data = df[smb_col].iloc[i-self.Dyp:i]
            past_data_normalized, mindf, maxdf = self.normalize(past_data)
            past_data_nm_im =  self.create_2d_image(past_data_normalized.values,'db1')
            future_data = df[smb_col].iloc[i:i+self.Dyf]
            future_data_rescaled = ((future_data - past_data.min()) /
                                    (past_data.max() - past_data.min()))
            flattened_coeffs, lengths = self.decompose_and_flatten(future_data_rescaled.values, 'db1')
            new_row = {
                'past_data': past_data_nm_im,
                'future_data': flattened_coeffs,
                'cLength': lengths,
                'minmax': [mindf,maxdf]
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

    def create_model(self, image_height, image_width):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(image_height, image_width, 1)),
    
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                           kernel_size=7, strides=1,
                           padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(filters=max(round(self.Dyp/4), 32),
                                   kernel_size=7, strides=1,
                                   padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Flatten(),
        
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
        
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
        
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
        
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
        
            tf.keras.layers.Dense(5 * self.Dyf, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
    
            tf.keras.layers.Dense(self.Dyf)
        ])
        return model

    def prepare_image(image):
        # Ensure the image has only one channel (grayscale)
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = np.mean(image, axis=2)
    
        # Add the channel dimension (1 channel) to the image
        image = np.expand_dims(image, axis=-1)

        return image
    
    def NNmake(self,symb='close', learning_rate=0.001, epochs=100, batch_size=32, load_train=False):
        print("Preparing Data...")
        self.HNrates = self.Hrz_Nrm(symb)
        mz = self.HNrates[0][0]['past_data'].shape[0]
        nz = self.HNrates[0][0]['past_data'].shape[1]
        # Define the neural network
        model = self.create_model(mz, nz) 

        # Compile the model with mean squared error loss
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mape', metrics=[self.r_squared])

        # Set up the callback to save the best model weights
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = checkpoint_dir + '/best_weights.hdf5'
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Train the model on the data in new_rates_lists
        X = np.array([d['past_data'] for lst in self.HNrates for d in lst])
        X = np.expand_dims(X, axis=-1)  # add a new axis for the input feature
        
        y = np.array([d['future_data'] for lst in self.HNrates for d in lst])
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        model.summary()
        print("Training Model...")
        if(load_train):
            model.load_weights(filepath)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.33, shuffle=False ,
                  callbacks=[tensorboard_callback, checkpoint_callback])
        # Load the best model weights
        model.load_weights(filepath)
        
        # Evaluate the model on the test set
        score = model.evaluate(X_test, y_test)
        print("The scores: ", score)
        self.nnmodel = model
    
    def predict_next(self, rate, smb):
        
        if isinstance(smb, str):
            smb_col = smb
        elif isinstance(smb, int):
            smb_col = rate.columns[smb]
        else:
            raise ValueError("smb should be either a string or an integer.")
        # Get the last Dyp rows of full_rates for the given symbol
        past_data = rate[smb_col].tail(self.Dyp)
    
        # Normalize the past data using the same min and max values used during training
        past_data_normalized, mindf, maxdf = self.normalize(past_data)
        past_data_nm_im =  self.create_2d_image(past_data_normalized.values,'db1')
        #print(past_data_nm_im.shape)
        # Reshape the past data for input to the neural network
        X = np.expand_dims(past_data_nm_im, axis=(0, -1))
    
        # Use the trained neural network model to predict the future data
        y_pred_w = self.nnmodel.predict(X)
        
        print(y_pred_w)
        print("y_pred_w shape:", y_pred_w.shape)  # Add this line
        y_pred_w_r = y_pred_w.squeeze()
        y_pred = self.reconstruct_from_flattened(y_pred_w_r, 'db1', self.HNrates[0][0]['cLength'])
        # Rescale the predicted future data to the original scale
        y_pred_rescaled = y_pred * (maxdf-mindf) + mindf
    
        return y_pred_rescaled

    
    def predict_all(self, smb):
        # Loop through all dataframes in full_rates
        for df in self.full_rates:
            # Predict the next values for the given symbol using the predict_next method
            y_pred = self.predict_next(df, smb)
            self.Predicted_Rates.append(y_pred)
            
    
            

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