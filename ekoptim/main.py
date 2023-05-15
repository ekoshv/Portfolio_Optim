import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import math
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.mode.use_inf_as_na = True
import datetime
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import traceback
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pywt
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import talib
#from numba import jit

class ekoptim():
    def __init__(self, returns, risk_free_rate,
                 target_SR, target_Return, target_Volat,
                 max_weight, toler,
                 full_rates):
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

    def matthews_correlation(self, y_true, y_pred):
        y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
    
        y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
        y_neg = 1 - y_pos
    
        tp = tf.reduce_sum(y_pos * y_pred_pos)
        tn = tf.reduce_sum(y_neg * y_pred_neg)
    
        fp = tf.reduce_sum(y_neg * y_pred_pos)
        fn = tf.reduce_sum(y_pos * y_pred_neg)
    
        numerator = (tp * tn) - (fp * fn)
        denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
        return numerator / (denominator + tf.keras.backend.epsilon())

    def multiclass_mcc(self, y_true, y_pred, num_classes=9):
        # obtain predictions here, we can add in a threshold if we would like to
        y_pred = tf.argmax(y_pred, axis=-1)
    
        # cast to int64
        y_true = tf.squeeze(tf.cast(y_true, tf.int64), axis=-1)
        y_pred = tf.cast(y_pred, tf.int64)
    
        # total number of samples
        s = tf.size(y_true, out_type=tf.int64)
    
        # total number of correctly predicted labels
        c = s - tf.math.count_nonzero(y_true - y_pred)
    
        # number of times each class truely occured
        t = []
    
        # number of times each class was predicted
        p = []
    
        for k in range(num_classes):
            k = tf.cast(k, tf.int64)
        
            # number of times that the class truely occured
            t.append(tf.reduce_sum(tf.cast(tf.equal(k, y_true), tf.int32)))
    
            # number of times that the class was predicted
            p.append(tf.reduce_sum(tf.cast(tf.equal(k, y_pred), tf.int32)))
    
    
        t = tf.expand_dims(tf.stack(t), 0)
        p = tf.expand_dims(tf.stack(p), 0)
    
        s = tf.cast(s, tf.int32)
        c = tf.cast(c, tf.int32)
    
        num = tf.cast(c*s - tf.matmul(t, tf.transpose(p)), tf.float32)
        dem = tf.math.sqrt(tf.cast(s**2 - tf.matmul(p, tf.transpose(p)), tf.float32)) \
              * tf.math.sqrt(tf.cast(s**2 - tf.matmul(t, tf.transpose(t)), tf.float32))
    
    
        mcc = tf.divide(num, dem + 1e-6)
    
        return mcc

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
        if np.ndim(data) <= 1:
            flattened_coeffs = []
            coeffs = pywt.wavedec(data, wavelet)
            lengths = [len(c) for c in coeffs]
            flattened_coeffs = np.concatenate(coeffs)
            return flattened_coeffs, lengths
        else:
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

    def normalize(self,data, dmn, dmx, xrnd=0):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()        
        return (((data - dmn) / (dmx - dmn))+
                np.random.uniform(-xrnd, xrnd, data.shape), data.min(), data.max())
    
    def calculate_signal(self, fd):
        # Thresholding
        mx_val = min(fd.max(),10)
        mn_val = max(fd.min(),-10)
        sigs=[]
        # Conditioning
        # for mx_val
        if mx_val>=2.0:
            sigs.append(2)
        elif 1.1<=mx_val<2.0:
            sigs.append(1)
        elif mx_val<1.1:
            sigs.append(0)
        else:
            sigs.append(0)
        # for mn_val
        if mn_val<=-2.0:
            sigs.append(-2)
        elif -2<mn_val<=-1.1:
            sigs.append(-1)
        elif -1.1<mn_val:
            sigs.append(0)
        else:
            sigs.append(0)
        
        state_map = {(2, -2): 0, (2, -1): 1, (2, 0): 2, (1, -2): 3, (1, -1): 4, (1, 0): 5, 
                     (0, -2): 6, (0, -1): 7, (0, 0): 8}
        state = state_map[tuple(sigs)]
        
        return state, sigs
    
    def more_data(self, dfp):
        dfp['dayofweek'] = dfp.index.dayofweek/7
        dfp['dayofmonth'] = dfp.index.day/31
        dfp['monthofyear'] = dfp.index.month/12
        
        # Calculate the difference for each column
        msma_gsma = dfp['MSMA'] - dfp['GSMA']
        ssma_gsma = dfp['SSMA'] - dfp['GSMA']
        dfp['MSMA_GSMA'] = msma_gsma / dfp['GSMA']
        dfp['SSMA_GSMA'] = ssma_gsma / dfp['GSMA']
        
        for ma in ['GSMA', 'MSMA', 'SSMA']:
            for price in ['open', 'high', 'low', 'close']:
                diff = dfp[price] - dfp[ma]
                dfp[f'{price}_{ma}'] = diff / dfp[ma]
        
        return dfp
    
    def apply_moving_horizon_norm(self,dfs,smb,spn, tile_size, xrnd=0):
        df = dfs[0]
        gld = dfs[1]
        oil = dfs[2]
        new_df = []
        if isinstance(smb, str):
            smb_col = smb
        elif isinstance(smb, int):
            smb_col = df.columns[smb]
        else:
            raise ValueError("smb should be either a string or an integer.")

        for i in range(self.Dyp+self.SMAP[0]+1, len(df)-self.Dyf+1, self.Thi):
            past_data = df[['open','high','low','close','GSMA','MSMA','SSMA']].iloc[i-self.Dyp:i]
            # Extract data from df2 using index of df1 and fill missing rows with NaN
            past_gld = gld.reindex(past_data.index)
            past_oil = oil.reindex(past_data.index)
            psdt_HH = past_data[['open','high','low','close']].max(axis=0)['high']
            psdt_LL = past_data[['open','high','low','close']].min(axis=0)['low']
            past_data_normalized, mindf, maxdf = self.normalize(past_data[['open','high','low','close']], psdt_LL, psdt_HH,xrnd)
            past_data_normalized_w, lng = self.decompose_and_flatten(past_data_normalized,'db1')
            pst_dt_tiled = np.tile(past_data_normalized, tile_size)
            pst_dt_tiled += np.random.uniform(-xrnd/5, xrnd/5, pst_dt_tiled.shape)
            
            past_data = self.more_data(past_data)
            past_gld = self.more_data(past_gld)
            past_oil = self.more_data(past_oil)
            
            future_data = df[smb_col].iloc[i:i+self.Dyf]
            future_data_rescaled, fdmn, fdmx = self.normalize(future_data, psdt_LL, psdt_HH, xrnd)
            state, signal = self.calculate_signal(future_data_rescaled)
            df.at[df.index[i-1], 'state'] = state
            new_row = {
                'name': df.name,
                'past_data': [pst_dt_tiled,
                              past_data.loc[:, 'dayofweek':].fillna(0),
                              past_gld.loc[:, 'dayofweek':].fillna(0),
                              past_oil.loc[:, 'dayofweek':].fillna(0)],
                'future_data': future_data_rescaled,
                'state': state,
                'signal': signal,
                'pstraw': past_data,
                'minmax': [psdt_LL,psdt_HH],
                'dati': df.index[i-1]
            }
            #print(new_row)
            new_df.append(new_row)
        return new_df        

    def Hrz_Nrm(self, rates, smb, spn, tile_size, xrnd=0):
        # Apply the moving horizon to each dataframe in rates_lists
        return [self.apply_moving_horizon_norm([df,rates[-2],rates[-1]], smb, spn, tile_size,xrnd) for 
                df in tqdm(rates, desc='Processing DataFrames')]

    def Prepare_Data(self, symb, spn=1, tile_size=(2,2), xrnd=0,
                     Selected_symbols=None,
                     Dyp=8, Dyf=32, Thi=3,
                     SMAP=[144,45,12]):
        print("Preparing Data...")
        self.spn = spn
        self.Dyp = Dyp   # Number of past days to consider in the moving horizon
        self.Dyf = Dyf   # Number of future days to predict in the moving horizon
        self.Thi = Thi   # Time horizon interval (in days)
        self.tile_size = tile_size
        self.SMAP = SMAP
        srates = []
        if Selected_symbols is None:
            srates = self.full_rates
        else:
            self.selected_rates = [df for df in self.full_rates 
                                   if df.columns[-1] in Selected_symbols]
            self.selected_rates.append(self.full_rates[-2])
            self.selected_rates.append(self.full_rates[-1])
            srates = self.selected_rates
        for df in srates:
            # Find the index of the 'close' column
            snam = df.columns[-1]
            close_idx = df.columns.get_loc('close')
            df['state']=-1
            df.insert(close_idx + 1, 'state', df.pop('state'))
            # Calculate the SMA with a time window of SMAP days
            #---G
            gsma = talib.SMA(df['close'], timeperiod=SMAP[0])
            # Add the SMA values to the DataFrame
            df['GSMA'] = gsma
            df.insert(close_idx + 2, 'GSMA', df.pop('GSMA'))
            #---M
            msma = talib.SMA(df['close'], timeperiod=SMAP[1])
            # Add the SMA values to the DataFrame
            df['MSMA'] = msma
            df.insert(close_idx + 3, 'MSMA', df.pop('MSMA'))
            #---S
            ssma = talib.SMA(df['close'], timeperiod=SMAP[2])
            # Add the SMA values to the DataFrame
            df['SSMA'] = ssma
            df.insert(close_idx + 4, 'SSMA', df.pop('SSMA'))
            df.name = snam
        self.HNrates = self.Hrz_Nrm(srates, symb, spn, tile_size, xrnd)            
        # self.mz = self.HNrates[0][0]['past_data'][0].shape[0]
        # self.nz = self.HNrates[0][0]['past_data'][0].shape[1]

    def create_model(self, image_height, image_width):
        input_layer = tf.keras.layers.Input(shape=(image_height, image_width, 1))
    
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1,
                                    padding="same", activation="relu")(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1,
                                    padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                    padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_layer = tf.keras.layers.Dropout(0.5)(x)
    
        return input_layer, output_layer

    def create_modelX(self):
        input0, output0 = self.create_model(self.mz0, self.nz0)
        input1, output1 = self.create_model(self.mz1, self.nz1)
        input2, output2 = self.create_model(self.mz2, self.nz2)
        input3, output3 = self.create_model(self.mz3, self.nz3)
        combined_output = Concatenate()([output0, output1, output2, output3])

        x = tf.keras.layers.Dense(1024, activation="relu")(combined_output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(5*self.Dyf, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        final_output = tf.keras.layers.Dense(9, activation='softmax')(x)
        model = Model(inputs=[input0, input1, input2, input3], outputs=final_output)
        return model

    def custom_loss(self, y_true, y_pred, num_classes=9, average='macro', name="custom_loss"):
        # Calculate the CategoricalCrossentropy loss
        self.n_classes = num_classes
        sce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = sce_loss(y_true, y_pred)
        
        # Calculate the accuracy
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        
        # Calculate the inverse of the accuracy
        inv_accuracy = 1.0 - tf.reduce_mean(accuracy)

        # Calculate precision and recall
        true_positives = tf.reduce_sum(y_true * y_pred, axis=0)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        
        # Calculate the F1-score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        mean_f1_score = tf.reduce_mean(f1_score)
        
        # Calculate the inverse of the mean F1-score
        inv_f1_score = 1.0 / (mean_f1_score+0.001)
        
        # Calculate the inverse of the mean F1-score
        inv_f1_score = 1.0 / (mean_f1_score+0.001)
        
        # Calculate Matthews Correlation Coefficient (MCC)
        rmcc = 2.0 - self.multiclass_mcc(y_true, y_pred)
        
        # Combine the loss and inverse of the accuracy and F1
        combined_loss = (loss + 2*inv_accuracy + 4*inv_f1_score + 8*rmcc)/15
        combined_loss.__name__ = name
        
        return combined_loss
    
    def NNmake(self, model=None,
               learning_rate=0.001, epochs=100, batch_size=32, k_n=None,
               load_train=False):
        self.k_n=5
        if k_n is not None:
            self.k_n = k_n
        # Define the neural network
        self.mz0 = self.HNrates[0][0]['past_data'][0].shape[0]
        self.nz0 = self.HNrates[0][0]['past_data'][0].shape[1]

        self.mz1 = self.HNrates[0][0]['past_data'][1].shape[0]
        self.nz1 = self.HNrates[0][0]['past_data'][1].shape[1]

        self.mz2 = self.HNrates[0][0]['past_data'][2].shape[0]
        self.nz2 = self.HNrates[0][0]['past_data'][2].shape[1]

        self.mz3 = self.HNrates[0][0]['past_data'][3].shape[0]
        self.nz3 = self.HNrates[0][0]['past_data'][3].shape[1]        

        model = self.create_modelX()
        
        # Compile the model with mean squared error loss
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,
               loss=self.custom_loss,
               metrics=['accuracy',tfa.metrics.F1Score(num_classes=9, average='macro'),
                        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        self.multiclass_mcc])

        #model.compile(optimizer=opt, loss='mape')
   
        # Set up the callback to save the best model weights
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = checkpoint_dir + '/best_weights.hdf5'
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                              verbose=1, save_best_only=True, mode='min')
   
        # Train the model on the data in new_rates_lists
        X0 = np.array([d['past_data'][0] for lst in self.HNrates for d in lst])
        X1 = np.array([d['past_data'][1] for lst in self.HNrates for d in lst])
        X2 = np.array([d['past_data'][2] for lst in self.HNrates for d in lst])
        X3 = np.array([d['past_data'][3] for lst in self.HNrates for d in lst])
        
        X0 = np.expand_dims(X0, axis=-1)
        X1 = np.expand_dims(X1, axis=-1)
        X2 = np.expand_dims(X2, axis=-1)
        X3 = np.expand_dims(X3, axis=-1)
        
        y = np.array([d['state'] for lst in self.HNrates for d in lst])

        train_indices, test_indices = next(iter(ShuffleSplit(n_splits=1, test_size=0.33).split(X0)))
        X_train0, X_test0 = X0[train_indices], X0[test_indices]
        X_train1, X_test1 = X1[train_indices], X1[test_indices]
        X_train2, X_test2 = X2[train_indices], X2[test_indices]
        X_train3, X_test3 = X3[train_indices], X3[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
 
        self.k_n=5
        if k_n is not None:
            self.k_n = k_n 
        
            # Initialize the SMOTE object
            smote = SMOTE(sampling_strategy='auto', k_neighbors=self.k_n, random_state=42)
            
            # Fit and resample the training data for each dataset
            X_train0_resampled, y_train_resampled = smote.fit_resample(X_train0.reshape(X_train0.shape[0], -1), y_train)
            X_train1_resampled, _ = smote.fit_resample(X_train1.reshape(X_train1.shape[0], -1), y_train)
            X_train2_resampled, _ = smote.fit_resample(X_train2.reshape(X_train2.shape[0], -1), y_train)
            X_train3_resampled, _ = smote.fit_resample(X_train3.reshape(X_train3.shape[0], -1), y_train)
            
            # Reshape the resampled data back to its original shape
            y_train = y_train_resampled.reshape((-1,) + y_train.shape[1:])
            X_train0 = X_train0_resampled.reshape((-1,) + X_train0.shape[1:])
            X_train1 = X_train1_resampled.reshape((-1,) + X_train1.shape[1:])
            X_train2 = X_train2_resampled.reshape((-1,) + X_train2.shape[1:])
            X_train3 = X_train3_resampled.reshape((-1,) + X_train3.shape[1:])
        
        # Create a label encoder for mapping the class labels
        label_encoder = LabelEncoder()
        unique_labels = np.unique(y_train)
        encoded_labels = label_encoder.fit_transform(unique_labels)
        
        # Fit the label encoder on the training labels and transform both train and test labels
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Get the number of unique classes
        num_classes = len(unique_labels)
        
        # Convert the encoded class labels to one-hot encoding
        y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
        
        # Compute the class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y_train)
        class_weight_dict = dict(zip(encoded_labels, class_weights))
        print(f"The class weights: {class_weight_dict}")
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        model.summary()
        print("Training Model...")
        if(load_train):
            model.load_weights(filepath)
        
        model.fit([X_train0, X_train1, X_train2, X_train3], y_train_one_hot, epochs=epochs, batch_size=batch_size,
                  validation_split=0.33, shuffle=True,
                  callbacks=[tensorboard_callback, checkpoint_callback],
                  class_weight=class_weight_dict)
        
        # Load the best model weights
        model.load_weights(filepath)
        
        # Evaluate the model on the test set
        score = model.evaluate([X_test0, X_test1, X_test2, X_test3], y_test_one_hot)
        print(score)
        self.nnmodel = model

    def load_model_fit(self):
        model = self.create_modelX()
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = checkpoint_dir + '/best_weights.hdf5'
        model.load_weights(filepath)
        self.nnmodel = model

    def predict_next(self, idf, smb):
        rate = idf[0]
        gld = idf[1]
        oil = idf[2]
        try:
            if isinstance(smb, str):
                smb_col = smb
            elif isinstance(smb, int):
                smb_col = rate.columns[smb]
            else:
                raise ValueError("smb should be either a string or an integer.")
            # Get the last Dyp rows of full_rates for the given symbol
            past_data = rate[['open','high','low','close','GSMA','MSMA','SSMA']].tail(self.Dyp)
            past_gld = gld.reindex(past_data.index)
            past_oil = oil.reindex(past_data.index)
            
            psdt_HH = past_data.max(axis=0)['high']
            psdt_LL = past_data.min(axis=0)['low']
            past_data_normalized, mindf, maxdf = self.normalize(past_data[['open','high','low','close']], psdt_LL, psdt_HH)
            past_data_normalized_w, lng = self.decompose_and_flatten(past_data_normalized,'db1')
            pst_dt_tiled = np.tile(past_data_normalized, self.tile_size)
            # Reshape the past data for input to the neural network       
            self.X0 = pst_dt_tiled
            self.X1 = (rate.tail(self.Dyp)).loc[:, 'dayofweek':].fillna(0)
            self.X2 = past_gld.loc[:, 'dayofweek':].fillna(0)
            self.X3 = past_oil.loc[:, 'dayofweek':].fillna(0)

            X0 = np.expand_dims(self.X0, axis=(0, -1))
            X1 = np.expand_dims(self.X1, axis=(0, -1))
            X2 = np.expand_dims(self.X2, axis=(0, -1))
            X3 = np.expand_dims(self.X3, axis=(0, -1))
            
            # Use the trained neural network model to predict the future data
            y_pred = np.array(self.nnmodel.predict([X0, X1, X2, X3]))
            y_pred = y_pred.squeeze()
            y_pred = [0 if x<1e-3 else round(100*x)/100 for x in y_pred] 
            return {i: value for i, value in enumerate(y_pred)}
        
        except Exception as e:
            print(f"An error occurred in Prediction Process: {e}")
            traceback.print_exc()
            return None
    
    def predict_all(self, smb):
        # Loop through all dataframes in full_rates
        self.Predicted_Rates = []
        for df in self.selected_rates:
            df = self.more_data(df)
            gol = self.more_data(self.full_rates[-2])
            oil = self.more_data(self.full_rates[-1])
            # Predict the next values for the given symbol using the predict_next method
            idf = [df, gol, oil]
            y_pred = self.predict_next(idf, smb)
            self.Predicted_Rates.append(y_pred)
            #print(self.Predicted_Rates)
    
    def draw_states(self, df):
        try:
            pio.renderers.default = "browser"
            # Filter the DataFrame to get the dates where state == 2 or state == 6
            blue_flash_dates = df[df['state'] == 2].index
            red_flash_dates = df[df['state'] == 6].index
            st_na = df.columns[-1]
            
            # Create a candlestick chart
            fig = go.Figure(go.Candlestick(x=df.index,
                                           open=df['open'],
                                           high=df['high'],
                                           low=df['low'],
                                           close=df['close']))
            
            # Add moving averages
            fig.add_trace(go.Scatter(x=df.index, y=df['GSMA'],
                                     mode='lines', name='GSMA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MSMA'],
                                     mode='lines', name='MSMA', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df.index, y=df['SSMA'],
                                     mode='lines', name='SSMA', line=dict(color='red')))
            
            # Add blue flash up markers
            fig.add_trace(go.Scatter(x=blue_flash_dates,
                                     y=df[df['state'] == 2]['high'],
                                     mode='markers',
                                     marker=dict(color='blue', size=15, symbol='triangle-up'),
                                     name='Blue Flash Up'))
            
            # Add purple flash down markers
            fig.add_trace(go.Scatter(x=red_flash_dates,
                                     y=df[df['state'] == 6]['low'],
                                     mode='markers',
                                     marker=dict(color='purple', size=15, symbol='triangle-down'),
                                     name='Purple Flash Down'))
            
            # Customize chart layout
            fig.update_layout(title=f'{st_na} Stock Price with Blue and Purple Flashes',
                              xaxis_title='Date',
                              yaxis_title='Price',
                              xaxis_rangeslider_visible=False)
            
            # Display the chart
            fig.show()
        except Exception as e:
            print(f"Caught an exception during Drawing: {e}")
            traceback.print_exc()

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
