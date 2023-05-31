# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:21:51 2023

@author: ekosh
"""

import pandas as pd
import numpy as np

class MyClass:
    def calculate_signalX(self, df, X0):
        P = []
        for columnx0 in X0.columns:
            for columndf in df.columns:
                P.append((df[columndf] - X0[columnx0][0]) / X0[columnx0][0])
        vec = np.concatenate([series.to_numpy() for series in P])
        return vec

# Create an instance of MyClass
my_instance = MyClass()

# Create a DataFrame df with 100 rows and 4 columns
df = pd.DataFrame(np.random.rand(10000, 4), columns=['open', 'high', 'low', 'close'])
df +=[1,5,3,2]
# Create a DataFrame X0 with 1 row and 4 columns
X0 = pd.DataFrame(np.random.rand(1, 4), columns=['open', 'high', 'low', 'close'])
X0 +=[1,5,3,2]
# Call the calculate_signalX method
P = my_instance.calculate_signalX(df[['high','low']], X0[['high','low']])

print(P)
