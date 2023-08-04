# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:59:37 2023

@author: ekosh
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Assuming df is your DataFrame and 'value' is the column with the time series data
df = pd.DataFrame({'value': np.random.rand(1444)})

N = len(df)
K = 32  # Set the size of your moving horizon here

# Initialize an empty DataFrame to store the results
P = pd.DataFrame(index=range(N), columns=range(N))

def calculate_ratio(i):
    result = {}
    for j in range(i+1, min(i+K+1, N)):
        result[j] = (df.loc[j, 'value'] - df.loc[i, 'value']) / df.loc[i, 'value']
    return result

if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        results = p.map(calculate_ratio, range(N))

    for i, result in enumerate(results):
        for j, value in result.items():
            P.loc[i, j] = value

print(P)







