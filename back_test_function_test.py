import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import tensorflow as tf

# Assuming daily_returns is a Pandas Series with daily return percentage
# Assuming signals is a list of strategy signals

def backtest_strategy_tf(daily_returns, signals, swap_percentage, commission_percentage):
    position = tf.Variable(0, dtype=tf.int32)  # 0: no position, 1: long, -1: short
    capital = tf.Variable(1.0, dtype=tf.float32)  # Initial capital
    capital_history = []

    for i in range(len(daily_returns)):
        signal = signals[i]
        daily_return = daily_returns.iloc[i]

        if signal == 0:  # Long
            if position != 1:
                position.assign(1)
                capital.assign(capital * (1 - commission_percentage))
        elif signal == 1:  # Short
            if position != -1:
                position.assign(-1)
                capital.assign(capital * (1 - commission_percentage))
        elif signal == 2:  # Hold
            pass
        elif signal == 3:  # Close buys
            if position == 1:
                position.assign(0)
                capital.assign(capital * (1 - commission_percentage))
        elif signal == 4:  # Close sells
            if position == -1:
                position.assign(0)
                capital.assign(capital * (1 - commission_percentage))

        # Update capital based on position, daily return, and swap
        if position == 1:
            capital.assign(capital * (1 + daily_return - swap_percentage))
        elif position == -1:
            capital.assign(capital * (1 - daily_return - swap_percentage))

        capital_history.append(capital.numpy())

    return pd.Series(capital_history, index=daily_returns.index)

# Generate random daily returns
num_days = 15*252  # For example, 252 trading days in a year
start_date = datetime(2022, 1, 1)
date_range = pd.date_range(start_date, periods=num_days)
random_daily_returns = pd.Series([random.uniform(-0.03, 0.03) for _ in range(num_days)], index=date_range)

# Generate random signals
random_signals = [random.choice([0, 1, 2, 3, 4]) for _ in range(num_days)]

# Set swap_percentage and commission_percentage
swap_percentage = 0.0001  # For example, 0.01% swap
commission_percentage = 0.0005  # For example, 0.05% commission

# Run backtest
capital_history = backtest_strategy_tf(random_daily_returns, random_signals, swap_percentage, commission_percentage)

# Display results
print("Final capital:", capital_history.iloc[-1])
capital_history.plot(title="Backtest Results", ylabel="Capital", xlabel="Time")
