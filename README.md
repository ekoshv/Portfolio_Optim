The ekoptim class is a portfolio optimization tool written in Python. It provides methods for portfolio optimization, data normalization, prediction, and visualization. This class is primarily used for financial portfolio optimization, which includes maximizing returns, minimizing risk, and optimizing the Sharpe ratio.

Here's a brief overview of the key methods in the class:

#Key Methods

load_model: This method loads a pre-trained neural network model from a specified file path. The model is used for prediction tasks.
load_model_fit: This method also loads a pre-trained model, but it creates the model structure first by calling the create_modelX method.
predict_next: Given a set of input data, this method uses the loaded model to predict the next state.
predict_all: This method loops through all dataframes in full_rates to predict the next values for all of them.
draw_states: This method visualizes the states of the provided dataframe using a candlestick chart.
Optim_...: A set of methods for portfolio optimization. These methods are differentiated by their objective functions and constraints, such as maximizing return, minimizing risk, or balancing the two based on the Sharpe ratio.
optiselect: This method selects which optimization method to use based on the provided argument.
calculate_metrics: This method calculates various portfolio metrics like risk, return, Sharpe ratio, Sortino ratio, CVAR, and maximum drawdown.
frontPlot: This method uses Monte Carlo simulation to generate the efficient frontier and plots it. The efficient frontier shows the set of optimal portfolios that offer the highest expected return for a defined level of risk.

#Usage

A sample usage of the ekoptim class can be seen in the provided sample_MT5_TradingView.py script. This script demonstrates how to use the ekoptim class to perform various tasks including connecting to MetaTrader 5, retrieving historical data, performing portfolio optimization, calculating portfolio metrics, and visualizing the results.

The script starts by instantiating the ekoptim class and collecting necessary input parameters from the user, such as the total equity, type of filter, group or path name, oil name, gold name, historical days, and whether the length of MetaTrader 5 data matches TradingView data.

Then it performs data retrieval and preprocessing using the TradingViewfeed and MetaTrader 5 API to retrieve historical data for the specified symbols. The data is then preprocessed, including filling missing values and calculating returns.

Next, the script performs portfolio optimization using the optiselect method of the ekoptim class. The user is given a choice to select an optimization type that suits their needs.

After obtaining the optimized weights, the script calculates various portfolio metrics using the calculate_metrics method and prints them out for the user to see.

The script also visualizes the efficient frontier using the frontPlot method and saves the data, including the optimizer object and historical data, to a pickle file using the dill library.

In the last part of the script, the script prepares data for deep learning, visualizes the states of the selected symbols using the draw_states method, creates a neural network model using the NNmake method, loads and fits the model using the load_model_fit method, and predicts the next values for all symbols using the predict_all method.

Please refer to the provided sample_MT5_TradingView.py script for a detailed understanding of how to use the ekoptim class in a real-world scenario.
