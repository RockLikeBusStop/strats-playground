import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import seaborn as sns
import os

# Step 1: Fetch OHLCV data for the top 500 largest companies
def fetch_data(tickers, start_date, end_date):
    """
    Fetch historical OHLCV data from Yahoo Finance for given tickers.
    """
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return pd.DataFrame(data)

# Step 1a: Save data to CSV
def save_data_to_csv(data, filename):
    """
    Save fetched data to a CSV file.
    """
    data.to_csv(filename)
    print(f"Data saved to {filename}")

# Step 1b: Read data from CSV
def read_data_from_csv(filename):
    """
    Read data from a CSV file into a DataFrame.
    """
    if os.path.exists(filename):
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}")
        return data
    else:
        raise FileNotFoundError(f"{filename} does not exist. Please fetch data first.")

# Step 2: Calculate daily returns
def calculate_daily_returns(data):
    """
    Calculate daily returns from the price data.
    """
    print(f"Calculating daily returns for {data.shape[1]} stocks...")
    return data.pct_change().dropna(how='all')

# Step 3: Compute correlation matrix
def compute_correlation_matrix(daily_returns):
    """
    Compute the correlation matrix for the daily returns.
    """
    print(f"Computing correlation matrix for {daily_returns.shape[1]} stocks...")
    return daily_returns.corr()

# Step 3a: Filter out low correlation pairs
def get_high_correlation_pairs(correlation_matrix, threshold=0.7):
    """
    Return a list of pairs with correlation above the threshold.
    """
    pairs = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if correlation_matrix.iloc[i, j] > threshold:
                pairs.append((correlation_matrix.index[i], correlation_matrix.columns[j]))
    print(f"Found {len(pairs)} high correlation pairs")
    return pairs

# Step 4: Perform cointegration tests
def find_cointegrated_pairs(stock_prices, high_correlation_pairs, significance_level=0.05):
    """
    Perform cointegration tests on pairs of stocks that are highly correlated.
    """
    n = stock_prices.shape[1]
    # pvalue_matrix = np.ones((n, n))
    pairs = []

    # Remove any columns with NaN values
    stock_prices = stock_prices.dropna(axis=1)

    # Get high correlation pairs
    print(f"Performing cointegration tests on {len(high_correlation_pairs)} pairs")

    # Perform cointegration tests on high correlation pairs
    for pair in high_correlation_pairs:
        try:
            stock1 = stock_prices[pair[0]]
            stock2 = stock_prices[pair[1]]
            score, pvalue, _ = coint(stock1, stock2)
            if pvalue < significance_level:
                pairs.append((pair[0], pair[1], pvalue))
        except:
            print(f"Error in cointegration test for {pair[0]} and {pair[1]}")

    # for i in range(n):
    #     for j in range(i+1, n):
    #         try:
    #             stock1 = data.iloc[:, i]
    #             stock2 = data.iloc[:, j]
    #             score, pvalue, _ = coint(stock1, stock2)
    #             pvalue_matrix[i, j] = pvalue
    #             if pvalue < significance_level:
    #                 pairs.append((data.columns[i], data.columns[j], pvalue))
    #         except:
    #             print(f"Error in cointegration test for {data.columns[i]} and {data.columns[j]}")

    return pairs

# Step 5: Visualize Correlation Matrix
def plot_correlation_matrix(corr_matrix):
    """
    Plot the correlation matrix as a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Stocks')
    plt.show()

def load_data_to_csv():
    # Define the parameters
    top_500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    start_date = '2023-09-08'
    end_date = '2024-09-08'

    csv_filename = './pairs_trading/top_500_tickers.csv'
    data = fetch_data(top_500_tickers, start_date, end_date)
    save_data_to_csv(data, csv_filename)

def main():
    # Define the parameters
    top_500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    start_date = '2023-09-08'
    end_date = '2024-09-08'

    csv_filename = './pairs_trading/top_500_tickers.csv'

    # Fetch data
    if csv_filename != None:
        data = read_data_from_csv(csv_filename)
    else:
        data = fetch_data(top_500_tickers, start_date, end_date)

    # Calculate daily returns
    daily_returns = calculate_daily_returns(data)

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(daily_returns)

    # Filter out low correlation pairs
    high_correlation_pairs = get_high_correlation_pairs(corr_matrix)

    # Find cointegrated pairs
    cointegrated_pairs = find_cointegrated_pairs(data, high_correlation_pairs)

    # Sort the cointegrated pairs by p-value
    cointegrated_pairs.sort(key=lambda x: x[2])
    print("\nTop 10 Cointegrated Pairs:")
    for pair in cointegrated_pairs[:10]:
        print(f"Pair: {pair[0]} and {pair[1]}, p-value: {pair[2]:.5f}")

    # plot the difference between the two stocks for the most cointegrated pair
    plt.figure(figsize=(12, 8))
    plt.plot(data[cointegrated_pairs[0][0]] - data[cointegrated_pairs[0][1]], label=f"{cointegrated_pairs[0][0]} - {cointegrated_pairs[0][1]}")
    plt.title(f"{cointegrated_pairs[0][0]} - {cointegrated_pairs[0][1]}")
    plt.legend()
    plt.show()

main()