import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import seaborn as sns
import os
from tqdm import tqdm

# Fetch OHLCV data for the top 500 largest companies
def fetch_data(tickers, start_date, end_date):
    """
    Fetch historical OHLCV data from Yahoo Finance for given tickers.
    """
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return pd.DataFrame(data)

# Save data to CSV
def save_data_to_csv(data, filename):
    """
    Save fetched data to a CSV file.
    """
    data.to_csv(filename)
    print(f"Data saved to {filename}")

# Read data from CSV
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

# Calculate daily returns
def calculate_daily_returns(data):
    """
    Calculate daily returns from the price data.
    """
    print(f"Calculating daily returns for {data.shape[1]} stocks...")
    return data.pct_change().dropna(how='all')

# Compute correlation matrix
def compute_correlation_matrix(daily_returns):
    """
    Compute the correlation matrix for the daily returns.
    """
    print(f"Computing correlation matrix for {daily_returns.shape[1]} stocks...")
    return daily_returns.corr()

# Filter out low correlation pairs
def get_high_correlation_pairs(correlation_matrix, threshold=0.75):
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

# Perform cointegration tests
def find_cointegrated_pairs(stock_prices, high_correlation_pairs=None, significance_level=0.05):
    """
    Perform cointegration tests on pairs of stocks that are highly correlated.
    """
    # Remove any columns with NaN values
    stock_prices = stock_prices.dropna(axis=1)

    n = stock_prices.shape[1]
    # pvalue_matrix = np.ones((n, n))
    pairs = []

    # if no high correlation pairs are provided, perform cointegration tests on all pairs
    if high_correlation_pairs is None:
        print("Performing cointegration tests on all pairs...")
        for i in range(n):
            print(f"Performing cointegration tests on {stock_prices.columns[i]} pairs...")
            for j in range(i+1, n):
                try:
                    stock1 = stock_prices.iloc[:, i]
                    stock2 = stock_prices.iloc[:, j]
                    score, pvalue, _ = coint(stock1, stock2)
                    if pvalue < significance_level:
                        pairs.append((stock_prices.columns[i], stock_prices.columns[j], pvalue))
                except:
                    print(f"Error in cointegration test for {stock_prices.columns[i]} and {stock_prices.columns[j]}")
        return pairs

    # Get high correlation pairs
    print(f"Performing cointegration tests on {len(high_correlation_pairs)} pairs...")

    # Perform cointegration tests on high correlation pairs
    for pair in tqdm(high_correlation_pairs, desc="Testing high correlation pairs"):
        try:
            stock1 = stock_prices[pair[0]]
            stock2 = stock_prices[pair[1]]
            score, pvalue, _ = coint(stock1, stock2)
            if pvalue < significance_level:
                pairs.append((pair[0], pair[1], pvalue))
        except:
            print(f"Error in cointegration test for {pair[0]} and {pair[1]}")

    return pairs

# Visualize Correlation Matrix
def plot_correlation_matrix(corr_matrix):
    """
    Plot the correlation matrix as a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Stocks')
    plt.show()

# Given a list of cointegrated pairs, calculate the spread between the two stocks
def calculate_spreads(cointegrated_pairs: list[tuple[str, str]], stock_prices: pd.DataFrame):
    """
    Calculate the spreads between the two stocks in each cointegrated pair.
    The spread = stock1_prices - beta * stock2_prices where beta is the coefficient
    of the linear regression of stock1_prices on stock2_prices
    """
    print(f"Calculating spreads for {len(cointegrated_pairs)} cointegrated pairs...")
    spreads = {}
    for pair in cointegrated_pairs:
        stock1 = stock_prices[pair[0]]
        stock2 = stock_prices[pair[1]]
        beta = np.polyfit(stock1, stock2, 1)[0]
        spread = stock1 - beta * stock2
        spreads[pair] = spread
    return spreads

# Calculate the mean and the standard deviation of the spread
def calculate_mean_and_std(spreads: dict[tuple[str, str], pd.Series]):
    """
    Calculate the mean and the standard deviation of the spread.
    """
    means = {}
    stds = {}
    for pair, spread in spreads.items():
        means[pair] = np.mean(spread)
        stds[pair] = np.std(spread)
    return means, stds

def load_data_to_csv():
    # Define the parameters
    # top_500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    top_1000_tickers = pd.read_html('https://en.wikipedia.org/wiki/Russell_1000_Index')[2]['Symbol'].tolist()
    start_date = '2023-09-08'
    end_date = '2024-09-08'

    csv_filename = './pairs_trading/2023_2024_stock_prices.csv'
    data = fetch_data(top_1000_tickers, start_date, end_date)
    # data = fetch_data(top_500_tickers, start_date, end_date)
    save_data_to_csv(data, csv_filename)

def pairs_trading_strategy(coint_pairs: list[tuple[str, str]], training_data: pd.DataFrame, testing_data: pd.DataFrame):
    """
    Implement the pairs trading strategy.
    """
    # Calculate the training data spreads
    training_spreads = calculate_spreads(coint_pairs, training_data)

    # Calculate the z-score of the testing data
    # first, calculate the spread of the testing data
    testing_spreads = calculate_spreads(coint_pairs, testing_data)
    # then, calculate the z-score of the testing spreads
    z_scores = {}
    for pair, testing_spread in testing_spreads.items():
        training_spread_mean = np.mean(training_spreads[pair])
        training_spread_std = np.std(training_spreads[pair])
        z_score = (testing_spread - training_spread_mean) / training_spread_std

        # only keep the pair, if the z-score gets within -0.1 to 0.1 at least 2 times
        if len(z_score) > 2 and np.any(np.abs(z_score) < 0.1):
            z_scores[pair] = z_score

    # for now, plot the z-scores
    for pair, z_score in z_scores.items():
        plt.figure(figsize=(12, 8))
        plt.plot(z_score, label=f"{pair[0]} - {pair[1]}")
        plt.title(f"{pair[0]} - {pair[1]} Z-Score")
        plt.legend()
        plt.show()

    # Implement the pairs trading strategy


def main():
    training_data_csv = './pairs_trading/2018_2023_stock_prices.csv'
    testing_data_csv = './pairs_trading/2023_2024_stock_prices.csv'

    training_data = read_data_from_csv(training_data_csv)
    testing_data = read_data_from_csv(testing_data_csv)

    # five_year_coint_pairs = [("IQV", "LH"), ("AWK", "RVTY"), ("LNT", "EXR"), ("CAT", "MA"), ("MS", "DGX")]
    # one_year_coint_pairs = [("FI", "L"), ("MMM", "EVRG"), ("T", "FICO"), ("NSC", "SYY"), ("ELV", "TT"), ("CMS", "ETR"), ("LHX", "STX"), ("EG", "KEYS"), ("ELV", "TSCO"), ("T", "MCO")]

    coint_pairs = [("LSTR", "PH"), ("BOKF", "BHF"), ("AXP", "LAMR"), ("IQV", "LH"), ("BHF", "WTFC"), ("EOG", "MRO"), ("AME", "MCHP"), ("BRO", "COST"), ("JBL", "LECO"), ("BHF", "RF")]

    # # Calculate daily returns
    # daily_returns = calculate_daily_returns(training_data)

    # # Compute correlation matrix
    # corr_matrix = compute_correlation_matrix(daily_returns)

    # # Filter out low correlation pairs
    # high_correlation_pairs = get_high_correlation_pairs(corr_matrix)

    # # Find cointegrated pairs
    # cointegrated_pairs = find_cointegrated_pairs(training_data, high_correlation_pairs)
    # # cointegrated_pairs = find_cointegrated_pairs(training_data)

    # # Sort the cointegrated pairs by p-value
    # cointegrated_pairs.sort(key=lambda x: x[2])
    # print("\nTop 10 Cointegrated Pairs:")
    # for pair in cointegrated_pairs[:10]:
    #     print(f"Pair: {pair[0]} and {pair[1]}, p-value: {pair[2]:.5f}")

    # # plot the top 3 cointegrated pairs on the same graph
    # plt.figure(figsize=(12, 8))
    # for pair in cointegrated_pairs[:3]:
    #     plt.plot(training_data[pair[0]] - training_data[pair[1]], label=f"{pair[0]} - {pair[1]}")
    # plt.title("Top 3 Cointegrated Pairs")
    # plt.legend()
    # plt.show()

    # Calculate the spreads for the 5 year cointegrated pairs
    training_spreads = calculate_spreads(coint_pairs, training_data)
    # for pair, spread in training_spreads.items():
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(spread, label=f"{pair[0]} - {pair[1]}")
    #     plt.plot(training_data[pair[0]], label=f"{pair[0]}")
    #     plt.plot(training_data[pair[1]], label=f"{pair[1]}")
    #     plt.title(f"{pair[0]} - {pair[1]} Spread")
    #     plt.legend()
    #     plt.show()

    # # Calculate the mean and the standard deviation of the spread and print them
    # mean, std = calculate_mean_and_std(training_spreads)
    # for pair in training_spreads.keys():
    #     print(f"{pair[0]} - {pair[1]}: Mean = {mean[pair]:.2f}, Std = {std[pair]:.2f}")
    #     print(f"Ratio of Std to Mean = {abs(std[pair] / mean[pair]):.2f}")

    pairs_trading_strategy(coint_pairs, training_data, testing_data)


main()
# load_data_to_csv()