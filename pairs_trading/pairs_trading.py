import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import seaborn as sns
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

    def test_for_coint(pair):
        try:
            stock1 = stock_prices[pair[0]]
            stock2 = stock_prices[pair[1]]
            _, pvalue, _ = coint(stock1, stock2)
            if pvalue < significance_level:
                return (pair[0], pair[1], pvalue)
        except:
            print(f"Error in cointegration test for {pair[0]} and {pair[1]}")
        return None

    if high_correlation_pairs is None:
        n = stock_prices.shape[1]
        all_pairs = [(stock_prices.columns[i], stock_prices.columns[j])
                     for i in range(n) for j in range(i+1, n)]
        print(f"Performing cointegration tests on all {len(all_pairs)} pairs...")
    else:
        all_pairs = high_correlation_pairs
        print(f"Performing cointegration tests on {len(all_pairs)} high correlation pairs...")

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(test_for_coint, all_pairs), total=len(all_pairs), desc="Testing pairs"))

    pairs = [result for result in results if result is not None]
    return pairs

    # n = stock_prices.shape[1]
    # # pvalue_matrix = np.ones((n, n))
    # pairs = []

    # # if no high correlation pairs are provided, perform cointegration tests on all pairs
    # if high_correlation_pairs is None:
    #     print("Performing cointegration tests on all pairs...")
    #     for i in range(n):
    #         print(f"Performing cointegration tests on {stock_prices.columns[i]} pairs...")
    #         for j in range(i+1, n):
    #             try:
    #                 stock1 = stock_prices.iloc[:, i]
    #                 stock2 = stock_prices.iloc[:, j]
    #                 score, pvalue, _ = coint(stock1, stock2)
    #                 if pvalue < significance_level:
    #                     pairs.append((stock_prices.columns[i], stock_prices.columns[j], pvalue))
    #             except:
    #                 print(f"Error in cointegration test for {stock_prices.columns[i]} and {stock_prices.columns[j]}")
    #     return pairs

    # # Get high correlation pairs
    # print(f"Performing cointegration tests on {len(high_correlation_pairs)} pairs...")

    # # Perform cointegration tests on high correlation pairs
    # for pair in tqdm(high_correlation_pairs, desc="Testing high correlation pairs"):
    #     try:
    #         stock1 = stock_prices[pair[0]]
    #         stock2 = stock_prices[pair[1]]
    #         score, pvalue, _ = coint(stock1, stock2)
    #         if pvalue < significance_level:
    #             pairs.append((pair[0], pair[1], pvalue))
    #     except:
    #         print(f"Error in cointegration test for {pair[0]} and {pair[1]}")

    # return pairs

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
def calculate_spreads(cointegrated_pairs: list[tuple[str, str]], stock_prices: pd.DataFrame, betas: dict[tuple[str, str], float] | None = None):
    """
    Calculate the spreads between the two stocks in each cointegrated pair.
    The spread = stock1_prices - beta * stock2_prices where beta is the coefficient
    of the linear regression of stock1_prices on stock2_prices
    """
    print(f"Calculating spreads for {len(cointegrated_pairs)} cointegrated pairs...")
    spreads = {}
    betas_used = {}
    for pair in cointegrated_pairs:
        stock1 = stock_prices[pair[0]]
        stock2 = stock_prices[pair[1]]
        if betas is None:
            beta = np.polyfit(stock1, stock2, 1)[0]
        else:
            beta = betas[pair]
        spread = stock1 - beta * stock2
        spreads[pair] = spread
        betas_used[pair] = beta
    return spreads, betas_used

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

# Given the spreads, calculate if the spreads are stationary using the ADF test
def test_spread_stationarity(spreads: dict[tuple[str, str], pd.Series], significance_level: float = 0.05):
    """
    Test the stationarity of spreads using the Augmented Dickey-Fuller test.
    """
    results = {}
    for pair, spread in spreads.items():
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]
        is_stationary = p_value < significance_level
        results[pair] = (is_stationary, p_value)

    return results


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

def calculate_mean_reversion_time(spreads: dict[tuple[str, str], pd.Series]) -> dict[tuple[str, str], tuple[float, list[int]]]:
    """
    Calculate the mean reversion time for each pair of stocks.
    """
    result = {}

    for pair, spread in spreads.items():
        # Calculate mean and standard deviation
        mean = np.mean(spread)
        std = np.std(spread)

        # Calculate z-scores
        z_scores = (spread - mean) / std

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(z_scores)))[0]

        # Calculate durations between zero crossings
        durations = np.diff(zero_crossings)

        # Calculate average duration
        avg_duration = np.mean(durations) if len(durations) > 0 else np.nan

        result[pair] = (avg_duration, durations.tolist())

    return result

def pairs_trading_strategy(stationary_pairs: list[tuple[str, str]], log_stationary_pairs: list[tuple[str, str]], training_data: pd.DataFrame, testing_data: pd.DataFrame):
    """
    Implement the pairs trading strategy.
    """
    # Log transform the training and testing data
    log_training_data = np.log(training_data)
    log_testing_data = np.log(testing_data)

    # Calculate the training data spreads
    training_spreads, betas = calculate_spreads(stationary_pairs, training_data)
    log_training_spreads, log_betas = calculate_spreads(log_stationary_pairs, log_training_data)

    # # Calculate the mean reversion time for each pair
    # mean_reversion_periods = calculate_mean_reversion_time(training_spreads)
    # log_mean_reversion_periods = calculate_mean_reversion_time(log_training_spreads)
    # print(f"Mean reversion periods: {mean_reversion_periods}")
    # print(f"Log mean reversion periods: {log_mean_reversion_periods}")


    # Calculate the z-score of the testing data
    # first, calculate the spread of the testing data
    testing_spreads, _ = calculate_spreads(stationary_pairs, testing_data, betas)
    log_testing_spreads, _ = calculate_spreads(log_stationary_pairs, log_testing_data, log_betas)
    # then, calculate the z-score of the testing spreads
    z_scores = {}
    log_z_scores = {}
    for pair, testing_spread in testing_spreads.items():
        training_spread_mean = np.mean(training_spreads[pair])
        training_spread_std = np.std(training_spreads[pair])
        print(f"Pair: {pair}, Training spread mean: {training_spread_mean}, Training spread std: {training_spread_std}, Ratio of Std to Mean: {training_spread_std / training_spread_mean}")
        print(f"Pair: {pair}, Testing spread mean: {np.mean(testing_spread)}, Testing spread std: {np.std(testing_spread)}, Ratio of Std to Mean: {np.std(testing_spread) / np.mean(testing_spread)}")
        z_score = (testing_spread - training_spread_mean) / training_spread_std
        z_scores[pair] = z_score

    for pair, testing_spread in log_testing_spreads.items():
        training_spread_mean = np.mean(log_training_spreads[pair])
        training_spread_std = np.std(log_training_spreads[pair])
        print(f"Pair: {pair}, Training spread mean: {training_spread_mean}, Training spread std: {training_spread_std}, Ratio of Std to Mean: {training_spread_std / training_spread_mean}")
        print(f"Pair: {pair}, Testing spread mean: {np.mean(testing_spread)}, Testing spread std: {np.std(testing_spread)}, Ratio of Std to Mean: {np.std(testing_spread) / np.mean(testing_spread)}")
        log_z_score = (testing_spread - training_spread_mean) / training_spread_std
        log_z_scores[pair] = log_z_score

    # plot the z-scores
    plt.figure(figsize=(12, 8))
    for pair, z_score in z_scores.items():
        plt.plot(z_score, label=f"{pair[0]} - {pair[1]}")
    for pair, z_score in log_z_scores.items():
        plt.plot(z_score, label=f"LOG {pair[0]} - {pair[1]}")
    plt.title("Z-Scores of Testing Data")
    plt.legend()
    plt.show()

    # Implement the pairs trading strategy
    # loop through the z scores and testing data
    # make a list of portfolio values; for each pair, start with 1_000_000 cash and 0 shares
    # take a position when the abs(z_score) is greater than 3 * std/mean
    # position is cash / testing_data[pair[0]][i]
    # close the position when the abs(z_score) is less than 1 * std/mean
    # update the portfolio value daily
    # update the betas daily
    # print the portfolio value at the end of the loop


def main():
    training_data_csv = './pairs_trading/2018_2023_stock_prices.csv'
    testing_data_csv = './pairs_trading/2023_2024_stock_prices.csv'

    training_data = read_data_from_csv(training_data_csv)
    testing_data = read_data_from_csv(testing_data_csv)

    # naturallog the training and testing data
    log_training_data = np.log(training_data)
    log_testing_data = np.log(testing_data)

    # five_year_coint_pairs = [("IQV", "LH"), ("AWK", "RVTY"), ("LNT", "EXR"), ("CAT", "MA"), ("MS", "DGX")]
    # one_year_coint_pairs = [("FI", "L"), ("MMM", "EVRG"), ("T", "FICO"), ("NSC", "SYY"), ("ELV", "TT"), ("CMS", "ETR"), ("LHX", "STX"), ("EG", "KEYS"), ("ELV", "TSCO"), ("T", "MCO")]

    coint_pairs = [("LSTR", "PH"), ("BOKF", "BHF"), ("AXP", "LAMR"), ("IQV", "LH"), ("BHF", "WTFC"), ("EOG", "MRO"), ("AME", "MCHP"), ("BRO", "COST"), ("JBL", "LECO"), ("BHF", "RF")]

    log_coint_pairs = [("DHI", "ITW"), ("AME", "MCHP"), ("APA", "MRO"), ("BHF", "RF"), ("BHF", "PRU"), ("BRO", "COST"), ("ON", "PAG"), ("EQH", "SF"), ("ADP", "LPLA"), ("IEX", "TT")]

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
    training_spreads, _ = calculate_spreads(coint_pairs, training_data)
    log_training_spreads, _ = calculate_spreads(log_coint_pairs, log_training_data)
    # for pair, spread in training_spreads.items():
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(spread, label=f"{pair[0]} - {pair[1]}")
    #     plt.plot(training_data[pair[0]], label=f"{pair[0]}")
    #     plt.plot(training_data[pair[1]], label=f"{pair[1]}")
    #     plt.title(f"{pair[0]} - {pair[1]} Spread")
    #     plt.legend()
    #     plt.show()

    # Test stationarity of spreads
    stationarity_results = test_spread_stationarity(training_spreads)
    log_stationarity_results = test_spread_stationarity(log_training_spreads)
    # print("\nStationarity Test Results:")
    # for pair, (is_stationary, p_value) in stationarity_results.items():
    #     print(f"{pair[0]} - {pair[1]}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {p_value:.4f})")
    # print("\nLog Stationarity Test Results:")
    # for pair, (is_stationary, p_value) in log_stationarity_results.items():
    #     print(f"{pair[0]} - {pair[1]}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {p_value:.4f})")
    stationary_pairs = [pair for pair, (is_stationary, p_value) in stationarity_results.items() if is_stationary]
    log_stationary_pairs = [pair for pair, (is_stationary, p_value) in log_stationarity_results.items() if is_stationary]



    # # Calculate the mean and the standard deviation of the spread and print them
    # stationary_spreads = calculate_spreads(stationary_pairs, training_data)
    # log_stationary_spreads = calculate_spreads(log_stationary_pairs, log_training_data)
    # mean, std = calculate_mean_and_std(stationary_spreads)
    # log_mean, log_std = calculate_mean_and_std(log_stationary_spreads)
    # for pair in stationary_spreads.keys():
    #     print(f"{pair[0]} - {pair[1]}: Mean = {mean[pair]:.2f}, Std = {std[pair]:.2f}")
    #     print(f"Ratio of Std to Mean = {abs(std[pair] / mean[pair]):.2f}")
    # for pair in log_stationary_spreads.keys():
    #     print(f"{pair[0]} - {pair[1]}: Mean = {log_mean[pair]:.2f}, Std = {log_std[pair]:.2f}")
    #     print(f"Ratio of Std to Mean = {abs(log_std[pair] / log_mean[pair]):.2f}")

    pairs_trading_strategy(stationary_pairs, log_stationary_pairs, training_data, testing_data)

    # # plot the trainging and testing data for the stationary pairs
    # for pair in stationary_pairs:
    #     plt.figure(figsize=(12, 8))
    #     # plt.plot(training_data[pair[0]], label=f"{pair[0]}")
    #     # plt.plot(training_data[pair[1]], label=f"{pair[1]}")
    #     # plt.plot(testing_data[pair[0]], label=f"TESTING {pair[0]}")
    #     # plt.plot(testing_data[pair[1]], label=f"TESTING {pair[1]}")
    #     plt.plot(training_data[pair[0]] - training_data[pair[1]], label=f"{pair[0]} - {pair[1]}")
    #     plt.plot(testing_data[pair[0]] - testing_data[pair[1]], label=f"TESTING {pair[0]} - {pair[1]}")
    #     plt.title(f"{pair[0]} - {pair[1]} Spread")
    #     plt.legend()
    #     plt.show()


main()
# load_data_to_csv()