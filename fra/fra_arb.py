import ccxt
import csv
import pandas as pd
from datetime import datetime, timedelta

def import_funding_rate_data():
    # Function to convert the date string to a timestamp in milliseconds
    def convert_to_timestamp1(date_str):
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return int(dt_obj.timestamp() * 1000)
    def convert_to_timestamp2(date_str):
        dt_obj = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
        return int(dt_obj.timestamp() * 1000)

    # Function to convert the funding rate to a float
    def convert_to_float(funding_rate_str):
        return float(funding_rate_str.strip('%')) / 100

    funding_data1 = []
    funding_data2 = []

    # Read the CSV file
    with open('./fra/Binance Funding Rate History_SOLUSDT Perpetual_2024-08-26.csv', mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_obj = {
                "timestamp": convert_to_timestamp1(row['ï»¿"Time"']),
                "fundingRate": convert_to_float(row['Funding Rate'])
            }
            funding_data1.append(data_obj)

    with open('./fra/Kucoin Funding History SOLUSDTM Perpetual.csv', mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_obj = {
                "timestamp": convert_to_timestamp2(row['ï»¿"Time"']),
                "fundingRate": convert_to_float(row['Funding Rate'])
            }
            funding_data2.append(data_obj)

    return funding_data1, funding_data2

def zip_rates_and_prices(funding_rates, ohlcvs):
    # Function to find the correct funding rate
    def get_funding_rate(timestamp, funding_data):
        eight_hours_in_ms = 8 * 60 * 60 * 1000
        four_hours_in_ms = 4 * 60 * 60 * 1000
        for data in funding_data:
            if data["timestamp"] == timestamp or data["timestamp"] + four_hours_in_ms == timestamp:
                return data["fundingRate"]
        return None

    # Create DataFrame
    data = {
        "timestamp": [item[0] for item in ohlcvs],
        "open": [item[1] for item in ohlcvs],
        "close": [item[4] for item in ohlcvs],
        "funding_rate": [get_funding_rate(item[0], funding_rates) for item in ohlcvs]
    }

    df = pd.DataFrame(data)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

    return df

def try_fra_arb(days_in_past=66.7):
    # Fetch funding rates and OHLCVs for SOL/USDT from Binance and KuCoin
    exchange1 = ccxt.binanceusdm()
    exchange2 = ccxt.kucoinfutures()

    end_day = datetime(2024, 8, 26)
    secs_since = end_day - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000
    data_count = int(3 * days_in_past)

    # # Rate capped to 100 or 33 days
    # funding_rate1 = exchange1.fetch_funding_rate_history(symbol="SOL/USDT", since=ms_since)
    # funding_rate2 = exchange2.fetch_funding_rate_history(symbol="SOLUSDTM", since=ms_since)

    funding_rate1, funding_rate2 = import_funding_rate_data()

    sol_usdt_prices1 = exchange1.fetch_ohlcv("SOL/USDT", timeframe="8h", since=ms_since, limit=data_count)
    sol_usdt_prices2 = exchange2.fetch_ohlcv("SOLUSDTM", timeframe="8h", since=ms_since, limit=data_count)

    # Zip funding rates and prices into one DataFrame with the same timestamps
    exchange1_data = zip_rates_and_prices(funding_rate1, sol_usdt_prices1)
    exchange2_data = zip_rates_and_prices(funding_rate2, sol_usdt_prices2)

    # Strategy simulation:
    # loop through each row
    # check if a position is open
    # if not open, take a 10x long or short position with the exchange with the higher funding rate
    # if a position is open, update the data, check if margin is sufficient & if the funding rates flipped
    # if margin is insufficient, reallocate margin across exchanges
    # if funding rates flipped, close the position and open a new one with the exchange with the higher funding rate; track pnl
    # consider adding a 0.5% fee for each trade
    # calculate and store the returns of the FRA

    # Simulate the FRA strategy
    starting_cash = 50_000
    cash_usdt1 = 50_000
    cash_usdt2 = 50_000
    leverage = 5
    exchange1_position = {}
    exchange2_position = {}
    funding_rate_returns = []
    funding_rate_returns_combined = []
    trading_fee = 0.05 / 100

    for i in range(len(exchange1_data)):
        # Check if a position is open
        if exchange1_position and exchange2_position:
            if exchange1_position["type"] == "long":
                # Track FRA returns
                fund1_return = (-1) * exchange1_position["funding_rate"] * exchange1_position["size"] * exchange1_data["close"][i]
                fund2_return = exchange2_position["funding_rate"] * exchange2_position["size"] * exchange2_data["close"][i]
                funding_rate_returns.append(fund1_return)
                funding_rate_returns.append(fund2_return)
                funding_rate_returns_combined.append(fund1_return + fund2_return)

                # update the position values
                exchange1_position["margin_return"] += (exchange1_data["close"][i] - exchange1_data["close"][i-1]) * exchange1_position["size"] + fund1_return
                exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])
                exchange1_position["funding_rate"] = exchange1_data["funding_rate"][i]

                exchange2_position["margin_return"] += (-1) * (exchange2_data["close"][i] - exchange2_data["close"][i-1]) * exchange2_position["size"] + fund2_return
                exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])
                exchange2_position["funding_rate"] = exchange2_data["funding_rate"][i]

                # Check if the funding rates flipped or simulation has ended and close out
                rate_diff = exchange1_data["funding_rate"][i] - exchange2_data["funding_rate"][i]
                if rate_diff >= 0 or i == len(exchange1_data) - 1:
                    # Close the positions
                    # Returns include margin + leveraged profit
                    is_profitable1 = (exchange1_data["close"][i] - exchange1_position["entry_price"]) > 0
                    cash_usdt1 += (exchange1_position["size"] * exchange1_data["close"][i] * exchange1_position["curr_margin_pct"])
                    cash_usdt1 += (exchange1_position["size"] * (exchange1_data["close"][i] - exchange1_position["entry_price"])) if is_profitable1 else 0

                    is_profitable2 = (-1)*(exchange2_data["close"][i] - exchange2_position["entry_price"]) > 0
                    cash_usdt2 += (exchange2_position["size"] * exchange2_data["close"][i] * exchange2_position["curr_margin_pct"])
                    cash_usdt2 += ((-1) * exchange2_position["size"] * (exchange2_data["close"][i] - exchange2_position["entry_price"])) if is_profitable2 else 0

                    # Charge trading fee
                    fee1 = exchange1_position["size"] * exchange1_data["close"][i] * trading_fee
                    fee2 = exchange2_position["size"] * exchange2_data["close"][i] * trading_fee
                    cash_usdt1 -= fee1
                    cash_usdt2 -= fee2

                    exchange1_position = {}
                    exchange2_position = {}
                    continue

                # Check if exchange1_position's margin is sufficient
                if exchange1_position["curr_margin_pct"] < 0.06:
                    # sell some of the exchange2 position and buy more of the exchange1 position
                    # calculate the amount to sell to get exchange1 margin back to 0.1
                    req_capital = (exchange1_data["close"][i] * exchange1_position["size"]) * 0.1 - (starting_cash - cash_usdt1 + exchange1_position["margin_return"])
                    sell_size = int(req_capital / exchange2_data["close"][i])
                    exchange2_position["size"] -= sell_size

                    # Charge trading fee
                    fee2 = sell_size * exchange2_data["close"][i] * trading_fee
                    cash_usdt2 -= fee2

                    # update the margin percentage
                    additional_capital = sell_size * exchange2_data["close"][i]
                    exchange1_position["margin_return"] += additional_capital
                    exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])

                # Check if exchange2_position's margin is sufficient
                if exchange2_position["curr_margin_pct"] < 0.06:
                    # sell some of the exchange1 position and buy more of the exchange2 position
                    # calculate the amount to sell to get exchange2 margin back to 0.1
                    req_capital = (exchange2_data["close"][i] * exchange2_position["size"]) * 0.1 - (starting_cash - cash_usdt2 + exchange2_position["margin_return"])
                    sell_size = int(req_capital / exchange1_data["close"][i])
                    exchange1_position["size"] -= sell_size

                    # Charge trading fee
                    fee1 = sell_size * exchange1_data["close"][i] * trading_fee
                    cash_usdt1 -= fee1

                    # update the margin percentage
                    additional_capital = sell_size * exchange1_data["close"][i]
                    exchange2_position["margin_return"] += additional_capital
                    exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])

            # Exchange 1 is short
            else:
                # Track FRA returns
                fund1_return = exchange1_position["funding_rate"] * exchange1_position["size"] * exchange1_data["close"][i]
                fund2_return = (-1) * exchange2_position["funding_rate"] * exchange2_position["size"] * exchange2_data["close"][i]
                funding_rate_returns.append(fund1_return)
                funding_rate_returns.append(fund2_return)
                funding_rate_returns_combined.append(fund1_return + fund2_return)

                # update the position values
                exchange1_position["margin_return"] += (-1)*(exchange1_data["close"][i] - exchange1_data["close"][i-1]) * exchange1_position["size"] + fund1_return
                exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])
                exchange1_position["funding_rate"] = exchange1_data["funding_rate"][i]

                exchange2_position["margin_return"] += (exchange2_data["close"][i] - exchange2_data["close"][i-1]) * exchange2_position["size"] + fund2_return
                exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])
                exchange2_position["funding_rate"] = exchange2_data["funding_rate"][i]

                # Check if the funding rates flipped or simulation has ended and close out
                rate_diff = exchange1_data["funding_rate"][i] - exchange2_data["funding_rate"][i]
                if rate_diff < 0 or i == len(exchange1_data) - 1:
                    # Close the positions
                    # Returns include margin + leveraged profit
                    is_profitable1 = (-1)*(exchange1_data["close"][i] - exchange1_position["entry_price"]) > 0
                    cash_usdt1 += (exchange1_position["size"] * exchange1_data["close"][i] * exchange1_position["curr_margin_pct"])
                    cash_usdt1 += ((-1) * exchange1_position["size"] * (exchange1_data["close"][i] - exchange1_position["entry_price"])) if is_profitable1 else 0

                    is_profitable2 = (exchange2_data["close"][i] - exchange2_position["entry_price"]) > 0
                    cash_usdt2 += (exchange2_position["size"] * exchange2_data["close"][i] * exchange2_position["curr_margin_pct"])
                    cash_usdt2 += (exchange2_position["size"] * (exchange2_data["close"][i] - exchange2_position["entry_price"])) if is_profitable2 else 0

                    # Charge trading fee
                    fee1 = exchange1_position["size"] * exchange1_data["close"][i] * trading_fee
                    fee2 = exchange2_position["size"] * exchange2_data["close"][i] * trading_fee
                    cash_usdt1 -= fee1
                    cash_usdt2 -= fee2

                    exchange1_position = {}
                    exchange2_position = {}
                    continue

                # Check if exchange1_position's margin is sufficient
                if exchange1_position["curr_margin_pct"] < 0.06:
                    # sell some of the exchange2 position and buy more of the exchange1 position
                    # calculate the amount to sell to get exchange1 margin back to 0.1
                    req_capital = (exchange1_data["close"][i] * exchange1_position["size"]) * 0.1 - (starting_cash - cash_usdt1 + exchange1_position["margin_return"])
                    sell_size = int(req_capital / exchange2_data["close"][i])
                    exchange2_position["size"] -= sell_size

                    # Charge trading fee
                    fee2 = sell_size * exchange2_data["close"][i] * trading_fee
                    cash_usdt2 -= fee2

                    # update the margin percentage
                    additional_capital = sell_size * exchange2_data["close"][i]
                    exchange1_position["margin_return"] += additional_capital
                    exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])

                # Check if exchange2_position's margin is sufficient
                if exchange2_position["curr_margin_pct"] < 0.06:
                    # sell some of the exchange1 position and buy more of the exchange2 position
                    # calculate the amount to sell to get exchange2 margin back to 0.1
                    req_capital = (exchange2_data["close"][i] * exchange2_position["size"]) * 0.1 - (starting_cash - cash_usdt2 + exchange2_position["margin_return"])
                    sell_size = int(req_capital / exchange1_data["close"][i])
                    exchange1_position["size"] -= sell_size

                    # Charge trading fee
                    fee1 = sell_size * exchange1_data["close"][i] * trading_fee
                    cash_usdt1 -= fee1

                    # update the margin percentage
                    additional_capital = sell_size * exchange1_data["close"][i]
                    exchange2_position["margin_return"] += additional_capital
                    exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])

        # If no position is open
        else:
            # Open a 10x long position with favorable funding rates
            rate_diff = exchange1_data["funding_rate"][i] - exchange2_data["funding_rate"][i]

            if rate_diff >= 0:
                exchange1_position["type"] = "short"
                exchange1_position["entry_price"] = exchange1_data["close"][i]
                exchange1_position["size"] = int(starting_cash * leverage / exchange1_data["close"][i])
                exchange1_position["margin_return"] = 0
                cash_usdt1 = 0
                exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])
                exchange1_position["funding_rate"] = exchange1_data["funding_rate"][i]

                exchange2_position["type"] = "long"
                exchange2_position["entry_price"] = exchange2_data["close"][i]
                exchange2_position["size"] = int(starting_cash * leverage / exchange2_data["close"][i])
                exchange2_position["margin_return"] = 0
                cash_usdt2 = 0
                exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])
                exchange2_position["funding_rate"] = exchange2_data["funding_rate"][i]

                # Charge trading fee
                fee1 = exchange1_position["size"] * exchange1_data["close"][i] * trading_fee
                cash_usdt1 -= fee1

                fee2 = exchange2_position["size"] * exchange2_data["close"][i] * trading_fee
                cash_usdt2 -= fee2

            else:
                exchange1_position["type"] = "long"
                exchange1_position["entry_price"] = exchange1_data["close"][i]
                exchange1_position["size"] = int(starting_cash * leverage / exchange1_data["close"][i])
                exchange1_position["margin_return"] = 0
                cash_usdt1 = 0
                exchange1_position["curr_margin_pct"] = (starting_cash - cash_usdt1 + exchange1_position["margin_return"]) / (exchange1_data["close"][i] * exchange1_position["size"])
                exchange1_position["funding_rate"] = exchange1_data["funding_rate"][i]

                exchange2_position["type"] = "short"
                exchange2_position["entry_price"] = exchange2_data["close"][i]
                exchange2_position["size"] = int(starting_cash * leverage / exchange2_data["close"][i])
                exchange2_position["margin_return"] = 0
                cash_usdt2 = 0
                exchange2_position["curr_margin_pct"] = (starting_cash - cash_usdt2 + exchange2_position["margin_return"]) / (exchange2_data["close"][i] * exchange2_position["size"])
                exchange2_position["funding_rate"] = exchange2_data["funding_rate"][i]

                # Charge trading fee
                fee1 = exchange1_position["size"] * exchange1_data["close"][i] * trading_fee
                cash_usdt1 -= fee1

                fee2 = exchange2_position["size"] * exchange2_data["close"][i] * trading_fee
                cash_usdt2 -= fee2

    # Sum the Funding Rate Arbitrage returns
    cum_funding_rate_returns = sum(funding_rate_returns)
    fra_return = cum_funding_rate_returns + cash_usdt1 + cash_usdt2 - starting_cash*2
    fra_returns_pct = fra_return / (starting_cash*2)
    annualized_fra_return = (1 + fra_returns_pct) ** (365 / days_in_past) - 1

    print(f"FRA Time Period: {len(exchange1_data)/3:.2f} days")
    print(f"Leverage: {leverage}")
    print(f"Trading Fee: {trading_fee*100}%")
    print(f"Funding Rate Arbitrage Returns: ${fra_return:.2f} | {fra_returns_pct * 100:.2f}% | APY: {annualized_fra_return * 100:.2f}%")
