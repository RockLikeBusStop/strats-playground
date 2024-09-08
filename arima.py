import backtrader as bt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import ccxt

# Custom Strategy using ARIMA
class ARIMAStrategy(bt.Strategy):
    params = (
        ('model_order', (5, 1, 0)),  # ARIMA order (p, d, q)
        ('lookback', 30),            # Lookback period for ARIMA model training
        ('forecast_period', 5),      # Forecast period for ARIMA model
    )

    def __init__(self):
        self.order_percentage = 0.95
        self.dataclose = self.datas[0].close
        self.order = None  # To keep track of pending orders
        self.forecast_values = []

    def next(self):
        # If we don't have enough data yet, skip this step
        if len(self.data) < self.params.lookback:
            return

        # Get the closing prices for the lookback period
        close_prices = np.array(self.dataclose.get(size=self.params.lookback))

        # Train ARIMA model and forecast
        model = ARIMA(close_prices, order=self.params.model_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=self.params.forecast_period)[0]  # Forecasting for the given period

        self.forecast_values.append(forecast)  # Store forecast values for analysis

        # Buy/Sell decision based on forecast
        current_price = self.dataclose[0]
        if forecast > current_price:  # If forecast is higher, we buy
            if not self.position:  # Check if no existing position
                self.order = self.buy(size=self.broker.getcash() * self.order_percentage / current_price)
        elif forecast < current_price:  # If forecast is lower, we sell
            if self.position:  # Check if there is an existing position
                self.order = self.sell(size=self.position.size)

# Data Preparation and Backtesting
def try_arima(days_in_past=365 , token="BTC"):
    secs_since = datetime.now() - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000

    # Fetch historical data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(f"{token}/USDT", timeframe="1d", since=ms_since)
    dataframe = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp to datetime
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
    dataframe.set_index("timestamp", inplace=True)

    data = bt.feeds.PandasData(dataname=dataframe)

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ARIMAStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(1_000_000)  # Set initial capital
    cerebro.broker.setcommission(commission=0.001)  # Set commission fee

    # Run the backtest
    start_value = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % start_value)
    cerebro.run()
    end_value = cerebro.broker.getvalue()
    print('Ending Portfolio Value: %.2f' % end_value)

    total_return = (end_value - start_value) / start_value
    print('Total Return for %d days: %.2f%%' % (days_in_past, total_return * 100))

    apy = (1 + total_return) ** (365 / days_in_past) - 1
    print("Days in past: %d" % days_in_past)
    print('APY: %.2f%%' % (apy * 100))

    # Plot the results
    cerebro.plot()

def optimize_arima(days_in_past=365 , token="BTC"):
    lookback = 15
    forecast_period = 5
    model_orders  = [(3, 1, 0), (5, 2, 0), (3, 2, 0), (5, 0, 0), (3, 0, 1)]
    # model_orders  = [(3, 0, 0), (5, 0, 0), (3, 0, 2), (5, 0, 1), (5, 0, 2), (3, 0, 1)]
    # model_orders  = [(10, 2, 0), (10, 0, 0), (10, 1, 0), (10, 0, 1)]

    secs_since = datetime.now() - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000

    # Fetch historical data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(f"{token}/USDT", timeframe="1d", since=ms_since)
    dataframe = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp to datetime
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
    dataframe.set_index("timestamp", inplace=True)

    data = bt.feeds.PandasData(dataname=dataframe)

    returns_by_model_order = {}

    # Initialize Cerebro engine
    for model_order in model_orders:
        print("\nTesting model order:", model_order, "\n")
        try:
            cerebro = bt.Cerebro()
            cerebro.addstrategy(ARIMAStrategy, model_order=model_order, lookback=lookback, forecast_period=forecast_period)
            cerebro.adddata(data)
            cerebro.broker.setcash(1_000_000)  # Set initial capital
            cerebro.broker.setcommission(commission=0.001)  # Set commission fee

            # Run the backtest
            start_value = cerebro.broker.getvalue()
            cerebro.run()
            end_value = cerebro.broker.getvalue()

            total_return = (end_value - start_value) / start_value
            apy = (1 + total_return) ** (365 / days_in_past) - 1

            returns_by_model_order[model_order] = total_return

            print("Testing model order:", model_order)
            print('Total $ Return: %.2f' % end_value)
            print('Total Return for %d days: %.2f%%' % (days_in_past, total_return * 100))
            print('APY: %.2f%%' % (apy * 100))
            cerebro.plot()
        except:
            print("Failed to test model order:", model_order)

    # Sort returns_by_model_order by total return in descending order
    sorted_returns = dict(sorted(returns_by_model_order.items(), key=lambda item: item[1], reverse=True))

    print("\nResults:")
    for model_order, total_return in sorted_returns.items():
        print(f"Model Order: {model_order}, Total Return: {total_return*100:.2f}%")
