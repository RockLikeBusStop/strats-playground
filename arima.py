import ccxt
import pandas as pd
import backtrader as bt
import empyrical as ep
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

# Build a ARIMA strategy
class ARIMAStrategy(bt.Strategy):
    params = (
            ('p', 1),
            ('d', 1),
            ('q', 1),
            ('window', 30),
        )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.arima_forecast = None

        self.start_value = self.broker.get_cash()

        self.daily_values = []

    def next(self):
        self.update_arima_forecast()

        percent_cash_per_trade = 0.1
        size = int(self.start_value * percent_cash_per_trade / self.data[0])

        if len(self.data_close) < self.params.window:
            return

        if self.order or not self.arima_forecast:
            return

        print(f'ARIMA Forecast: {self.arima_forecast:.2f}, Close: {self.data_close[0]:.2f}')
        if self.position.size == 0:
            if self.arima_forecast > self.data_close[0]:
                # self.close()
                self.order = self.buy(size=size)
        else:
            if self.arima_forecast < self.data_close[0]:
                # self.close()
                self.order = self.sell(size=size)


        self.daily_values.append(self.broker.getvalue())

        if self.position.size != 0:
            print(f'{self.position}\nPosition Value: {self.broker.getvalue():.2f}')

    def update_arima_forecast(self):
        # print(f'ARIMA Params: {self.params.p}, {self.params.d}, {self.params.q}, {self.params.window}')
        # curr_window = len(self.data_close) if len(self.data_close) - self.params.window < 0 else len(self.data_close) - self.params.window
        # print(f'Current Window: {curr_window}, Data Length: {len(self.data_close)}')
        data = []
        for i in range(self.params.window):
            # print(f'Deatum: {self.data_close[i]}, {len(self.data_close)}')
            if i < len(self.data_close):
                data.append(self.data_close[i])

        # data = self.data_close.get(size=curr_window)
        # for datum in self.data_close:
        #     print(f'Datum: {datum}, {len(self.data_close)}')
        model = ARIMA(data, order=(self.params.p, self.params.d, self.params.q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        print(f'ARIMA Forecast1: {forecast[0]}')
        self.arima_forecast = forecast[0]


def try_arima(
    days_in_past=365, coin="BTC/USDT", exchange=ccxt.binanceusdm(), arima_params=None, mode="default"
):
    # Historical data timeframe
    secs_since = datetime.now() - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000

    # Fetch historical data
    # exchange = ccxt.binanceusdm()
    ohlcv = exchange.fetch_ohlcv(coin, timeframe="1d", since=ms_since)
    data = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # print(ohlcv.__len__())
    # print(ohlcv[0], ohlcv[1], ohlcv[ohlcv.__len__()-2], ohlcv[ohlcv.__len__()-1])

    # Convert timestamp to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("timestamp", inplace=True)

    # Split data into training and testing sets
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Backtest the strategy
    if mode == "train":
        data = train_data
    elif mode == "test":
        data = test_data

    cerebro = bt.Cerebro()
    datafeed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(datafeed)
    # Add the strategy with custom parameters
    if arima_params is not None:
        cerebro.addstrategy(ARIMAStrategy, **arima_params)
    else:
        cerebro.addstrategy(ARIMAStrategy)
    cerebro.broker.setcash(1_000_000)
    start_value = cerebro.broker.getvalue()
    arima_strat = cerebro.run()
    end_value = cerebro.broker.getvalue()

    print(f"\nStart Value: {start_value:.2f}")
    print(f"End Value: {end_value:.2f}")

    # # Calculate the strategy's performance
    # returns = arima_strat[0].daily_values
    # returns_percent = pd.Series(returns).pct_change()

    # cum_returns = ep.cum_returns(returns_percent).iloc[-1]
    # annual_return = 0.0 if cum_returns < 0 else ep.annual_return(returns_percent, annualization=365)
    # # annual_return = ep.annual_return(returns_percent, annualization=365)

    # sharpe_ratio = 0.0 if cum_returns <= 0 else ep.sharpe_ratio(returns_percent, annualization=365)

    # max_drawdown = ep.max_drawdown(returns_percent)

    # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    # print(f"Strategy Returns: {cum_returns * 100:.2f}%")
    # print(f"Annualized Returns: {annual_return * 100:.2f}%")
    # print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    # return {
    #     "sharpe_ratio": sharpe_ratio,
    #     "cum_returns": cum_returns,
    #     "annualized_return": annual_return,
    #     "max_drawdown": max_drawdown,
    # }
