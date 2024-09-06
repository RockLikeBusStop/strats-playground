import ccxt
import pandas as pd
import backtrader as bt
import empyrical as ep
from datetime import datetime, timedelta
import numpy as np


def btc_hold_strat_results(days_in_past=365):
    # Historical data timeframe
    secs_since = datetime.now() - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000

    # Fetch historical data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", since=ms_since)
    data = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("timestamp", inplace=True)

    # Buy-and-Hold Strategy
    class BuyAndHold(bt.Strategy):
        def start(self):
            self.val_start = self.broker.get_cash()
            # print(f'Original Portfolio Value: {self.val_start}')

        def nextstart(self):
            self.order_target_value(target=self.broker.get_cash())
            # print(f'Price when buying: {self.data[0]}')

        # def stop(self):
        # print(f'Final Portfolio Value: {self.broker.getvalue()}')
        # self.roi = (self.broker.get_value() / self.val_start) - 1.0
        # print(f'ROI: {self.roi*100:.2f}%')

    # Backtest the strategy
    cerebro = bt.Cerebro()
    datafeed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(datafeed)
    cerebro.addstrategy(BuyAndHold)
    cerebro.broker.setcash(1_000_000)
    start_value = cerebro.broker.getvalue()
    sma = cerebro.run()
    end_value = cerebro.broker.getvalue()

    print("Testing Buy-and-Hold Strategy for BTC/USDT...")
    print(f"\nStart Value: {start_value:.2f}")
    print(f"End Value: {end_value:.2f}")

    # Calculate the strategy's performance
    returns = data["close"].pct_change()
    sharpe_ratio = ep.sharpe_ratio(returns, annualization=365)
    cum_returns = ep.cum_returns(returns)[-1]
    annual_return = ep.annual_return(returns, annualization=365)
    max_drawdown = ep.max_drawdown(returns)

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Strategy Returns: {cum_returns * 100:.2f}%")
    print(f"Annualized Returns: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    return {
        "sharpe_ratio": sharpe_ratio,
        "cum_returns": cum_returns,
        "annualized_return": annual_return,
    }
