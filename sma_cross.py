import ccxt
import pandas as pd
import backtrader as bt
import empyrical as ep
from datetime import datetime, timedelta
from top_coins import get_top_coins

def try_sma_cross(
    days_in_past=365, coin="BTC/USDT", exchange=ccxt.binanceusdm(), sma_params=None, mode="default"
):
    # Historical data timeframe
    secs_since = datetime.now() - timedelta(days=days_in_past)
    ms_since = int(secs_since.timestamp()) * 1000

    # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(coin, timeframe="1d", since=ms_since)
    data = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("timestamp", inplace=True)

    # Split data into training and testing sets
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Build a simple moving average crossover strategy
    class SmaCross(bt.Strategy):
        params = (("pfast", 5), ("pslow", 15))

        def __init__(self):
            sma1 = bt.ind.SMA(period=self.p.pfast)
            sma2 = bt.ind.SMA(period=self.p.pslow)
            self.crossover = bt.ind.CrossOver(sma1, sma2)

            self.diff = bt.ind.NonZeroDifference(sma1, sma2)
            self.start_value = self.broker.get_cash()
            self.daily_values = []

        def next(self):
            percent_cash_per_trade = 0.75
            size = int(self.start_value * percent_cash_per_trade / self.data[0])

            if self.crossover > 0:
                self.close()
                self.buy(size=size)
            elif self.crossover < 0:
                self.close()
                self.sell(size=size)

            self.daily_values.append(self.broker.getvalue())

    # Backtest the strategy
    if mode == "train":
        data = train_data
    elif mode == "test":
        data = test_data

    cerebro = bt.Cerebro()
    datafeed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(datafeed)
    # Add the strategy with custom parameters
    if sma_params is not None:
        cerebro.addstrategy(SmaCross, **sma_params)
    else:
        cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(1_000_000)
    start_value = cerebro.broker.getvalue()
    sma = cerebro.run()
    end_value = cerebro.broker.getvalue()

    print(f"\nStart Value: {start_value:.2f}")
    print(f"End Value: {end_value:.2f}")

    # Calculate the strategy's performance
    returns = sma[0].daily_values
    returns_percent = pd.Series(returns).pct_change()

    cum_returns = ep.cum_returns(returns_percent).iloc[-1]
    annual_return = 0.0 if cum_returns < 0 else ep.annual_return(returns_percent, annualization=365)
    # annual_return = ep.annual_return(returns_percent, annualization=365)

    sharpe_ratio = 0.0 if cum_returns <= 0 else ep.sharpe_ratio(returns_percent, annualization=365)

    max_drawdown = ep.max_drawdown(returns_percent)

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Strategy Returns: {cum_returns * 100:.2f}%")
    print(f"Annualized Returns: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

    return {
        "sharpe_ratio": sharpe_ratio,
        "cum_returns": cum_returns,
        "annualized_return": annual_return,
        "max_drawdown": max_drawdown,
    }


def optimize_sma_cross(days_in_past=365, mode="default"):
    top_coins = get_top_coins(30)
    # sma_param_pairs = [(5, 15), (10, 30), (20, 50)]
    # sma_param_pairs = [(1, 5), (5, 30), (10, 50)]
    # sma_param_pairs = [(3, 50), (5, 20), (15, 40)]
    # sma_param_pairs = [(3, 50), (5, 20), (1, 5)]
    sma_param_pairs = [(3, 10), (5, 20), (1, 5)]
    # sma_param_pairs = [(3, 10), (5, 20), (10, 50)]
    # sma_param_pairs = [(1, 5), (3, 10), (5, 20), (10, 50)]
    # sma_param_pairs = [(5, 20)]
    # sma_param_pairs = [(3, 10)]
    # sma_param_pairs = [(1, 5)]

    returns_by_coin = {}
    missed_coins = []

    for coin in top_coins:
        usdt_coin_pair = f"{coin['symbol'].upper()}/USDT"
        # usdc_coin_pair = f"{coin['symbol'].upper()}/USDC"

        for param in sma_param_pairs:
            try:
                print(f"\nTesting SMA Crossover strategy for {usdt_coin_pair} x {param}...")
                returns_by_coin[f"{usdt_coin_pair} x {param}"] = try_sma_cross(
                    coin=usdt_coin_pair,
                    sma_params={"pfast": param[0], "pslow": param[1]},
                    days_in_past=days_in_past,
                    mode=mode,
                )
            except:
                print(f"Failed to test {usdt_coin_pair}...")
                missed_coins.append(usdt_coin_pair)
                break

    # Sort coins by cumulative returns
    sorted_coins = sorted(
        returns_by_coin.items(), key=lambda x: x[1]["cum_returns"], reverse=True
    )

    print("\nResults:")
    for coin, results in sorted_coins:
        print(
            f"{coin} => Returns: {results['cum_returns'] * 100:.2f}% | APY: {results['annualized_return'] * 100:.2f}% | Sharpe: {results['sharpe_ratio']:.2f} | Max Drawdown: {results['max_drawdown'] * 100:.2f}%"
        )

    if missed_coins:
        print("\nFailed to test the strategy for the following coins:", missed_coins)

    # Score of each sma param pair tested
    sma_param_scores = {}
    for param in sma_param_pairs:
        for coin, results in sorted_coins:
            if coin.endswith(f"{param}"):
                if param not in sma_param_scores:
                    sma_param_scores[param] = sorted_coins.index((coin, results))
                sma_param_scores[param] += sorted_coins.index((coin, results))

    print("\nSMA Param Scores:", sma_param_scores)

    # Score of each sma param pair tested
    sma_param_scores2 = {}
    for param in sma_param_pairs:
        for coin, results in sorted_coins:
            if coin.endswith(f"{param}"):
                if param not in sma_param_scores2:
                    sma_param_scores2[param] = 0
                sma_param_scores2[param] += results["cum_returns"]

    print("\nSMA Param Returns:", sma_param_scores2)


def get_good_sma_results():
    try_sma_cross(coin='SOL/USDT', sma_params={'pfast': 3, 'pslow': 10}, days_in_past=365)

    # try_sma_cross(coin='SOL/USDT', sma_params={'pfast': 1, 'pslow': 5}, days_in_past=365)
    # try_sma_cross(coin='FET/USDT', sma_params={'pfast': 1, 'pslow': 5}, days_in_past=400)
    # try_sma_cross(coin='NEAR/USDT', sma_params={'pfast': 1, 'pslow': 5}, days_in_past=400)
    # try_sma_cross(coin='STX/USDT', sma_params={'pfast': 1, 'pslow': 5}, days_in_past=400)

