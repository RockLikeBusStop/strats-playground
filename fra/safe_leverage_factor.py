import pandas as pd
import ccxt
from datetime import datetime, timedelta

def calculate_top_negative_returns(last_n_days: int = 365, top_n: int = 10) -> pd.DataFrame:
    exchange1 = ccxt.binanceusdm()
    exchange2 = ccxt.kucoinfutures()

    ms_since = int((datetime.now() - timedelta(days=last_n_days)).timestamp()) * 1000

    sol_usdt_prices1 = exchange1.fetch_ohlcv("SOL/USDT", timeframe="1d", since=ms_since)
    sol_usdt_prices2 = exchange2.fetch_ohlcv("SOLUSDTM", timeframe="1d", since=ms_since)

    df1 = pd.DataFrame(sol_usdt_prices1, columns=["date", "open", "high", "low", "close", "volume"])
    df2 = pd.DataFrame(sol_usdt_prices2, columns=["date", "open", "high", "low", "close", "volume"])

    # Convert timestamp to datetime
    df1["date"] = pd.to_datetime(df1["date"], unit="ms")
    df2["date"] = pd.to_datetime(df2["date"], unit="ms")
    df1.set_index("date", inplace=True)
    df2.set_index("date", inplace=True)

    # Calculate daily returns
    df1['daily_return(%)'] = df1['close'].pct_change() * 100
    df2['daily_return(%)'] = df2['close'].pct_change() * 100

    # Find the top N highest negative returns
    top_negative_returns1 = df1['daily_return(%)'].nsmallest(top_n).to_frame()
    top_negative_returns2 = df2['daily_return(%)'].nsmallest(top_n).to_frame()
    top_negative_returns1.columns = ['negative_return(%)']
    top_negative_returns2.columns = ['negative_return(%)']

    # Include the corresponding date for context
    top_negative_returns1['date'] = top_negative_returns1.index
    top_negative_returns2['date'] = top_negative_returns2.index

    return top_negative_returns1.reset_index(drop=True), top_negative_returns2.reset_index(drop=True)

def calculate_max_safe_leverage(print_results: bool = False):
    returns1, returns2 = calculate_top_negative_returns(last_n_days=365*1.5, top_n=5)

    maintenance_margin_rate = 0.05
    safe_max_leverage_factor1 = abs(1 / (returns1['negative_return(%)'].min()/100 - maintenance_margin_rate))
    safe_max_leverage_factor2 = abs(1 / (returns2['negative_return(%)'].min()/100 - maintenance_margin_rate))

    if print_results:
        print(f"Binance largest SOL drawdowns in last 1.5 years:\n{returns1}")
        print(f"\nKuCoin largest SOL drawdowns in last 1.5 years:\n{returns2}")
        print(f"\nSafe maximum leverage factor1: %.2f" % safe_max_leverage_factor1)
        print(f"Safe maximum leverage factor2: %.2f" % safe_max_leverage_factor2)

    return safe_max_leverage_factor1, safe_max_leverage_factor2