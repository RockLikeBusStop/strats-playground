import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def run_buy_x_pct_dip(start_date=(datetime.now() - timedelta(days=5 * 365)), end_date=datetime.now(), num_years=5, drop_threshold=0.05, ticker='^GSPC'):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Define the strategy
    class BuyXPctDip(bt.Strategy):
        params = (('sma_period', 1), ('drop_threshold', drop_threshold), ('investment_amount', 100_000))

        def __init__(self):
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
            self.additional_cash = 0
            self.last_sma_value = None
            self.initial_investment_done = False

        def next(self):
            if not self.initial_investment_done:
                self.initial_investment_done = True
                size = 1_000_000 / self.data.close[0]
                self.buy(size=size)
                self.last_sma_value = self.sma[0]
                return

            # Calculate drop in SMA
            sma_pct_drop = (self.last_sma_value - self.sma[0]) / self.last_sma_value if self.last_sma_value != 0 else 0

            # Check if SMA has dropped by more than the threshold
            if sma_pct_drop >= self.p.drop_threshold:
                # Buy S&P 500 with 10% of the remaining cash
                self.additional_cash += self.p.investment_amount
                size = self.p.investment_amount / self.data.close[0]
                self.buy(size=size)
                print(f"Drop in SMA: {sma_pct_drop}; Size: {size}")

            # Update last SMA value
            self.last_sma_value = self.sma[0]

    # Initialize cerebro engine
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BuyXPctDip)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set initial cash
    starting_cash = 50_000_000
    cerebro.broker.set_cash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)

    # Run backtest
    backtest_results = cerebro.run()

    # Calculate strategy performance
    initial_position = 1_000_000
    total_capital = backtest_results[0].additional_cash + initial_position
    unused_cash = starting_cash - total_capital

    portfolio_value = cerebro.broker.getvalue() - unused_cash
    print(f"Additional cash: {backtest_results[0].additional_cash}; Portfolio value: {portfolio_value}; Total capital: {total_capital}")
    strategy_total_return = (portfolio_value - total_capital) / total_capital
    strategy_annualized_return = ((1 + strategy_total_return) ** (1 / num_years)) - 1

    # Calculate S&P 500 performance
    sp500_total_return = (data['Close'][-1] - data['Close'][0]) / data['Close'][0]
    sp500_annualized_return = ((1 + sp500_total_return) ** (1 / num_years)) - 1

    print(f"\nTotal Return of Strategy over {num_years} Years: {strategy_total_return * 100:.2f}%")
    print(f"Total Return of S&P 500 over {num_years} Years: {sp500_total_return * 100:.2f}%")

    print(f"\nAnnualized Total Return of Strategy: {strategy_annualized_return * 100:.2f}%")
    print(f"Annualized Total Return of S&P 500: {sp500_annualized_return * 100:.2f}%")

    return strategy_total_return, sp500_total_return, strategy_annualized_return, sp500_annualized_return

# def main():
#     num_years = 5
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=num_years * 365)
#     drop_threshold = 0.15
#     strategy_total_return, sp500_total_return, strategy_annualized_return, sp500_annualized_return = run_buy_x_pct_dip(start_date, end_date, num_years, drop_threshold)

#     print(f"\nTotal Return of Strategy over {num_years} Years: {strategy_total_return * 100:.2f}%")
#     print(f"Total Return of S&P 500 over {num_years} Years: {sp500_total_return * 100:.2f}%")

#     print(f"\nAnnualized Total Return of Strategy: {strategy_annualized_return * 100:.2f}%")
#     print(f"Annualized Total Return of S&P 500: {sp500_annualized_return * 100:.2f}%")


# main()