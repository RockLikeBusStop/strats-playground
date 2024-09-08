# Trading Strategies Playground

This repository is playground for various trading strategies. The strategies include Simple Moving Average (SMA) Crossover, Buy-and-Hold, Funding Rate Arbitrage (FRA), Put-Call Parity Arbitrage (PCPA), ARIMA forecasting, and a lot more.

## Strategies

1. **Buy-and-Hold**: A basic strategy that buys and holds Bitcoin for a specified period.
2. **SMA Crossover**: Implements a simple moving average crossover strategy with customizable parameters.
3. **Funding Rate Arbitrage (FRA)**: Exploits the difference in funding rates between exchanges.
4. **Put-Call Parity Arbitrage (PCPA)**: Identifies arbitrage opportunities based on put-call parity in options markets.
5. **ARIMA**: Uses the ARIMA model for time series forecasting and trading.
6. **Buy-X-Pct-Dip**: Buys an asset when the SMA drops by a certain percentage.

## Files

- `main.py`: The main script to run different strategies.
- `sma_cross.py`: Contains the SMA Crossover strategy implementation.
- `btc_hold_strat.py`: Implements the Buy-and-Hold strategy for Bitcoin.
- `fra_arb.py`: Contains the Funding Rate Arbitrage strategy.
- `safe_leverage_factor.py`: Calculates the safe leverage factor for FRA.
- `pcpa.py`: Implements the Put-Call Parity Arbitrage strategy.
- `arima.py`: Contains the ARIMA forecasting strategy.
- `top_coins.py`: Utility to fetch top cryptocurrencies by market cap and volume.
- `buy_x_pct_dip.py`: Contains the Buy-X-Pct-Dip strategy.

## Usage

To run a strategy, uncomment the relevant function call in `main.py` and execute:

```
python main.py
```

## Requirements

- Python 3.7+
- ccxt
- pandas
- backtrader
- empyrical
- statsmodels
- pycoingecko

Install the required packages using:

```
pip install -r requirements.txt
```