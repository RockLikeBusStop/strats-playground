from btc_hold_strat import btc_hold_strat_results
from sma_cross import optimize_sma_cross, get_good_sma_results
from fra_arb import try_fra_arb
from pcpa import calculate_pcpa_return
from buy_x_pct_dip import run_buy_x_pct_dip


def main():
    # print("\nRunning an example of Buy-and-Hold strategy for BTC/USDT...")
    # btc_hold_strat_results(days_in_past=400)

    # print("Running an example of SMA Crossover strategy optimization...")
    # optimize_sma_cross(days_in_past=400)

    # print("\nRunning an example of FRA strategy...")
    # try_fra_arb(days_in_past=66.7)

    # print("\nRunning an example of PCPA strategy...")
    # calculate_pcpa_return("deribit")
    # calculate_pcpa_return("binance")

    print("\nRunning an example of Buy-X-Pct-Dip strategy...")
    run_buy_x_pct_dip(num_years=5, drop_threshold=0.05, ticker='^GSPC')


main()