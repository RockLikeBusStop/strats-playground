import math
import pandas as pd
import requests
from datetime import datetime, timedelta
import requests

def get_binance_data(symbol="BTC"):
    options_data_url = "https://eapi.binance.com/eapi/v1/ticker"

    # Fetch options data from Binance API
    options_response = requests.get(options_data_url)
    options_data = options_response.json()

    scenarios_unfiltered = {}
    r = 0.03
    for option in options_data:
        if option["symbol"][:3] == symbol:
            expiration = datetime.strptime(option["symbol"].split('-')[1], '%y%m%d')
            scenario_name = option["symbol"][:-2]
            option_type = option["symbol"][-1]
            stock_price = float(option["exercisePrice"])
            strike_price = int(option["strikePrice"])

            # if the ask_price or bid_price is None, skip the option
            if option["askPrice"] == None or option["bidPrice"] == None:
                continue

            # if you've already come across this scenario
            if scenario_name in scenarios_unfiltered:
                # if you've seen the put already, this is the call. update the call price
                if option_type == 'C':
                    scenarios_unfiltered[scenario_name][4] = float(option["askPrice"])
                    scenarios_unfiltered[scenario_name][5] = float(option["bidPrice"])
                else:
                    scenarios_unfiltered[scenario_name][6] = float(option["askPrice"])
                    scenarios_unfiltered[scenario_name][7] = float(option["bidPrice"])
            else:
                t = (expiration - datetime.now()).days
                if option_type == 'C':
                    c_a = float(option["askPrice"])
                    c_b = float(option["bidPrice"])
                    scenarios_unfiltered[scenario_name] = [stock_price, strike_price, r, t, c_a, c_b, 0, 0]
                else:
                    p_a = float(option["askPrice"])
                    p_b = float(option["bidPrice"])
                    scenarios_unfiltered[scenario_name] = [stock_price, strike_price, r, t, 0, 0, p_a, p_b]

    # Go through the scenarios_dict and remove the scenarios where any of the prices are 0 or None
    scenarios = {}
    for scenario_name, scenario in scenarios_unfiltered.items():
        if not (0 in scenario or None in scenario):
            scenarios[scenario_name] = scenario

    return scenarios

def get_deribit_data():
    base_url = "https://www.deribit.com/api/v2/public/"

    options_response = requests.get(f"{base_url}get_book_summary_by_currency?currency=BTC&kind=option")
    options_data = options_response.json()['result']

    scenarios_unfiltered = {}
    r = 0.03
    for option in options_data:
        expiration = datetime.strptime(option["underlying_index"].split('-')[1], '%d%b%y')
        scenario_name = option["instrument_name"][:-2]
        option_type = option["instrument_name"][-1]
        stock_price = option["estimated_delivery_price"]
        strike_price = int(option["instrument_name"].split('-')[2])

        # if the ask_price or bid_price is None, skip the option
        if option["ask_price"] == None or option["bid_price"] == None:
            continue

        # if you've already come across this scenario
        if scenario_name in scenarios_unfiltered:
            # if you've seen the put already, this is the call. update the call price
            if option_type == 'C':
                scenarios_unfiltered[scenario_name][4] = option["ask_price"] * stock_price
                scenarios_unfiltered[scenario_name][5] = option["bid_price"] * stock_price
            else:
                scenarios_unfiltered[scenario_name][6] = option["ask_price"] * stock_price
                scenarios_unfiltered[scenario_name][7] = option["bid_price"] * stock_price
        else:
            t = (expiration - datetime.now()).days
            if option_type == 'C':
                c_a = option["ask_price"] * stock_price
                c_b = option["bid_price"] * stock_price
                scenarios_unfiltered[scenario_name] = [stock_price, strike_price, r, t, c_a, c_b, 0, 0]
            else:
                p_a = option["ask_price"] * stock_price
                p_b = option["bid_price"] * stock_price
                scenarios_unfiltered[scenario_name] = [stock_price, strike_price, r, t, 0, 0, p_a, p_b]

    # Go through the scenarios_dict and remove the scenarios where any of the prices are 0 or None
    scenarios = {}
    for scenario_name, scenario in scenarios_unfiltered.items():
        if not (0 in scenario or None in scenario):
            scenarios[scenario_name] = scenario

    return scenarios


def calculate_pcpa_return(exchange):
    if exchange == "deribit":
        data = get_deribit_data()
    elif exchange == "binance":
        data = get_binance_data()
    else:
        raise ValueError("Invalid exchange")

    results = []

    for scenario_name, scenario in data.items():
        s, k, r, t, c_a, c_b, p_a, p_b = scenario
        T = t / 365

        # Put-Call Parity: C + PV(K) = P + S
        # S = stock price
        # K = strike price
        # r = risk-free rate
        # t = time to expiration (in days)
        # pv(k) = present value of the strike price
        # Check if arbitrage opportunity exists when buying c + pv(k) and selling p + s or vice versa
        # If arbitrage opportunity exists, calculate the return
        # Store the results
        sell_put_return = 0
        sell_call_return = 0

        # when buying c + pv(k) and selling p + s
        sell_put_return = p_b + s - c_a - k * math.exp(-r * T)
        # when buying p + s and selling c + pv(k)
        sell_call_return = c_b + k - p_a - s

        if sell_call_return <= 0 and sell_put_return <= 0:
            results.append({
                'Scenario': scenario_name,
                'Data': scenario,
                'Arb. Type': 'No Arbitrage Opportunity',
                'Return': sell_call_return if sell_call_return > sell_put_return else sell_put_return,
                # 'Other Return': sell_call_return if sell_call_return < sell_put_return else sell_put_return,
                'returns_pct': 0
            })
        elif sell_put_return > sell_call_return:
            returns_pct = sell_put_return / (c_a + k - s - p_b) if (c_a + k - s - p_b) != 0 else 0
            results.append({
                'Scenario': scenario_name,
                'Data': scenario,
                'Arb. Type': 'Sold Put',
                'Return': sell_put_return,
                # 'Other Return': sell_call_return,
                'returns_pct': returns_pct
            })
        else:
            returns_pct = sell_call_return / (s + p_a - c_b) if (s + p_a - c_b) != 0 else 0
            results.append({
                'Scenario': scenario_name,
                'Data': scenario,
                'Arb. Type': 'Sold Call',
                'Return': sell_call_return,
                # 'Other Return': sell_put_return,
                'returns_pct': returns_pct
            })

    results_pd = pd.DataFrame(results)

    # add a annualized return column
    results_pd['APY (%)'] = ((1 + results_pd['returns_pct']) ** (365 / results_pd['Data'].apply(lambda x: x[3])) - 1) * 100
    results_pd = results_pd.sort_values(by='APY (%)', ascending=False)

    # pd.set_option('display.max_rows', None)
    print(f"\nPut-Call Arbitrage Opportunities on {exchange}:")
    print(results_pd)
    return results_pd
