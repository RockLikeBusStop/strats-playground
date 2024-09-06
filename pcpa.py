import math
import pandas as pd
import requests
from datetime import datetime, timedelta
import requests

def get_binance_data(symbol="BTC"):
    options_data_url = "https://eapi.binance.com/eapi/v1/ticker"
    price_data_url = "https://fapi.binance.com/fapi/v1/ticker/price"

    # Fetch options data from Binance API
    options_response = requests.get(options_data_url)
    options_data = options_response.json()

    # price_response = requests.get(price_data_url, params={"symbol": "BTCUSDT"})
    # price_data = price_response.json()["price"]

    scenarios_unfiltered = {}
    r = 0.03
    for option in options_data:
        if option["symbol"][:3] == symbol:

        # {
        #     "symbol": "BTC-240927-45000-C",
        #     "priceChange": "0",
        #     "priceChangePercent": "0",
        #     "lastPrice": "17000",
        #     "lastQty": "0",
        #     "open": "17000",
        #     "high": "17000",
        #     "low": "17000",
        #     "volume": "0",
        #     "amount": "0",
        #     "bidPrice": "10",
        #     "askPrice": "0",
        #     "openTime": 0,
        #     "closeTime": 0,
        #     "firstTradeId": 0,
        #     "tradeCount": 0,
        #     "strikePrice": "45000",
        #     "exercisePrice": "59065.46085106"
        # },
        # {
        #     "symbol": "BTC-240927-75000-P",
        #     "priceChange": "0",
        #     "priceChangePercent": "0",
        #     "lastPrice": "16800",
        #     "lastQty": "1.6",
        #     "open": "16800",
        #     "high": "16800",
        #     "low": "16800",
        #     "volume": "1.67",
        #     "amount": "28056",
        #     "bidPrice": "15875",
        #     "askPrice": "0",
        #     "openTime": 1725266157207,
        #     "closeTime": 1725266157207,
        #     "firstTradeId": 416,
        #     "tradeCount": 1,
        #     "strikePrice": "75000",
        #     "exercisePrice": "59065.46085106"
        # }

            expiration = datetime.strptime(option["symbol"].split('-')[1], '%y%m%d')
            scenario_name = option["symbol"][:-2]
            option_type = option["symbol"][-1]
            stock_price = float(option["exercisePrice"])
            strike_price = int(option["strikePrice"])

            # if the ask_price or bid_price is None, skip the option
            if option["askPrice"] == None or option["bidPrice"] == None:
                continue

            if scenario_name in scenarios_unfiltered:
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
    # print(options_data)

#     Example options_data
#     options_data = [  {
#     "mid_price": "None",
#     "estimated_delivery_price": 58144.17,
#     "volume_usd": 0.0,
#     "quote_currency": "BTC",
#     "creation_timestamp": 1725213681632,
#     "base_currency": "BTC",
#     "underlying_index": "BTC-28MAR25",
#     "underlying_price": 61048.04,
#     "mark_iv": 73.48,
#     "interest_rate": 0.0,
#     "volume": 0.0,
#     "price_change": "None",
#     "mark_price": 1.63560187,
#     "open_interest": 2.2,
#     "ask_price": "None",
#     "bid_price": "None",
#     "instrument_name": "BTC-28MAR25-160000-P",
#     "last": 1.5695,
#     "low": "None",
#     "high": "None"
#   },
#   {
#     "mid_price": 0.22975,
#     "estimated_delivery_price": 58144.17,
#     "volume_usd": 1467.04,
#     "quote_currency": "BTC",
#     "creation_timestamp": 1725213681632,
#     "base_currency": "BTC",
#     "underlying_index": "BTC-25OCT24",
#     "underlying_price": 58748.35,
#     "mark_iv": 57.75,
#     "interest_rate": 0.0,
#     "volume": 0.1,
#     "price_change": 0.0,
#     "mark_price": 0.2302342,
#     "open_interest": 0.3,
#     "ask_price": 0.2325,
#     "bid_price": 0.227,
#     "instrument_name": "BTC-25OCT24-46000-C",
#     "last": 0.247,
#     "low": 0.247,
#     "high": 0.247
#   }]

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

        if scenario_name in scenarios_unfiltered:
            if option_type == 'C':
                scenarios_unfiltered[scenario_name][4] = option["ask_price"] * stock_price
                scenarios_unfiltered[scenario_name][5] = option["bid_price"] * stock_price
            else:
                scenarios_unfiltered[scenario_name][6] = option["ask_price"] * stock_price
                scenarios_unfiltered[scenario_name][7] = option["bid_price"] * stock_price
        else:
            # if expiration date is over 33 days from now, skip it
            # if (expiration - datetime.now()).days > 33:
                # continue

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


# def calculate_pcpa_return(data):
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
    # # sort results_pd by Return
    # results_pd = results_pd.sort_values(by='Return', ascending=False)

    # # add a return_per_day column and sort by that
    # results_pd['return_per_day'] = results_pd['Return'] / results_pd['Data'].apply(lambda x: x[3])
    # results_pd = results_pd.sort_values(by='return_per_day', ascending=False)

    # # add a return_delta_strike column, which is return / abs((strike price - stock price)) and sort by that
    # results_pd['return_delta_strike'] = results_pd['Return'] / abs(results_pd['Data'].apply(lambda x: x[1] - x[0]))
    # # results_pd = results_pd.sort_values(by='return_delta_strike', ascending=False)

    # add a retuns_pct column
    # if arb type is sold call, return / (s + p_a - c_b)
    # if arb type is sold put, return / (c_a + s - p_b)


    # add a annualized return column
    results_pd['APY (%)'] = ((1 + results_pd['returns_pct']) ** (365 / results_pd['Data'].apply(lambda x: x[3])) - 1) * 100
    results_pd = results_pd.sort_values(by='APY (%)', ascending=False)

    # pd.set_option('display.max_rows', None)
    print(f"\nPut-Call Arbitrage Opportunities on {exchange}:")
    print(results_pd)
    return results_pd
