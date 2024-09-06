from pycoingecko import CoinGeckoAPI


def get_top_coins(top_n=30):
    cg = CoinGeckoAPI()

    # Get top 50 coins by market cap
    top_50_coins = cg.get_coins_markets(
        vs_currency="usd", order="market_cap_desc", per_page=50, page=1
    )

    # Remove stablecoins by checking if the price is within 10% of $1
    filtered_coins = [
        coin for coin in top_50_coins if not (0.9 <= coin["current_price"] <= 1.1)
    ]

    # Sort filtered coins by trading volume (top 30)
    assert top_n <= len(
        filtered_coins
    ), f"top_n must be less than or equal to {len(filtered_coins)}"
    sorted_by_volume = sorted(
        filtered_coins, key=lambda x: x["total_volume"], reverse=True
    )[:top_n]

    # for coin in sorted_by_volume:
    #     print(f"Name: {coin['name']}, Symbol: {coin['symbol']}, Market Cap Rank: {coin['market_cap_rank']}, Volume: {coin['total_volume']}")

    return sorted_by_volume
