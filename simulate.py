import os

import pandas as pd
from binance.client import Client

api_key = os.environ["api_key"]
secret_key = os.environ["secret_key"]
client = Client(api_key, secret_key)


class Wallet:
    def __init__(self):
        pass


def retrieve_x(rows):
    hours = int(rows / 60) + 1

    klines = client.get_historical_klines(
        "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{hours} hours ago UTC"
    )

    formatted_klines = []
    for kline in klines:
        kline.pop(-1)
        formatted_klines.append(list(map(float, kline)))

    df = pd.DataFrame(
        formatted_klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
    )
    df = df.drop(
        labels=[
            "open_time",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        axis=1,
    )

    mults = {col: df[col].max() for col in df.columns}
    df_scaled = pd.DataFrame()
    for col in df.columns:
        df_scaled[col] = df[col].apply(lambda x: x / mults[col])

    # x = df.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df_scaled = pd.DataFrame(x_scaled, columns=["open", "high", "low", "close", "volume"])
    # mults = {col:df[col][0]/df_scaled[col][0] for col in df_scaled.columns}

    print(df_scaled['high'].max())
    print(df_scaled['high'].min())
    print(mults)

    for index, row in df.iterrows():
        assert (row['open'] == df_scaled['open'][index] * mults['open'])
    return (df_scaled, mults)


def trade():
    # Load model
    # Load input
    # Normalize Input
    #
    pass


if __name__ == "__main__":
    retrieve_x(100)
