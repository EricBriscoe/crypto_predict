import os
import time

import pandas as pd
from binance.client import Client
from binance.websockets import BinanceSocketManager
import tensorflow as tf

api_key = os.environ["api_key"]
secret_key = os.environ["secret_key"]
client = Client(api_key, secret_key)


def set_bounds():
    rows = model.layers[0].output_shape[1]
    print(rows)
    # Retrieve previous Klines
    # Predict next Klines
    #


def process_trade(msg):
    print(msg)
    # Read most recent state from table
    # Determine if trade crossed high or low val
    # Subtract from the relevant high or low amt if so
    # Apply Fees
    # Add new row to table with time incremented by 1


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

    print(df_scaled["high"].max())
    print(df_scaled["high"].min())
    print(mults)

    for index, row in df.iterrows():
        assert row["open"] == df_scaled["open"][index] * mults["open"]
    return df_scaled.values, mults


def sim_trade():
    # Initialize wallet
    # Start processing trades
    bm = BinanceSocketManager(client)
    bm.start_aggtrade_socket("BNBBTC", process_trade)
    bm.start()


if __name__ == "__main__":
    model = tf.keras.models.load_model("scratch_model.md5")
    sim_trade()
    while True:
        set_bounds()
        time.sleep(60)
