import datetime
import time

import numpy
import pandas as pd
import tensorflow as tf
from binance.client import Client
from binance.websockets import BinanceSocketManager

from crypto_predict import config
from crypto_predict import load

api_key = config.retrieve_config_value("Binance", "api_key")
secret_key = config.retrieve_config_value("Binance", "secret_key")
client = Client(api_key, secret_key)
engine = load.grab_engine()
model = tf.keras.models.load_model("scratch_model.md5")


def set_bounds():
    rows = model.layers[0].output_shape[1] / 5
    # Read last state from table
    df = pd.read_sql("SELECT * FROM sim_wallet_log", con=engine)
    # Retrieve previous Klines
    klines = client.get_historical_klines(
        "BNBBTC",
        Client.KLINE_INTERVAL_1MINUTE,
        # f"0 day ago UTC",
        f"{int(rows / 60 / 24) + 1} day ago UTC",
    )
    klines = load.klines_to_df(klines)
    klines = klines.reset_index()
    klines = klines.sort_values(by=["close_time"], axis=0, ascending=False)
    while len(klines) > rows:
        klines = klines.drop(klines.index[0])
    klines = klines.drop(
        [
            "open_time",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        axis=1,
    )
    print(klines.columns)
    klines = klines.values
    klines = numpy.array([klines])
    print(klines)
    # Predict next Klines
    output = model.predict(klines)
    output = output[0]
    print(output)
    # update datatframe and write to database


def process_trade(msg):
    print(msg)
    # Price in BTC per BNB
    price = msg["p"]
    # Read most recent state from table
    df = pd.read_sql(
        sql="SELECT * FROM public.sim_wallet_log ORDER BY timestamp DESC LIMIT 1",
        con=engine,
    )
    print(df)
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
    # Start processing trades
    bm = BinanceSocketManager(client)
    bm.start_aggtrade_socket("BNBBTC", process_trade)
    bm.start()
    time.sleep(10)
    bm.close()


def initialize():
    df = pd.DataFrame(
        data={
            "timestamp": datetime.datetime.now(),
            "bnb_bal": [100],
            "btc_bal": [100],
            "low_trade": [0],
            "btc_sell_vol": [0],
            "high_trade": [0],
            "bnb_sell_vol": [0],
        }
    )
    df.to_sql(name="sim_wallet_log", schema="public", con=engine, if_exists="replace")


if __name__ == "__main__":
    initialize()
    set_bounds()
    exit()
    sim_trade()
    exit()
    while True:
        set_bounds()
        time.sleep(60)