import datetime
import time

import numpy
import pandas as pd
import tensorflow as tf
from binance.client import Client
from binance.websockets import BinanceSocketManager
from sqlalchemy.dialects.postgresql import TIMESTAMP, FLOAT

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
            "btc_bal": [1],
            "low_trade": [0],
            "low_trade_age": [0],
            "btc_sell_vol": [0],
            "high_trade": [0],
            "high_trade_age": [0],
            "bnb_sell_vol": [0],
        }
    )
    df.to_sql(
        name="sim_wallet_log",
        schema="public",
        con=engine,
        if_exists="replace",
        dtype={
            "timestamp": TIMESTAMP,
            "bnb_bal": FLOAT,
            "btc_bal": FLOAT,
            "low_trade": FLOAT,
            "low_trade_age": FLOAT,
            "btc_sell_vol": FLOAT,
            "high_trade": FLOAT,
            "high_trade_age": FLOAT,
            "bnb_sell_vol": FLOAT,
        },
    )


def ml_trade():
    initialize()
    set_bounds()
    exit()
    sim_trade()
    exit()
    while True:
        set_bounds()
        time.sleep(60)


def test_manual_strategy():
    initialize()
    history_df = pd.read_sql(
        sql="SELECT DISTINCT open_time, open, high, low, close, volume FROM binance_klines ORDER BY open_time DESC",
        con=engine,
    )
    wallet_log_df = pd.read_sql(sql="SELECT * FROM sim_wallet_log", con=engine)
    wallet_current_df = wallet_log_df
    trade_amount = 0.75
    price_range_mult = 0.2

    for index, row in history_df.iterrows():
        if index < 10:
            continue
        # Look at wallet_current_df and see if either limit order is satisfied by the current set ranges
        high = row["high"]
        low = row["low"]
        price_range = high - low
        bnb_bal = wallet_current_df["bnb_bal"][0]
        btc_bal = wallet_current_df["btc_bal"][0]
        low_trade = wallet_current_df["low_trade"][0]
        low_trade_age = wallet_current_df["low_trade_age"][0]
        btc_sell_vol = wallet_current_df["btc_sell_vol"][0]
        high_trade = wallet_current_df["high_trade"][0]
        high_trade_age = wallet_current_df["high_trade_age"][0]
        bnb_sell_vol = wallet_current_df["bnb_sell_vol"][0]

        # TODO add binance fees
        if high_trade < high and bnb_sell_vol != 0:
            btc_bal += bnb_sell_vol * high_trade
            bnb_sell_vol = 0
        else:
            high_trade_age += 1

        if low_trade > low and btc_sell_vol != 0:
            bnb_bal += btc_sell_vol / low_trade
            btc_sell_vol = 0
        else:
            low_trade_age += 1
        wallet_current_df["timestamp"] = datetime.datetime.now()

        # Set new limit orders if conditions are met
        if bnb_sell_vol == 0 or high_trade_age == 60:
            high_trade = high + (price_range * price_range_mult)
            bnb_sell_vol += trade_amount * bnb_bal
            bnb_bal -= trade_amount * bnb_bal
            high_trade_age = 0

        if btc_sell_vol == 0 or low_trade_age == 60:
            low_trade = low - (price_range * price_range_mult)
            btc_sell_vol += trade_amount * btc_bal
            btc_bal -= trade_amount * btc_bal
            low_trade_age = 0

        # Update wallet log
        wallet_current_df["bnb_bal"] = [bnb_bal]
        wallet_current_df["btc_bal"] = [btc_bal]
        wallet_current_df["low_trade"] = [low_trade]
        wallet_current_df["low_trade_age"] = [low_trade_age]
        wallet_current_df["btc_sell_vol"] = [btc_sell_vol]
        wallet_current_df["high_trade"] = [high_trade]
        wallet_current_df["high_trade_age"] = [high_trade_age]
        wallet_current_df["bnb_sell_vol"] = [bnb_sell_vol]
        wallet_log_df = wallet_current_df.append(wallet_log_df, ignore_index=True)
        if index % 1440 == 0:
            wallet_log_df.to_sql(name="sim_wallet_log", con=engine, if_exists="replace")
            print(
                f"Wallet Value in BTC: {wallet_current_df['btc_bal'][0] + wallet_current_df['btc_sell_vol'][0] + (
                            wallet_current_df['bnb_bal'][0] + wallet_current_df['bnb_sell_vol'][0]) * high}"
            )
    input("Press enter to continue")
    pass


if __name__ == "__main__":
    test_manual_strategy()
