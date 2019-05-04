import os
import time as t
import random

import numpy as np
import pandas as pd
import sqlalchemy
from binance.client import Client
from tqdm import tqdm

api_key = os.environ["api_key"]
secret_key = os.environ["secret_key"]
client = Client(api_key, secret_key)


def grab_engine():
    engine = sqlalchemy.create_engine("postgresql://localhost:5432/postgres")
    engine.connect()
    return engine


def save_klines(kline_list):
    engine = grab_engine()
    formatted_klines = []
    for kline in kline_list:
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
    df.set_index(keys="open_time", inplace=True)
    print(df)
    df.to_sql(name="binance_klines", con=engine, if_exists="replace")


def load_training_data(train_tables, test_tables, predict_time, rows_per_table):
    engine = grab_engine()
    rows = pd.read_sql(sql="SELECT COUNT(open_time) FROM binance_klines", con=engine)[
        "count"
    ][0]
    if train_tables + test_tables + predict_time + rows_per_table >= rows:
        print("Your database does not have enough entries.")
        exit()

    # Read and normalize dataset
    df = pd.read_sql(
        sql="SELECT open, high, low, close, volume FROM binance_klines", con=engine
    )

    mults = {col: df[col].max() for col in df.columns}
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x / mults[col])

    x_train = []
    x_test = []

    max_offset = rows - (
        train_tables + rows_per_table + test_tables + rows_per_table + predict_time
    )
    offset = random.randint(0, max_offset)

    print("Loading X Training Data")
    t.sleep(0.1)
    for i in tqdm(range(train_tables)):
        x_train.append(df[i: i + rows_per_table].values)
    x_train = np.array(x_train)
    t.sleep(0.1)
    print("Loading Y Training Data")
    sub_df = df[
             rows_per_table + predict_time: rows_per_table + predict_time + train_tables
             ]
    sub_df = sub_df.drop("volume", axis=1)
    y_train = sub_df.values
    print("Loading X Testing Data")
    t.sleep(0.1)
    for i in tqdm(range(test_tables)):
        x_test.append(df[i + train_tables: i + train_tables + rows_per_table].values)
    x_test = np.array(x_test)
    t.sleep(0.1)
    print("Loading Y Testing Data")
    sub_df = df[
             rows_per_table
             + predict_time
             + train_tables: rows_per_table
                             + predict_time
                             + test_tables
                             + train_tables
             ]
    sub_df = sub_df.drop("volume", axis=1)
    y_test = sub_df.values
    print(f"X Training Data Structure: ({x_train.shape})")
    print(f"Y Training Data Structure: ({y_train.shape})")
    print(f"X Testing  Data Structure: ({x_test.shape})")
    print(f"Y Testing  Data Structure: ({y_test.shape})")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    days_ago = 100
    while True:
        klines = client.get_historical_klines(
            "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{days_ago} day ago UTC"
        )
        if len(klines == 0):
            break
        save_klines(klines)
        print(f"Saved data for {days_ago} days ago")
