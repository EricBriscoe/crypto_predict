import sqlalchemy
from binance.client import Client
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
from sklearn import preprocessing
import time as t

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


def load_training_data(
    train_tables, test_tables, predict_time, rows_per_table, predict_param
):
    engine = grab_engine()
    rows = pd.read_sql(sql="SELECT COUNT(open_time) FROM binance_klines", con=engine)[
        "count"
    ][0]
    if train_tables + test_tables + predict_time + rows_per_table >= rows:
        print("Your database does not have enough entries.")
        exit()

    df = pd.read_sql(
        sql="SELECT open, high, low, close, volume FROM binance_klines", con=engine
    )
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=["open", "high", "low", "close", "volume"])

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    print("Loading X Training Data")
    t.sleep(.1)
    for i in tqdm(range(train_tables)):
        x_train.append(df[i : i + rows_per_table].values)
    t.sleep(0.1)
    print("Done")
    print("Loading Y Training Data")
    sub_df = df[
        rows_per_table + predict_time : rows_per_table + predict_time + train_tables
    ]
    y_train = sub_df[predict_param].values
    print("Done")
    print("Loading X Testing Data")
    t.sleep(.1)
    for i in tqdm(range(test_tables)):
        x_test.append(df[i+train_tables:i+train_tables+rows_per_table].values)
    t.sleep(.1)
    print("Done")
    print("Loading Y Testing Data")
    sub_df = df[
        rows_per_table + predict_time + train_tables : rows_per_table + predict_time + test_tables + train_tables
    ]
    y_test = sub_df[predict_param].values
    print("Done")

    print(f"X Training Data Structure: ({len(x_train)},{len(x_train[0])},{len(x_train[0][0])})")
    print(f"Y Training Data Structure: ({len(y_train)})")
    print(f"X Testing  Data Structure: ({len(x_test)},{len(x_test[0])},{len(x_test[0][0])})")
    print(f"Y Testing  Data Structure: ({len(y_test)})")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # days_ago = 100
    # klines = client.get_historical_klines(
    #     "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{days_ago} day ago UTC"
    #     )
    # save_klines(klines)
    # print(f"Saved data for {days_ago} days ago")
    # load_training_data(
    #     train_tables=10000,
    #     test_tables=2000,
    #     predict_time=1,
    #     rows_per_table=100,
    #     predict_param="high",
    # )
    pass
