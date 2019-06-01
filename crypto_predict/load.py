import random
import time as t

import numpy as np
import pandas as pd
import sqlalchemy
from binance.client import Client
from tqdm import tqdm

from crypto_predict import config

api_key = config.retrieve_config_value("Binance", "api_key")
secret_key = config.retrieve_config_value("Binance", "secret_key")
client = Client(api_key, secret_key)


def grab_engine():
    engine = sqlalchemy.create_engine("postgresql://localhost:5432/postgres")
    engine.connect()
    return engine


def klines_to_df(kline_list):
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
    return df


def save_klines(kline_list):
    engine = grab_engine()
    df = klines_to_df(kline_list)
    df.to_sql(name="binance_klines", con=engine, if_exists="append")
    return df


def wipe_table():
    clearer = grab_engine()
    clearer.execute("DELETE FROM binance_klines WHERE open > 0")


def load_training_data(train_tables, test_tables, predict_time, rows_per_table):
    engine = grab_engine()
    rows = pd.read_sql(
        sql="SELECT COUNT(DISTINCT open_time) FROM binance_klines", con=engine
    )["count"][0]
    if train_tables + test_tables + predict_time + rows_per_table >= rows:
        print("Your database does not have enough entries.")
        exit()

    # Read and normalize dataset
    df = pd.read_sql(
        sql="SELECT DISTINCT open_time, open, high, low, close, volume FROM binance_klines ORDER BY open_time DESC",
        con=engine,
    )
    df = df.drop("open_time", axis=1)

    # mults = {col: df[col].max() for col in df.columns}
    # for col in df.columns:
    #     df[col] = df[col].apply(lambda x: x / mults[col])

    x_train = []
    x_test = []

    max_offset = rows - (
            train_tables + rows_per_table + test_tables + rows_per_table + predict_time
    )
    offset = random.randint(0, max_offset)

    print("Loading Training Data")
    t.sleep(0.1)
    for i in tqdm(range(offset, offset + train_tables)):
        x_train.append(df[i: i + rows_per_table].values)
    x_train = np.array(x_train)
    t.sleep(0.1)
    sub_df = df[
             offset
             + rows_per_table
             + predict_time: offset
                             + rows_per_table
                             + predict_time
                             + train_tables
             ]
    sub_df = sub_df.drop(["volume", "open", "close"], axis=1)
    y_train = sub_df.values
    print("Loading Testing Data")
    t.sleep(0.1)
    for i in tqdm(range(offset, offset + test_tables)):
        x_test.append(df[i + train_tables: i + train_tables + rows_per_table].values)
    x_test = np.array(x_test)
    t.sleep(0.1)
    sub_df = df[
             offset
             + rows_per_table
             + predict_time
             + train_tables: offset
                             + rows_per_table
                             + predict_time
                             + test_tables
                             + train_tables
             ]
    sub_df = sub_df.drop(["volume", "open", "close"], axis=1)
    y_test = sub_df.values
    # print(f"X Training Data Structure: ({x_train.shape})")
    # print(f"Y Training Data Structure: ({y_train.shape})")
    # print(f"X Testing  Data Structure: ({x_test.shape})")
    # print(f"Y Testing  Data Structure: ({y_test.shape})")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    wipe_table()
    days_ago = 0
    while True:
        days_ago += 1
        try:
            klines = client.get_historical_klines(
                "BNBBTC",
                Client.KLINE_INTERVAL_1MINUTE,
                f"{days_ago} day ago UTC",
                f"{days_ago - 1} day ago UTC",
            )
            dframe = save_klines(klines)
            if len(dframe) == 0:
                break
            print(f"Saved data for {days_ago} days ago")
        except:
            print(f"Timeout error, repeating day {days_ago}")
            days_ago -= 1
