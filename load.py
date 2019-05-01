import sqlalchemy
from binance.client import Client
import pandas as pd
import os
from tqdm import tqdm
import numpy
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


def load_training_data(x_train_tables, x_test_tables, predict_time, rows_per_table):
    engine = grab_engine()
    rows = pd.read_sql(sql="SELECT COUNT(open_time) FROM binance_klines", con=engine)[
        "count"
    ][0]
    if x_train_tables + x_test_tables + predict_time + rows_per_table >= rows:
        print("Your database does not have enough entries.")
        exit()

    df = pd.read_sql(sql="SELECT open, high, low, close, volume FROM binance_klines", con=engine)
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=['open', 'high', 'low', 'close', 'volume'])

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    print("Loading X Training Data")
    for i in tqdm(range(x_train_tables)):
        x_train.append(df[i:i+rows_per_table])
    t.sleep(.1)
    print("Done")
    print("Loading Y Training Data")
    for i in tqdm(range(y_train_tables)):
        pass



if __name__ == "__main__":
    # days_ago = 100
    # klines = client.get_historical_klines(
    #     "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{days_ago} day ago UTC"
    #     )
    # save_klines(klines)
    # print(f"Saved data for {days_ago} days ago")
    load_training_data(x_train_tables=100000, x_test_tables=20000, predict_time=1, rows_per_table=100)
