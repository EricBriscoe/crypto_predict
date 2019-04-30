import sqlalchemy
from binance.client import Client
import pandas as pd
import os
from tqdm import tqdm
import numpy
import pickle

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
    df.set_index(keys='open_time', inplace=True)
    print(df)
    df.to_sql(name="binance_klines", con=engine, if_exists="replace")


def load_training_data():
    engine = grab_engine()
    rows = pd.read_sql(sql="SELECT COUNT(open_time) FROM binance_klines", con=engine)['count'][0]
    input_rows = 100
    temp_list = []
    # how many minutes in the future to
    future_predict_time = 1

    print("Loading X Training Data")
    for i in tqdm(range(rows-(input_rows+future_predict_time))):
        array = pd.read_sql(sql=f"SELECT * FROM binance_klines LIMIT {input_rows} OFFSET {i}", con=engine).values
        temp_list.append(array)
    x_train = numpy.array(temp_list)
    temp_list = []

    print("Loading Y Training Data")
    for i in tqdm(range(rows-(input_rows+future_predict_time))):
        array = pd.read_sql(sql=f"SELECT * FROM binance_klines LIMIT 1 OFFSET {i+input_rows+future_predict_time}", con=engine).values
        temp_list.append(array)
    y_train = numpy.array(temp_list)
    temp_list = []

    pickle.dump(x_train, open(os.path.join(os.getcwd(), 'x_train.pickle'), 'wb'))
    pickle.dump(y_train, open(os.path.join(os.getcwd(), 'y_train.pickle'), 'wb'))





if __name__ == "__main__":
    # days_ago = 100
    # klines = client.get_historical_klines(
    #     "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{days_ago} day ago UTC"
    #     )
    # save_klines(klines)
    # print(f"Saved data for {days_ago} days ago")
    load_training_data()

