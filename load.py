import sqlalchemy
from binance.client import Client
import pandas as pd
import os
from tqdm import tqdm
import numpy
import pickle
import tensorflow as tf

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

    # print("Loading X Training Data")
    # for i in tqdm(range(rows-(input_rows+future_predict_time))):
    #     array = pd.read_sql(sql=f"SELECT * FROM binance_klines LIMIT {input_rows} OFFSET {i}", con=engine).values
    #     temp_list.append(array)
    # x_train = numpy.array(temp_list)
    # temp_list = []

    # print("Loading Y Training Data")
    # for i in tqdm(range(rows-(input_rows+future_predict_time))):
    #     array = pd.read_sql(sql=f"SELECT * FROM binance_klines LIMIT 1 OFFSET {i+input_rows+future_predict_time}", con=engine).values
    #     temp_list.append(array[0])
    # y_train = numpy.array(temp_list)
    y_train = pd.read_sql(sql=f"SELECT * FROM binance_klines OFFSET {input_rows+future_predict_time}", con=engine).values
    print(y_train.shape)
    temp_list = []

    # pickle.dump(x_train, open(os.path.join(os.getcwd(), 'x_train.pickle'), 'wb'))
    pickle.dump(y_train, open(os.path.join(os.getcwd(), 'y_train.pickle'), 'wb'))


def train_model():
    x_train = pickle.load(open(os.path.join(os.getcwd(), 'x_train.pickle'), 'rb'))
    y_train = pickle.load(open(os.path.join(os.getcwd(), 'y_train.pickle'), 'rb'))
    print(x_train.shape)
    print(y_train.shape)
    print(max([y[2] for y in y_train]))
    print(min([y[2] for y in y_train]))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(100, 11)),
        tf.keras.layers.Dense(2000, activation=tf.nn.relu),
        tf.keras.layers.Dense(50)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model.fit(x_train, [int(y[2]*10000) for y in y_train], epochs=5)


if __name__ == "__main__":
    # days_ago = 100
    # klines = client.get_historical_klines(
    #     "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, f"{days_ago} day ago UTC"
    #     )
    # save_klines(klines)
    # print(f"Saved data for {days_ago} days ago")
    # load_training_data()
    train_model()

