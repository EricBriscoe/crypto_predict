import pandas as pd
import tensorflow as tf

from crypto_predict import load

try:
    from crypto_predict import sauce
except ImportError:
    print("Please create your own model by copying sauce_example.py to sauce.py.")
    input("Press enter to continue using sauce_example or ctrl+c to quit.")
    from crypto_predict import sauce_example as sauce

hours_of_data = 1
rows = hours_of_data * 60


# noinspection PyBroadException
def train_model(training_model, epochs):
    # 500000 total tables will take around 16 gigs of ram alone
    ((x_train, y_train), (x_test, y_test)) = load.load_training_data(
        train_tables=100000, test_tables=100000, predict_time=1, rows_per_table=rows
    )
    print("Example Data:")
    print(x_train[0])
    print(y_train[0])
    results = training_model.fit(x_train, y_train, epochs=epochs)
    print(f"Results:\n{results.history}")
    results = pd.DataFrame(data={"loss": results.history["loss"]})
    try:
        log_df = pd.read_sql(
            sql="SELECT loss FROM public.loss_tracker", con=load.grab_engine()
        )
        if len(log_df) > 0:
            results = log_df.append(results, ignore_index=True)
    except:
        pass
    results.to_sql(name="loss_tracker", con=load.grab_engine(), if_exists="replace")
    results = training_model.evaluate(x_test, y_test)
    print(f"Results:\n{results}")
    results = pd.DataFrame(data={"loss": [results[0]]})
    try:
        log_df = pd.read_sql(
            sql="SELECT loss FROM public.loss_tracker", con=load.grab_engine()
        )
        if len(log_df) > 0:
            results = log_df.append(results, ignore_index=True)
    except:
        pass
    results.to_sql(name="loss_tracker", con=load.grab_engine(), if_exists="replace")
    return training_model


if __name__ == "__main__":
    model = tf.keras.models.load_model("scratch_model.md5")
    # model = sauce.build_model(rows)
    while True:
        model = train_model(model, 9)
        model.save("scratch_model.md5")
