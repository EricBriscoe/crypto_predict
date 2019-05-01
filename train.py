import tensorflow as tf

import load


def train_model():
    ((x_train, y_train), (x_test, y_test)) = load.load_training_data(
        train_tables=200000, test_tables=80000, predict_time=100, rows_per_table=100
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(100, 5)),
            tf.keras.layers.Dense(500),
            tf.keras.layers.Dense(250),
            tf.keras.layers.Dense(250),
            tf.keras.layers.Dense(5),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    model.fit(x_train, y_train, epochs=300)

    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    train_model()
