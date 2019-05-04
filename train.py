import tensorflow as tf

import load

hours_of_data = 12
rows = hours_of_data * 60


def build_model():
    model_frame = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(rows, 5)),
            tf.keras.layers.Dense(500),
            tf.keras.layers.Dense(250),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(4),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop()
    model_frame.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    return model_frame


def train_model(training_model, epochs):
    ((x_train, y_train), (x_test, y_test)) = load.load_training_data(
        train_tables=150000, test_tables=20000, predict_time=1, rows_per_table=rows
    )

    training_model.fit(x_train, y_train, epochs=epochs)
    training_model.evaluate(x_test, y_test)

    return training_model


if __name__ == "__main__":
    # model = build_model()
    model = tf.keras.models.load_model("scratch_model.md5")
    model = train_model(model, 2)
    model.save("scratch_model.md5")
