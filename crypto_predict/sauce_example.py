import load
import tensorflow as tf


def build_model(rows):
    engine = load.grab_engine()
    try:
        engine.execute("DROP table public.loss_tracker")
    except:
        pass
    model_frame = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(rows, 5)),
            tf.keras.layers.Dense(125, activation=tf.nn.relu),
            tf.keras.layers.Dense(2),
        ]
    )

    model_frame.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    return model_frame
