import matplotlib.pyplot as plt
import pandas as pd

from crypto_predict import load

engine = load.grab_engine()

df = pd.read_sql(sql="SELECT * FROM public.loss_tracker", con=engine)

x = df["index"]
y = df["loss"]
plt.scatter(x, y, c="r", alpha=1, marker=".", label="Mean Squarred Error")
plt.xlabel("Epoch (including tests)")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.yscale("log")
plt.show()
