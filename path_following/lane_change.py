import pandas as pd
from matplotlib import pyplot as plt

g = pd.read_csv("lane_change.csv")
a = g["x"]
b = g["y"]
plt.plot(a,b)
plt.savefig("path.png")

