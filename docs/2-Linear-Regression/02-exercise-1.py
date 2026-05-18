import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LinearRegression

dataset = load_dataset("krishal07/student-performance", split = "train")

df = dataset.to_pandas()
df.head()
df.info()

#plotting the data
reg = LinearRegression()
reg.fit(df[["StudyHours"]], df["TestScore_Math"])
reg.score(df[["StudyHours"]], df["TestScore_Math"])
reg.coef_, reg.intercept_

plt.scatter(df["StudyHours"], df["TestScore_Math"])
plt.plot(df["StudyHours"], reg.predict(df[["StudyHours"]]), color='red')
plt.xlabel("Study Time (hours)")
plt.ylabel("Final Grade")
plt.title("Study Time vs Final Grade")
plt.grid()

plt.legend()

plt.show()