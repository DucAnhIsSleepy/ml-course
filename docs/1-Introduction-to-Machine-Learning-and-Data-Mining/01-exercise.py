#create dataset
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("beans", split = "train[:100]")

df = dataset.to_pandas()
df.head()
df.info()

#show an image example
from PIL import Image
import matplotlib.pyplot as plt

example = dataset[0]
image = example["image"]
label = example["labels"]

plt.imshow(image)
plt.title(label)
plt.axis("off")
plt.show()