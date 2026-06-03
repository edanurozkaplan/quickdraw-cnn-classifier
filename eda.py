import numpy as np
import matplotlib.pyplot as plt

flower = np.load("data/flower.npy")
cat = np.load("data/cat.npy")
dog = np.load("data/dog.npy")
house = np.load("data/house.npy")
castle = np.load("data/castle.npy")

print("Flower:", flower.shape)
print("Cat:", cat.shape)
print("Dog:", dog.shape)
print("House:", house.shape)
print("Castle:", castle.shape)

fig, axes = plt.subplots(1, 5, figsize=(15,3))

datasets = [
    (flower, "Flower"),
    (cat, "Cat"),
    (dog, "Dog"),
    (house, "House"),
    (castle, "Castle")
]

for ax, (data, title) in zip(axes, datasets):
    img = data[0].reshape(28,28)
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.show()