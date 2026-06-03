import numpy as np

SAMPLES_PER_CLASS = 10000

flower = np.load("data/flower.npy")[:SAMPLES_PER_CLASS]

cat = np.load("data/cat.npy")[:5000]
dog = np.load("data/dog.npy")[:5000]

house = np.load("data/house.npy")[:5000]
castle = np.load("data/castle.npy")[:5000]

animal = np.concatenate([cat, dog], axis=0)
building = np.concatenate([house, castle], axis=0)

X = np.concatenate([
    flower,
    animal,
    building
])

y = np.concatenate([
    np.zeros(len(flower)),
    np.ones(len(animal)),
    np.full(len(building), 2)
])

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset saved.")