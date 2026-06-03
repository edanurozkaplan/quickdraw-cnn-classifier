import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

# normalize
X = X.astype(np.float32) / 255.0

# CNN formatı
X = X.reshape(-1, 28, 28, 1)

# karıştır
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# %70 train
train_size = int(len(X) * 0.70)

# %15 validation
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Dataset split completed.")