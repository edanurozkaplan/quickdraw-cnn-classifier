import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("baseline_model.keras")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print("\nTEST RESULTS")
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)