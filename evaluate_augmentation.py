import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("augmentation_model.keras")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)