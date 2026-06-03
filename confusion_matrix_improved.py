import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("improved_model.keras")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

cm = tf.math.confusion_matrix(
    y_test,
    y_pred,
    num_classes=3
)

print(cm.numpy())