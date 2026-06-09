import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.models import load_model

model = load_model("dropout03_model.keras")

classes = [
    "flowers",
    "animals",
    "buildings"
]

img = Image.open("input3.png").convert("L")

# QuickDraw formatına yaklaştır
img = ImageOps.invert(img)

# 28x28
img = img.resize((28, 28))

# çizgileri kalınlaştır
img = img.filter(ImageFilter.MaxFilter(5))

# siyah-beyaz threshold
img = img.point(lambda p: 255 if p > 20 else 0)

# kontrol için kaydet
img.save("processed_input.png")

img = np.array(img).astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)

prediction = model.predict(img)

print("Probabilities:", prediction[0])

predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)