import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch

from tensorflow.keras.models import load_model
from diffusers import AutoPipelineForImage2Image, LCMScheduler

# =========================

# CNN MODEL

# =========================

cnn_model = load_model("dropout03_model.keras")

classes = [
"flowers",
"animals",
"buildings"
]

# =========================

# INPUT IMAGE

# =========================

input_path = "input.png"

img_cnn = Image.open(input_path).convert("L")

img_cnn = ImageOps.invert(img_cnn)
img_cnn = img_cnn.resize((28, 28))
img_cnn = img_cnn.filter(ImageFilter.MaxFilter(5))
img_cnn = img_cnn.point(lambda p: 255 if p > 20 else 0)

img_cnn.save("processed_input.png")

x = np.array(img_cnn).astype("float32") / 255.0
x = x.reshape(1, 28, 28, 1)

prediction = cnn_model.predict(x)

predicted_class = classes[np.argmax(prediction)]

print("Probabilities:", prediction[0])
print("Prediction:", predicted_class)

# =========================

# PROMPT SELECTION

# =========================

FLOWER_PROMPT = """
beautiful colorful flower,
highly detailed petals,
digital art,
vibrant colors,
masterpiece,
professional illustration,
4k
"""

ANIMAL_PROMPT = """
cute animal,
highly detailed fur,
digital art,
vibrant colors,
masterpiece,
professional illustration,
4k
"""

BUILDING_PROMPT = """
beautiful building,
architectural illustration,
highly detailed,
digital art,
vibrant colors,
masterpiece,
4k
"""

if predicted_class == "flowers":
    prompt = FLOWER_PROMPT

elif predicted_class == "animals":
    prompt = ANIMAL_PROMPT

else:
    prompt = BUILDING_PROMPT

print("Selected prompt:", predicted_class)

# =========================

# DIFFUSION

# =========================

MODEL_ID = "Lykon/dreamshaper-7"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading diffusion model...")

pipe = AutoPipelineForImage2Image.from_pretrained(
MODEL_ID,
torch_dtype=DTYPE,
variant="fp16" if DEVICE == "cuda" else None,
)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LCM_LORA_ID)
pipe.fuse_lora()
pipe = pipe.to(DEVICE)

print("Diffusion model ready.")

img_diffusion = Image.open(input_path).convert("RGB")
img_diffusion = ImageOps.contain(img_diffusion, (512, 512))

result = pipe(
prompt=prompt,
image=img_diffusion,
strength=0.75,
guidance_scale=7.5,
num_inference_steps=25
)

result.images[0].save("output.png")

print("Done -> output.png")
