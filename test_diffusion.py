from PIL import Image, ImageOps
import torch

from diffusers import AutoPipelineForImage2Image, LCMScheduler

MODEL_ID = "Lykon/dreamshaper-7"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading model...")

pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16" if DEVICE == "cuda" else None,
)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LCM_LORA_ID)
pipe.fuse_lora()
pipe = pipe.to(DEVICE)

print("Model ready.")

img = Image.open("input.png").convert("RGB")

img = ImageOps.contain(img, (512, 512))

prompt = """
beautiful colorful flower,
highly detailed petals,
digital art,
vibrant colors,
masterpiece,
professional illustration,
4k
"""

result = pipe(
    prompt=prompt,
    image=img,
    strength=0.75,
    guidance_scale=7.5,
    num_inference_steps=25
)

result.images[0].save("output.png")

print("Done -> output.png")