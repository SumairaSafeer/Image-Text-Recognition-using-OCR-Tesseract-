"""
Generates a sample test image containing printed text, with a bit of
noise/blur added, so ocr_recognition.py has something realistic to
read on a first run. Run this once before ocr_recognition.py if you
don't have your own image (e.g. sample_input.png) yet.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

W, H = 900, 300
img = Image.new("RGB", (W, H), color=(235, 235, 230))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
except Exception:
    font = ImageFont.load_default()

draw.text((40, 100), "DecodeLabs Project 4", fill=(20, 20, 20), font=font)

# Add slight blur + noise to simulate a real photographed / scanned document
img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
arr = np.array(img).astype(np.int16)
noise = np.random.randint(-12, 12, arr.shape)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
Image.fromarray(arr).save("sample_input.png")
print("Created sample_input.png")
