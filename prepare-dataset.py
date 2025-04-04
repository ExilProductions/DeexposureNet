import os
from PIL import Image, ImageEnhance
from tqdm import tqdm

def simulate_overexposure(input_path, output_path, brightness_factor=1.8, saturation_factor=1.3):
    img = Image.open(input_path).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    img.save(output_path)

input_folder = "data/original"
output_folder = "data/overexposed"
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        simulate_overexposure(
            os.path.join(input_folder, filename),
            os.path.join(output_folder, filename)
        )
