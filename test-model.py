import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from DeexposureNetModel import DeexposureNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the model
model = DeexposureNet().to(device)
model.load_state_dict(torch.load("saved_models/exposure_correction_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_patches(image, patch_size=128):
    width, height = image.size
    patches = []

    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)

    return patches

def test_image(image_path):
    img = Image.open(image_path).convert("RGB")
    patches = get_patches(img, patch_size=128)

    model_outputs = []

    with torch.no_grad():
        for patch in patches:
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            output_patch = model(patch_tensor)
            model_outputs.append(output_patch.cpu().squeeze(0))

    output_patches = [output_patch.permute(1, 2, 0).numpy() for output_patch in model_outputs]
    output_patches = [np.uint8(output_patch * 255) for output_patch in output_patches]

    patch_width = patches[0].size[0]
    patch_height = patches[0].size[1]

    num_patches_wide = img.width // patch_width
    num_patches_tall = img.height // patch_height

    stitched_image = np.zeros((img.height, img.width, 3), dtype=np.uint8)

    idx = 0
    for row in range(num_patches_tall):
        for col in range(num_patches_wide):
            patch_start_y = row * patch_height
            patch_start_x = col * patch_width
            stitched_image[patch_start_y:patch_start_y+patch_height, patch_start_x:patch_start_x+patch_width] = output_patches[idx]
            idx += 1

    stitched_image_pil = Image.fromarray(stitched_image)
    stitched_image_pil.show()

test_image("test-images/Bild020_Neg.Nr.22A.jpg")