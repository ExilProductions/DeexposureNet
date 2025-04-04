import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeexposureNetDataset(Dataset):
    def __init__(self, original_dir, overexposed_dir, patch_size=128, transform=None):
        self.original_images = sorted(os.listdir(original_dir))
        self.original_dir = original_dir
        self.overexposed_dir = overexposed_dir
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.original_images)

    def _get_random_patch(self, image):
        width, height = image.size
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        return image.crop((left, top, left + self.patch_size, top + self.patch_size))

    def __getitem__(self, idx):
        orig_path = os.path.join(self.original_dir, self.original_images[idx])
        over_path = os.path.join(self.overexposed_dir, self.original_images[idx])

        orig = Image.open(orig_path).convert("RGB")
        over = Image.open(over_path).convert("RGB")

        orig_patch = self._get_random_patch(orig)
        over_patch = self._get_random_patch(over)

        if self.transform:
            orig_patch = self.transform(orig_patch)
            over_patch = self.transform(over_patch)

        return over_patch, orig_patch
