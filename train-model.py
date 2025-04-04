import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from DeexposureNetModel.model import DeexposureNet
from DeexposureNetModel.dataset import DeexposureNetDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
])

original_dir = "data/original"
overexposed_dir = "data/overexposed"

dataset = DeexposureNetDataset(original_dir, overexposed_dir, patch_size=128, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = DeexposureNet().to(device)

mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for over_patch, clean_patch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        over_patch = over_patch.to(device)
        clean_patch = clean_patch.to(device)

        optimizer.zero_grad()
        output = model(over_patch)

        loss_mse = mse_loss(output, clean_patch)
        loss = loss_mse

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/DeexposureNet.pth")
print("Model saved to saved_models/DeexposureNet.pth")
