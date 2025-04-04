import torch
from PIL import Image
from torchvision import transforms
from DeexposureNetModel.model import DeexposureNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeexposureNet().to(device)
model.load_state_dict(torch.load("saved_models/exposure_correction_model.pth"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

def test_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    output_image = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_image = (output_image * 255).astype("uint8")
    output_img = Image.fromarray(output_image)
    output_img.show()

test_image("path/to/your/image.jpg")
