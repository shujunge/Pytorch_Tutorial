from PIL import Image
from torchvision import transforms
import torch

img_path = "./figs/test_image.jpg"


img = Image.open(img_path)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
images = transform(img)
print(images.size())