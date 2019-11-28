from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
import shutil
from tqdm import tqdm
from pytorch_seg import deeplabv3_resnet101


class Segmentation(object):
    def __init__(self, root, mode=None,shuffle=True, transform=None):
        super(Segmentation, self).__init__()

        self.images = glob(root + "/*")

        self.mode = mode
        self.transform = transform
        self.shuffle = shuffle
        self.num_class = 5

    def __getitem__(self, index):

        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images[index]

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("0","1","2","3","4")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ##判断是否有gpu

# dataset and dataloader
train_img_path = "/data/LSUN17/train/images"
val_img_path = "/data/LSUN17/val/images"

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_kwargs = {'transform': input_transform}
train_dataset = Segmentation(train_img_path, mode='train',shuffle=False,**data_kwargs)
val_dataset = Segmentation(val_img_path, mode='val',shuffle=False,**data_kwargs)


model = deeplabv3_resnet101(pretrained=True)
model.eval()

if os.path.exists("./train_results"):
    shutil.rmtree("./train_results")
os.makedirs("./train_results")
if os.path.exists("./val_results"):
    shutil.rmtree("./val_results")
os.makedirs("./val_results")

with torch.no_grad():

    for index,(images,filename) in tqdm(enumerate(train_dataset)):
        images = images.unsqueeze(0)
        input_batch = images.to(device)
        model.to(device)
        output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(images.size()[-2:])
        r.putpalette(colors)
        out_img = np.array(r)
        cv2.imwrite("train_results/%s_results.png"%filename.split("/")[-1].split(".")[0], out_img)


    for index,(images,filename) in tqdm(enumerate(val_dataset)):
        images = images.unsqueeze(0)
        input_batch = images.to(device)
        model.to(device)
        output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(images.size()[-2:])
        r.putpalette(colors)
        out_img = np.array(r)
        cv2.imwrite("train_results/%s_results.png"%filename.split("/")[-1].split(".")[0], out_img)











