import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import os
import numpy as np
import pandas as pd
from skimage import io, transform

class MyTrainData(data.Dataset):
  def __init__(self):
    self.video_path = '/data/FrameFeature/Penn/'
    self.video_file = '/data/FrameFeature/Penn_train.txt'
    fp = open(self.video_file, 'r')
    lines = fp.readlines()
    fp.close()
    self.video_name = []
    for line in lines:
      self.video_name.append(line.strip().split(' ')[0])
  
  def __len__(self):
    return len(self.video_name)
  
  def __getitem__(self, index):
    data = load_feature(os.path.join(self.video_path, self.video_name[index]))
    data = np.expand_dims(data, 2)
    return data


class FaceLandmarksDataset(data.Dataset):
  """Face Landmarks dataset."""

  def __init__(self, csv_file, root_dir, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.landmarks_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.landmarks_frame)

  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir,
                            self.landmarks_frame.iloc[idx, 0])
    image = io.imread(img_name)
    landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)
    sample = {'image': image, 'landmarks': landmarks}
  
    if self.transform:
      sample = self.transform(sample)
  
    return sample



import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
