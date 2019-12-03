import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np
import cv2

def PIL_read(path = 'dog.png'):

    image = Image.open(path).convert('RGB')
    print("Image image:", np.array(transforms.Resize([224, 224], interpolation=Image.BILINEAR)(image))[:6, :6, 0])

    default_transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = default_transform(image)
    return image


def cv2_read(path='dog.png'):

    cv_img = cv2.imread(path)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (224, 224))

    cv_img_tensor = torch.from_numpy(cv_img.astype(np.float32))
    cv_img_tensor = cv_img_tensor.permute((2, 0, 1))
    cv_img_tensor = cv_img_tensor/ 255.0
    cv_img_tensor[0] = (cv_img_tensor[0] - 0.485) / 0.229
    cv_img_tensor[1] = (cv_img_tensor[1] - 0.456) / 0.224
    cv_img_tensor[2] = (cv_img_tensor[2] - 0.406) / 0.225

    return cv_img_tensor


if __name__ =="__main__":


    # An instance of your model.
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("resnet18_model.pt")

    # evalute time
    batch = torch.rand(64, 3, 224, 224)
    start = time()
    output = traced_script_module(batch)
    stop = time()
    print(str(stop-start) + "s")

    # read image
    image = cv2_read('dog.png')

    # forward
    output = traced_script_module(image.unsqueeze(0))
    print("python inference results:",output[0, :10])

    # print top-5 predicted labels
    labels = np.loadtxt('synset_words.txt', dtype= str, delimiter='\n')

    data_out = output[0].data.numpy()
    sorted_idxs = np.argsort(-data_out)

    for i, idx in enumerate(sorted_idxs[:5]):
      print('top-%d label: %s, score: %f' % (i, labels[idx], data_out[idx]))







