from PIL import Image
from torchvision import transforms
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import cv2
import shutil
from srcs.segment_model import deeplabv3_resnet101
from srcs.DenseCRF import my_post_CRF


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ##判断是否有gpu

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = deeplabv3_resnet101(pretrained=True)
model.eval()

images = cv2.imread("test_image.jpg", -1)
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
images = input_transform(images)

output_dir = "./results"
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
# os.makedirs(output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def TTA_inference(x, model, hfilp=True, vflip= False):
    """
    :param x: torch.float (N,C,H,W)
    :param model: torch model
    :param hfilp: bool True, False
    :param vflip: bool True, False
    :return: (N,C,H,W)
    """
    images = x.to(device)
    model.to(device)
    pred = model(images)['out']
    count = 1
    if hfilp:
        input_batch = images.flip(3)
        pred += model(input_batch)['out'].flip(3)
        count += 1
    if vflip:
        input_batch = images.flip(2)
        pred += model(input_batch)['out'].flip(2)
        count += 1

    return  pred/count



with torch.no_grad():
    test_images = images.unsqueeze(0).float()
    pred = TTA_inference(test_images, model)
    out_pred = pred[0]
    out_pred = out_pred.argmax(0)
    out_pred = out_pred.data.cpu().numpy()
    print(np.unique(out_pred))
    plt.figure(figsize=(6,6))
    plt.subplot(121)
    plt.imshow(out_pred)
    plt.title("without_CRF_TTA")
    cv2.imwrite("%s/without_CRF_TTA_result.png" % output_dir, out_pred)

    zz = out_pred.copy()
    images = images.data.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    crf_output = my_post_CRF(images, out_pred)
    print(np.unique(crf_output))
    plt.subplot(122)
    plt.imshow(crf_output)
    plt.title("with_CRF_TTA")
    cv2.imwrite("%s/with_CRF_TTA_results.png"% output_dir, crf_output)
    print("differt:", (out_pred-crf_output).sum())
    plt.savefig("%s/TTA_results.png" % output_dir, dpi=300)
    plt.show()

