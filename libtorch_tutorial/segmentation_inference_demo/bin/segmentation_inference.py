
import torch
import os
import numpy as np
import cv2
import shutil
from srcs.segment_model import deeplabv3_resnet101, fcn_resnet101


def cv2_imread(path="test_image.jpg"):

    images = cv2.imread(path, -1)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    images = cv2.resize(images, (512, 512))

    cv_img_tensor = torch.from_numpy(images.astype(np.float32))

    cv_img_tensor = cv_img_tensor.permute((2, 0, 1))
    cv_img_tensor = cv_img_tensor / 255.0
    cv_img_tensor[0] = (cv_img_tensor[0] - 0.485) / 0.229
    cv_img_tensor[1] = (cv_img_tensor[1] - 0.456) / 0.224
    cv_img_tensor[2] = (cv_img_tensor[2] - 0.406) / 0.225
    test_images = cv_img_tensor.unsqueeze(0).float()
    return test_images


if __name__=="__main__":

    output_dir = "./results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # An instance of segmentation model.
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()

    example = torch.rand(1, 3, 512, 512)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("deeplabv3_resnet101_model.pt")

    cv2_img_tensor = cv2_imread("test_image.jpg")
    print(cv2_img_tensor.size())
    # forward
    output = traced_script_module(cv2_img_tensor)
    print("python output results:", output.size(), output[0, 0, :6, :6])
    out_pred = output[0].argmax(0)
    out_pred = out_pred.data.cpu().numpy()
    print(np.unique(out_pred))
    cv2.imwrite("%s/python_inference.png" % output_dir, out_pred)



