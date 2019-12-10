
import torch
import os
import numpy as np
import cv2
import shutil
from srcs.segment_model import deeplabv3_resnet101, fcn_resnet101
from PIL import Image
from scipy.io import loadmat


def colorize_mask(mask, output_dir):
    """

    :param mask: predmask : (H,W)
    :param output_dir: the path to save file
    :return:
    """
    # mask: numpy array of the mask
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask.save('%s/pil_results.png' % output_dir)


def colorEncode(labelmap, colors, mode='RGB'):
    """

    :param labelmap: predmask : (H,W)
    :param colors: (nclasses,3)
    :param mode: 'RGB','BGR'
    :return:
    """
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def colorwithimage(img, color_pred):
    """

    :param img: (H,W,3)
    :param color_pred:  (H,W,3)
    :return:  (H,W,3)
    """
    cv2.addWeighted(img, 1, color_pred, 0.7, 0, img)
    return img

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
    return test_images,images


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

    cv2_img_tensor,images = cv2_imread("test_image.jpg")
    print(cv2_img_tensor.size())
    # forward
    output = traced_script_module(cv2_img_tensor)
    print("python output results:", output.size(), output[0, 0, :6, :6])
    out_pred = output[0].argmax(0)
    out_pred = out_pred.data.cpu().numpy()
    print(np.unique(out_pred))
    cv2.imwrite("%s/without_CRF_result.png" % output_dir, out_pred)

    # PIL color
    out = colorize_mask(out_pred,output_dir )

    # the way 2
    ade20k_color = loadmat('ade20k_color150.mat')['colors'] ## shape:(150,3)
    out = colorEncode(out_pred, ade20k_color, mode='RGB')
    cv2.imwrite("%s/color_result.png" % output_dir, out)

    out = colorwithimage(images, out)
    cv2.imwrite("%s/color_image.png" % output_dir, out)






