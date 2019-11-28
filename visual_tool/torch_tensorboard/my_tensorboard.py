from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np

def genarte_result(outpu_dir = "tensorboard_results", loss_values=None):
    """

    :param outpu_dir: type:str output name
    :param loss_values: np.float (N,)
    :return:
    """
    import os
    import shutil
    if os.path.exists(outpu_dir):
        shutil.rmtree(outpu_dir)
    os.makedirs(outpu_dir)
    with SummaryWriter(log_dir=outpu_dir) as writer:
        for n_iter in range(100):
            writer.add_scalar('Loss/train', np.random.random(), n_iter)
            writer.add_scalar('Loss/test', np.random.random(), n_iter)
            writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

genarte_result()

"""
tensorboard --logdir='./tensorboard_results' --port=6006
"""