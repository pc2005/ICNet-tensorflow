
# %%
import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from tqdm import trange
from utils.config import Config
from model import ICNet, ICNet_BN

# %% [markdown]
# # Setup configurations

# %%
model_config = {'train': ICNet, 'trainval': ICNet,
                'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'cityscapes'
filter_scale = 1


class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)

    # You can choose different model here, see "model_config" dictionary. If you choose "others",
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval'

    # Set pre-trained weights here (You can download weight from Google Drive)
    model_weight = './model/cityscapes/icnet_cityscapes_trainval_90k.npy'

    # Define default input size here
    INFER_SIZE = (1024, 2048, 3)


cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()

# %% [markdown]
# # Create graph, session, and restore weights

# %%
# Create graph here
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)

# %% [markdown]
# # Run segmentation on single image

# %%
im1 = cv2.imread('/home/phil/workspace/code/ENet-Segmentation/data/Rec20190222164035/FrontCamera/594.png')

if im1.shape != cfg.INFER_SIZE:
    im1 = cv2.resize(im1, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))

results1 = net.predict(im1)
overlap_results1 = 0.5 * im1 + 0.5 * results1[0]
vis_im1 = np.concatenate(
    [im1/255.0, results1[0]/255.0, overlap_results1/255.0], axis=1)

plt.figure(figsize=(20, 15))
plt.imshow(vis_im1)
plt.show()


# # %%
# im2 = cv2.imread('/home/phil/workspace/code/ENet-Segmentation/data/Rec20190222164035/FrontCamera/594.png')
# if im2.shape != cfg.INFER_SIZE:
#     im2 = cv2.resize(im2, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))

# results2 = net.predict(im2)

# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
# overlap_results2 = 0.5 * im2 + 0.5 * results2[0]

# vis_im2 = np.concatenate(
#     [im2/255.0, results2[0]/255.0, overlap_results2/255.0], axis=1)

# plt.figure(figsize=(20, 15))
# plt.imshow(vis_im2)
# plt.show()

# %% [markdown]
# # Test inference speed

# %%
elapsed_times = []

for i in range(50):
    start_t = time.time()

    _ = net.predict(im1)

    duration = time.time() - start_t

    if i > 0:
        elapsed_times.append(duration)

print('Average time: {:.4f}, about {:.6f} fps'.format(
    np.mean(elapsed_times), 1/np.mean(elapsed_times)))
