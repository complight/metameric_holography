from matplotlib import image
import torch
import cv2
import sys
sys.path.append(".")
sys.path.append("./loss_functions")
from loss_functions.metamer_mse_loss import MetamerMSELoss
import math

n_orientations = 2

def load_image_torch(filename, format="source"):
    im_cv = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if len(im_cv.shape) == 3:
        if format == "gray":
            im_cv = cv2.cvtColor(im_cv,  cv2.COLOR_BGR2GRAY)
            im_torch = torch.tensor(im_cv).float()[None, None,...] / 255.
        else:
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2YCrCb)
            im_torch = torch.tensor(im_cv).permute(2,0,1).float()[None,...] / 255.
    else:
        if format == "RGB":
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_GRAY2YCrCb)
            im_torch = torch.tensor(im_cv).permute(2,0,1).float()[None,...] / 255.
        else:
            im_torch = torch.tensor(im_cv).float()[None, None,...] / 255.
    return im_torch

input_image = load_image_torch(sys.argv[1])

loss_func = MetamerMSELoss(n_orientations=n_orientations, n_pyramid_levels=5, real_image_width=0.6, real_viewing_distance=0.8)

noise_image = torch.rand_like(input_image)
loss = loss_func(noise_image, input_image, gaze=[0.5, 0.5])

metamer = loss_func.target_metamer

min_divisor = 2 ** loss_func.metameric_loss.n_pyramid_levels
height = input_image.size(2)
width = input_image.size(3)
required_height = math.ceil(height / min_divisor) * min_divisor
required_width = math.ceil(width / min_divisor) * min_divisor
padded_height = required_height-height
padded_width = required_width-width

if padded_height > 0:
    metamer = metamer[:,:,padded_height:,:]
if padded_width > 0:
    metamer = metamer[:,:,:,padded_width:]

metamer_cv = metamer[0,...].permute(1,2,0).numpy()
input_cv = input_image[0,...].permute(1,2,0).numpy()
if metamer_cv.shape[2] == 3:
    metamer_cv = cv2.cvtColor(metamer_cv, cv2.COLOR_YCrCb2RGB)
    input_cv = cv2.cvtColor(input_cv, cv2.COLOR_YCrCb2RGB)
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(input_cv)
plt.subplot(1,2,2)
plt.imshow(metamer_cv)
plt.show()

import numpy as np
image_to_save = metamer_cv
if image_to_save.shape[2] == 3:
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
elif image_to_save.shape[2] == 1:
    image_to_save = image_to_save[:,:,0]
cv2.imwrite("../output_images/metamer.png", (np.clip(image_to_save, 0, 1) * 255).astype(np.uint8))
