import sys
sys.path.append(".")
sys.path.append("./loss_functions")
from loss_functions import metameric_loss
import cv2
import torch

if len(sys.argv) < 3:
    print("Usage: $ python calc_metameric_loss im1.jpg im2.jpg [format]")
    print("NOTE the format argument is optional. Write \"gray\" to force images to be gray")
    print("and compare on 1 channel. Write \"RGB\" to force images to be RGB and compare on 3 channels.")
    print("If not supplied will follow source image file format.")
    print("Need to supply 2 image filenames. Exiting...")
    exit()

def load_image_torch(filename, format="source"):
    im_cv = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if len(im_cv.shape) == 3:
        if format == "gray":
            im_cv = cv2.cvtColor(im_cv,  cv2.COLOR_BGR2GRAY)
            im_torch = torch.tensor(im_cv).float()[None, None,...] / 255.
        else:
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
            im_torch = torch.tensor(im_cv).permute(2,0,1).float()[None,...] / 255.
    else:
        if format == "RGB":
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_GRAY2RGB)
            im_torch = torch.tensor(im_cv).permute(2,0,1).float()[None,...] / 255.
        else:
            im_torch = torch.tensor(im_cv).float()[None, None,...] / 255.
    return im_torch

format = "source"

if len(sys.argv) > 3:
    format = sys.argv[3]

im1 = load_image_torch(sys.argv[1], format)
im2 = load_image_torch(sys.argv[2], format)

loss_func = metameric_loss.MetamericLoss(alpha=0.03, real_viewing_distance=0.7, real_image_width=0.2, mode="quadratic")

loss = loss_func(im1, im2, visualise_loss=True)

print("Metameric loss between images %0.10f" % loss.item())

import matplotlib.pyplot as plt
plt.imshow(loss_func.loss_map)
plt.colorbar()
plt.show()
