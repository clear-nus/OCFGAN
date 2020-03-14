#  Provides some utility functions for images.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import numpy as np
import cv2
from math import sin, cos


def sanitize_images(imgs):
    if len(imgs.shape) == 3:
        imgs = imgs[:, :, :, None]
    if imgs.shape[-1] > 3:
        imgs = imgs.transpose((0, 2, 3, 1))
    if imgs[0].min() < -0.0001:
        imgs = (imgs + 1) / 2.0
    if imgs[0].max() <= 1.0:
        imgs *= 255.0
    return imgs.astype(np.uint8)

'''
Adapted from: https://stackoverflow.com/questions/42040747/more-idomatic-way-to-display-images-in-a-grid-with-numpy
'''

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def im2grid(imgs, out_file='image.png', shuffle=True, num_imgs=None):
    if num_imgs is None:
        num_imgs = imgs.shape[0]
    imgs = sanitize_images(imgs)
    if shuffle:
        imgs = imgs[np.random.permutation(imgs.shape[0])]
    imgs = imgs[:num_imgs]
    grid_image = gallery(imgs, ncols=int(np.sqrt(num_imgs)))
    if grid_image.shape[-1] == 1:
        cv2.imwrite(out_file, grid_image)
    else:
        cv2.imwrite(out_file, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
