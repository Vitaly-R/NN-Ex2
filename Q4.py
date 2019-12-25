from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Model import load_trained_model, process_image, show_image


def read_image(path):
    img = Image.open(path)
    return process_image(img)


def set_zero_block(im, pos_x, pos_y, width_block=20, height_block=20):
    im_with_block = im.copy()
    im_with_block[:, pos_x:pos_x + height_block, pos_y:pos_y + width_block, :] = 0
    return im_with_block


def get_batch_of_blocked_images(im):
    # assume the image is (224, 224)
    blocked = list()
    for i in range(0, 204, 20):
        for j in range(0, 204, 20):
            new_im = set_zero_block(im, i, j)
            blocked.append(new_im)
    return blocked


def build_heat_map(probs):
    heat_map = np.zeros((220, 220))
    k = 0
    l = 0
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            heat_map[k:k + 20, l:l + 20] = probs[i, j]
            l += 20
        k += 20
        l = 0
    return heat_map


def q4(class_index, image_path='./poodle.png'):
    model = load_trained_model()
    im = read_image(image_path)
    print("building batch...")
    batch = get_batch_of_blocked_images(im)
    probs = list()
    print("predicting...")
    for i, image in enumerate(batch):
        print("iteration ", i, "out of ", len(batch))
        prediction = model(image)
        prediction = prediction.numpy()
        probs.append(prediction[0, class_index])
    print("building heat map...")
    probs = np.array(probs).reshape((11, 11))
    heat_map = build_heat_map(probs)
    plt.imshow(heat_map, cmap=plt.cm.gray)
