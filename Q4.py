from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from classes import classes
from Model import load_trained_model, read_image, show_image, HEIGHT, WIDTH


def set_zero_block(im, pos_x, pos_y, width_block, height_block):
    """
    Creates a copy of the given image and zeros-out the area with the given top-left corner, width and height.
    :param im: The image in which to zero-out the area.
    :param pos_x: x coordinate of the top-left corner.
    :param pos_y: y coordinate of the top left corner.
    :param width_block: Width of the area to zero-out.
    :param height_block: Height of the area to zero-out.
    :return: A copy of the given image with the selected area zeroed-out.
    """
    im_with_block = im.copy()
    im_with_block[:, pos_x: pos_x + height_block, pos_y: pos_y + width_block, :] = 0
    return im_with_block


def get_batch_of_blocked_images(im, zero_block_width, zero_block_height):
    """
    Generates a list of copies of the input image, such that each image has a block of zeros with the given width and height in different non-overlapping areas.
    :param im: The image over which to generate the zero blocks.
    :param zero_block_width: Width of each block.
    :param zero_block_height: Height of each block.
    :return: A list of copies of the image with zeroed-out blocks.
    """
    blocked = list()
    for i in range(0, HEIGHT - zero_block_height, zero_block_height):
        for j in range(0, WIDTH - zero_block_width, zero_block_width):
            new_im = set_zero_block(im, i, j, zero_block_width, zero_block_height)
            blocked.append(new_im)
    return blocked


def build_heat_map(probs, block_width, block_height):
    """
    Given an array of probabilities, constructs a heat map such that each probability is represented by a color in a block with size determined by the given width and height.
    :param probs: An array of probabilities to represent.
    :param block_width: Width of each block.
    :param block_height: Height of each block.
    :return: An image representing a heat map.
    """
    heat_map = np.zeros((block_height * probs.shape[0], block_width * probs.shape[1]))
    k = 0
    for i in range(probs.shape[0]):
        l = 0
        for j in range(probs.shape[1]):
            heat_map[k:k + block_height, l:l + block_width] = probs[i, j]
            l += block_width
        k += block_height
    return heat_map


def q4(model, class_index=267, image_path='./poodle.jpg', block_width=20, block_height=20):
    """
    Given an image which is correctly classified by the given model (and visually obvious to us), constructs a heat map which shows the significance of each block of the image
    to the overall classification, by zeroing-out blocks of the original image and checking the probabilities of the class over the modified image.
    :param model: A pre-trained model.
    :param class_index: The class index of the image.
    :param image_path: The path of the image to classify.
    :param block_width: Width of each block to zero-out.
    :param block_height: Height of each block to zero-out.
    """
    im = read_image(image_path)
    batch = get_batch_of_blocked_images(im, block_width, block_height)
    probs = list()
    for i, image in enumerate(batch):
        print("Step", i + 1, "out of", len(batch)) if i in [0, len(batch) - 1] or not (len(batch) // 10) or not (i % (len(batch) // 10)) else None
        prediction = model(image)
        prediction = prediction.numpy()
        probs.append(prediction[0, class_index])
    probs = np.array(probs).reshape((HEIGHT // block_height, WIDTH // block_width))
    heat_map = build_heat_map(probs, block_width, block_height)
    show_image(heat_map, 'Question 4\nHeat Map with blocks of width {}, height {}\nClass: {}\nImage: {}'.format(block_width, block_height, classes[class_index], image_path), 'gray')
