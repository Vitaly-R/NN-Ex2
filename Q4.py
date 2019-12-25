from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from Model import MyModel, load_weights,  normalize_image

WIDTH = 224
HEIGHT = 224
CHANNELS = 3
CLASS = 267
wdir = './AlexnetWeights/'


def load_trained_model():
    model = MyModel()
    model(tf.zeros((1, HEIGHT, WIDTH, CHANNELS)))
    load_weights(model)
    return model


def read_image(path='./AlexnetWeights/poodle.png'):
    img = Image.open(path)
    return process_image(img)


def process_image(image):
    im = image.resize([HEIGHT, WIDTH])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]
    I = np.flip(I, 2)
    I = I - np.mean(I, axis=(0, 1), keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index=13):
    with tf.GradientTape() as tape:
        norm = tf.square(tf.norm(image))
        prediction = model(image)  # since we predict for one image at each step, no need for the batch index
        loss = -tf.math.reduce_mean(prediction) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(im, model, reg_coeff, optimization_steps, learning_rate):
    image = tf.Variable(im)
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    image_norm = tf.keras.metrics.Mean(name='input_norm')  # todo: maybe remove?
    losses = list()
    norms = list()  # todo: might remove
    predictions = list()
    for i in range(1, optimization_steps + 1):
        prediction = model(image)
        predictions.append()
        losses.append(training_loss.result())
        norms.append(image_norm.result())
        print('step {}: loss - {}, image norm - {}'.format(i, losses[-1], norms[-1])) if i in [1, optimization_steps] or not i % (optimization_steps // 100) else None
    return predictions, losses, norms


def q4():

    model = load_trained_model()
    im = read_image()
    plt.imshow(im[0])
    plt.show()
    print("building batch...")
    batch = get_batch_of_blocked_images(im)
    probs = list()
    print("predicting...")
    for i, image in enumerate(batch):
        print("iteration ", i, "out of ", len(batch))
        prediction = model(image)
        prediction = prediction.numpy()
        probs.append(prediction[0, CLASS])
    print("building heat map...")
    probs = np.array(probs).reshape((11, 11))
    heat_map = build_heat_map(probs)
    plt.imshow(heat_map, cmap=plt.cm.gray)
    plt.show()


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

