from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from classes import classes
import tensorflow as tf
from PIL import Image
from Model import load_trained_model, normalize_image, show_image, plot, process_image, generate_random_image


def get_image(path):
    image = Image.open(path)
    image = process_image(image)
    return image


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, neuron_index, original_norm):
    with tf.GradientTape() as tape:
        norm = tf.square(tf.norm(image) - original_norm)
        prediction = model(image)[0]
        loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, image_path, neuron_index, reg_coeff=1e-2, optimization_steps=50000, learning_rate=0.1):
    image = tf.Variable(generate_random_image() + get_image(image_path))
    original_norm = tf.constant(tf.norm(get_image(image_path)))
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    image_norm = tf.keras.metrics.Mean(name='input_norm')
    losses = list()
    norms = list()
    for i in range(1, optimization_steps + 1):
        step(image, model, training_loss, image_norm, optimizer, reg_coeff, neuron_index, original_norm)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
        print('step {}: loss - {}, norm difference - {}'.format(i, losses[-1], norms[-1])) if i in [1, optimization_steps] or not (optimization_steps // 100) or not i % (optimization_steps // 100) else None
    return np.asarray(image[0]), losses, norms


def q3(neuron_index, image_path='./dog.png', regularization_coefficient=0.5e-4, num_steps=55000, learning_rate=0.1, show_resulting_image=True, show_plots=False):
    model = load_trained_model()
    result, loss, norm = visualize_activation(model, image_path, neuron_index, regularization_coefficient, num_steps, learning_rate)
    print("Target classification: {}\nActual classification: {}".format(classes[neuron_index], classes[np.argmax(model(result.reshape((1, ) + result.shape)))]))
    if show_resulting_image:
        result = np.flip(result, 2)
        result = normalize_image(result)
        show_image(result)
    if show_plots:
        x = [i for i in range(len(loss))]
        plot(x, loss, title='loss')
        plot(x, norm, title='norm difference from original image')
