from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
import tensorflow as tf
from Model import load_trained_model, process_image,  show_image, plot, normalize_image


def read_image(path):
    img = Image.open(path)
    return process_image(img)


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index):
    with tf.GradientTape() as tape:
        y = tf.signal.rfft2d(image)
        y = tf.math.real(y)
        y = tf.norm(y)
        norm = tf.square(y)
        prediction = model.run_up_to(layer_index, image)[0]
        if len(prediction.shape) == 3:
            loss = -tf.math.reduce_mean(prediction[:, :, neuron_index]) + reg_coeff * norm
        else:
            loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, image_path, layer_index, neuron_index, reg_coeff=1e-4, optimization_steps=55000, learning_rate=0.01):
    image = tf.Variable(read_image(image_path))
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    image_norm = tf.keras.metrics.Mean(name='input_norm')
    losses = list()
    norms = list()
    for i in range(1, optimization_steps + 1):
        print('step {}'.format(i)) if i in [1, optimization_steps] or not i % (optimization_steps // 100) else None
        step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
    return np.asarray(image[0]), losses, norms


def q2(layer_index, neuron_index, image_path='./poodle.png', regularization_coefficient=0.5e-4, num_steps=55000, learning_rate=0.1, show_resulting_image=True, show_plots=False):
    model = load_trained_model()
    image, losses, norms = visualize_activation(model, image_path, layer_index, neuron_index, regularization_coefficient, num_steps, learning_rate)
    if show_resulting_image:
        image = normalize_image(image)
        show_image(image, 'Optimized Image After {} Steps'.format(len(losses)))
    if show_plots:
        plot([i for i in range(len(losses))], losses, title='Loss', xlabel='iteration', ylabel='loss')
        plot([i for i in range(len(norms))], norms, title='Norm', xlabel='iteration', ylabel='norm')
