from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Model import load_trained_model, generate_random_image, normalize_image, show_image, plot


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index):
    with tf.GradientTape() as tape:
        norm = tf.square(tf.norm(image))
        prediction = model.run_up_to(layer_index, image)[0]  # since we predict for one image at each step, no need for the batch index
        if len(prediction.shape) == 3:
            loss = -tf.math.reduce_mean(prediction[:, :, neuron_index]) + reg_coeff * norm
        else:
            loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, layer_index, neuron_index, reg_coeff, optimization_steps, learning_rate):
    image = tf.Variable(generate_random_image())
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    image_norm = tf.keras.metrics.Mean(name='input_norm')  # todo: maybe remove?
    losses = list()
    norms = list()  # todo: might remove
    for i in range(1, optimization_steps + 1):
        step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
        print('step {}: loss - {}, image norm - {}'.format(i, losses[-1], norms[-1])) if i in [1, optimization_steps] or not i % (optimization_steps // 100) else None
    return np.asarray(image[0]), losses, norms


def q1(layer_index, neuron_index, regularization_coefficient=0.5e-4, num_steps=55000, learning_rate=0.1, show_resulting_image=True, show_plots=False):
    model = load_trained_model()
    result, loss, norm = visualize_activation(model, layer_index, neuron_index, regularization_coefficient, num_steps, learning_rate)
    if show_resulting_image:
        result = normalize_image(result)
        show_image(result)
    if show_plots:
        x = [i for i in range(len(loss))]
        plot(x, loss, title='loss')
        plot(x, norm, title='norm')
