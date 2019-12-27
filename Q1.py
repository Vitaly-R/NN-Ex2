from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Model import generate_random_image, normalize_image, show_image, plot


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index):
    """
        A single input optimization step.
        :param image: The input which will be optimized this step.
        :param model: The model for which the input will be optimized.
        :param training_loss: A tensorflow metric keeping track of the loss during optimization.
        :param image_norm: A tensorflow metric keeping track of the norm of the image during optimization.
        :param optimizer: The optimizer used for applying the gradients.
        :param reg_coeff: The regularization coefficient over the norm of the image.
        :param layer_index: The index of the layer in which we would like to optimize for a certain neuron.
        :param neuron_index: The index of the neuron within that layer which we would like to maximize.
    """
    with tf.GradientTape() as tape:
        norm = tf.square(tf.norm(image))
        prediction = model.run_up_to(layer_index, image)[0]
        if len(prediction.shape) == 3:
            loss = -tf.math.reduce_mean(prediction[:, :, neuron_index]) + reg_coeff * norm
        else:
            loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, layer_index, neuron_index, reg_coeff, optimization_steps, learning_rate):
    """
        Runs the main optimization loop, which optimizes a random RGB image with values in range [-128, 128] to maximize the
        activation of the neuron {neuron_index} in layer {layer_index}.
        :param model: The model over which the optimization is made.
        :param layer_index: The index of the layer in the model in which we will maximize for a neuron.
        :param neuron_index: The index of the neuron in the layer which we want to maximize.
        :param reg_coeff: A regularization coefficient for the norm of the resulting image.
        :param optimization_steps: Number of optimization steps to preform.
        :param learning_rate: The learning rate for the optimizer.
        :return: The resulting image (RGB), a list of loss values per iteration, a list of image norms per iteration.
    """
    image = tf.Variable(generate_random_image())
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    image_norm = tf.keras.metrics.Mean(name='input_norm')
    losses = list()
    norms = list()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for i in range(1, optimization_steps + 1):
        step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
        print('step {}: loss - {}, image norm - {}'.
              format(i, losses[-1], norms[-1])) \
            if i in [1, optimization_steps] or not (optimization_steps // 100) or not i % (optimization_steps // 100) else None
    return np.asarray(image[0]), losses, norms


def q1(model, layer_index=13, neuron_index=84, regularization_coefficient=0.001, num_steps=50000, learning_rate=0.1, show_resulting_image=True, show_plots=False):
    """
        Runs an input optimization to maximize the neuron {neuron_index} in layer {layer_index} in the given model.
        :param model: A pre-trained model for which we optimize the input.
        :param layer_index: The index of the layer in the model in which we will maximize for a neuron.
        :param neuron_index: The index of the neuron in the layer which we want to maximize.
        :param regularization_coefficient: A regularization coefficient for the norm of the resulting image.
        :param num_steps: Number of optimization steps to preform.
        :param learning_rate: The learning rate for the optimizer.
        :param show_resulting_image: Weather to show the resulting image.
        :param show_plots: Weather to plot the loss and the norm per iteration.
    """
    image, loss, norm = visualize_activation(model, layer_index, neuron_index, regularization_coefficient, num_steps, learning_rate)
    if show_resulting_image:
        image = normalize_image(image)
        show_image(image, 'Question 1\nOptimized Image\niterations: {} | learning rate: {} | regularization coefficient: {}'.format(len(loss), learning_rate, regularization_coefficient))
    if show_plots:
        x = [i + 1 for i in range(len(loss))]
        plot(x, loss, 'Question 1\nLoss per Iteration\niterations: {} | learning rate: {} | regularization coefficient: {}'.
             format(num_steps, learning_rate, regularization_coefficient),
             'iteration', 'loss')
        plot(x, norm, 'Question 1\nSquared Image Norm per Iteration\niterations: {} | learning rate: {} | regularization coefficient: {}'.
             format(num_steps, learning_rate, regularization_coefficient),
             'iteration', 'image norm')
