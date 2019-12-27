from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Model import normalize_image, show_image, plot, generate_random_image, read_image


@tf.function
def step(image, model, training_loss, image_difference, optimizer, reg_coeff, neuron_index, original_image):
    """
        A single input optimization step.
        :param image: The input which will be optimized this step.
        :param model: The model for which the input will be optimized.
        :param training_loss: A tensorflow metric keeping track of the loss during optimization.
        :param image_difference: A tensorflow metric keeping track of the average difference between the original image and the optimized image pixel-wise..
        :param optimizer: The optimizer used for applying the gradients.
        :param reg_coeff: The regularization coefficient over the norm of the image.
        :param neuron_index: The index of the neuron within that layer which we would like to maximize.
        :param original_image: The original image.
    """
    with tf.GradientTape() as tape:
        abs_avg_diff = tf.abs(tf.math.reduce_mean(image - original_image))
        prediction = model.run_up_to(13, image)[0]
        loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * abs_avg_diff
    training_loss(loss)
    image_difference(abs_avg_diff)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, image_path, neuron_index, reg_coeff, optimization_steps, learning_rate):
    """
        Runs the main optimization loop, which optimizes an image which is correctly classified by the model
        such that after the optimization the image will be classified as the class given by the neuron in position {neuron_index}.
        :param model: The model over which the optimization is made.
        :param image_path: The path of the image to optimize.
        :param neuron_index: The index of the neuron in the layer which we want to maximize.
        :param reg_coeff: A regularization coefficient for the norm of the resulting image.
        :param optimization_steps: Number of optimization steps to preform.
        :param learning_rate: The learning rate for the optimizer.
        :return: The resulting image (RGB), a list of loss values per iteration, a list of image norms per iteration.
    """
    image = tf.Variable(read_image(image_path))
    original_norm = tf.constant(read_image(image_path))
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    image_difference = tf.keras.metrics.Mean(name='input_norm')
    losses = list()
    difference = list()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for i in range(1, optimization_steps + 1):
        step(image, model, training_loss, image_difference, optimizer, reg_coeff, neuron_index, original_norm)
        losses.append(training_loss.result())
        difference.append(image_difference.result())
        print('step {}: loss - {}, absolute average pixel value difference - {}'.
              format(i, losses[-1], difference[-1])) \
            if i in [1, optimization_steps] or not (optimization_steps // 100) or not i % (optimization_steps // 100) else None
    return np.asarray(image[0]), losses, difference


def q3(model, neuron_index=97, image_path='./dog.jpg', regularization_coefficient=5, num_steps=1000, learning_rate=0.01, show_resulting_image=True, show_plots=False):
    """
        Runs an input optimization to force the model to classify the input image as a different chosen class.
        :param model: A pre-trained model for which we optimize the input.
        :param neuron_index: The index of the neuron in the layer which we want to maximize.
        :param image_path: The path of the image to optimize.
        :param regularization_coefficient: A regularization coefficient for the norm of the resulting image.
        :param num_steps: Number of optimization steps to preform.
        :param learning_rate: The learning rate for the optimizer.
        :param show_resulting_image: Weather to show the resulting image.
        :param show_plots: Weather to plot the loss and the norm per iteration.
    """
    result, loss, difference = visualize_activation(model, image_path, neuron_index, regularization_coefficient, num_steps, learning_rate)
    if show_resulting_image:
        result = normalize_image(result)
        show_image(result, 'Question 3\nOptimized Image\niterations: {} | learning rate: {} | regularization coefficient: {}'.format(len(loss), learning_rate, regularization_coefficient))
    if show_plots:
        x = [i for i in range(len(loss))]
        plot(x, loss, 'Question 3\nLoss per Iteration\niterations: {} | learning rate: {} | regularization coefficient: {}'.
             format(num_steps, learning_rate, regularization_coefficient),
             'iteration', 'loss')
        plot(x, difference,
             'Question 3\nAbsolute Average Difference between the original and optimized images (pixel-wise) per Iteration\niterations: {} | learning rate: {} | regularization coefficient: {}'.
             format(num_steps, learning_rate, regularization_coefficient),
             'iteration', 'absolute average difference')
