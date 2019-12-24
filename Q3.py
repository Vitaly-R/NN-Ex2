from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from classes import classes
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from Model import load_trained_model, normalize_image, show_image, plot, process_image, generate_random_image


# todo: run on larger number of steps before attempting to change the code!!!!!!!!!


def get_image():
    image = Image.open("dog.png")
    image = process_image(image)
    return image


@tf.function
def step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index, original_norm):
    with tf.GradientTape() as tape:
        norm = tf.square(tf.norm(image) - original_norm)
        prediction = model.run_up_to(layer_index, image)[0]  # since we predict for one image at each step, no need for the batch index
        if len(prediction.shape) == 3:
            loss = -tf.math.reduce_mean(prediction[:, :, neuron_index]) + reg_coeff * norm
        else:
            loss = -tf.math.reduce_mean(prediction[neuron_index]) + reg_coeff * norm
    training_loss(loss)
    image_norm(norm)
    gradients = tape.gradient(loss, [image, ])
    optimizer.apply_gradients(zip(gradients, [image, ]))


def visualize_activation(model, layer_index, neuron_index, reg_coeff=1e-2, optimization_steps=50000, learning_rate=0.1):
    image = tf.Variable(generate_random_image() + get_image())
    original_norm = tf.constant(tf.norm(get_image()))
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    image_norm = tf.keras.metrics.Mean(name='input_norm')  # todo: maybe remove?
    losses = list()
    norms = list()  # todo: might remove
    for i in range(1, optimization_steps + 1):
        step(image, model, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index, original_norm)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
        print('step {}: loss - {}, norm differnce - {}'.format(i, losses[-1], norms[-1])) if i in [1, optimization_steps] or not i % (optimization_steps // 100) else None
    return np.asarray(image[0]), losses, norms


def main():
    layer_index = 14
    neuron_index = 9
    model = load_trained_model()
    result, loss, norm = visualize_activation(model, layer_index, neuron_index)
    print("Target classification: {}\nActual classification: {}".format(classes[neuron_index], classes[np.argmax(model(result.reshape((1, ) + result.shape)))]))
    result = np.flip(result, 2)
    result = normalize_image(result)
    show_image(result)
    x = [i for i in range(len(loss))]
    plot(x, loss, title='loss')
    plot(x, norm, title='norm difference from original image')
    plt.show()


if __name__ == '__main__':
    main()
