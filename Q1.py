from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
from classes import classes
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras import Model
import matplotlib.pyplot as plt


WIDTH = 224
HEIGHT = 224
CHANNELS = 3
wdir = './AlexnetWeights/'


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # OPS
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        self.softmax = Activation('softmax')
        # self.dropout = Dropout(0.4)
        # Conv layers
        self.conv1 = Conv2D(filters=96, input_shape=(WIDTH, HEIGHT, CHANNELS), kernel_size=(11, 11), strides=(4, 4), padding='same')
        self.conv2a = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.conv2b = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4a = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4b = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5a = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5b = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        # Fully-connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(4096, input_shape=(100,))
        self.dense2 = Dense(4096)
        self.dense3 = Dense(1000)
        self.model_layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5, self.layer_6,
                             self.layer_7, self.layer_8, self.layer_9, self.layer_10, self.layer_11, self.layer_12, self.layer_13]

    @staticmethod
    def local_response_normalization(x):
        return tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    # Network definition
    def __call__(self, x, **kwargs):
        for layer in self.model_layers:
            x = layer(x)
        return x

    def run_up_to(self, layer_index, x):
        for layer in self.model_layers[: layer_index]:
            x = layer(x)
        return x

    def layer_1(self, x):
        return self.relu(self.conv1(x))

    def layer_2(self, x):
        return self.local_response_normalization(x)

    def layer_3(self, x):
        return self.maxpool(x)

    def layer_4a(self, x):
        return self.conv2a(x[:, :, :, :48])

    def layer_4b(self, x):
        return self.conv2b(x[:, :, :, 48:])

    def layer_4(self, x):
        return self.relu(tf.concat((self.conv2a(x[:, :, :, :48]), self.conv2b(x[:, :, :, 48:])), 3))

    def layer_5(self, x):
        return self.local_response_normalization(x)

    def layer_6(self, x):
        return self.maxpool(x)

    def layer_7(self, x):
        return self.relu(self.conv3(x))

    def layer_8(self, x):
        return self.relu(tf.concat((self.conv4a(x[:, :, :, :192]), self.conv4b(x[:, :, :, 192:])), 3))

    def layer_9(self, x):
        return self.relu(tf.concat((self.conv5a(x[:, :, :, :192]), self.conv5b(x[:, :, :, 192:])), 3))

    def layer_10(self, x):
        return self.maxpool(x)

    def layer_11(self, x):
        return self.relu(self.dense1(self.flatten(x)))

    def layer_12(self, x):
        return self.relu(self.dense2(x))

    def layer_13(self, x):
        return self.softmax(self.dense3(x))


def load_weights(model):
    model.conv1.set_weights((np.load(wdir + 'conv1.npy'), np.load(wdir + 'conv1b.npy')))
    model.conv2a.set_weights((np.load(wdir + 'conv2_a.npy'), np.load(wdir + 'conv2b_a.npy')))
    model.conv2b.set_weights((np.load(wdir + 'conv2_b.npy'), np.load(wdir + 'conv2b_b.npy')))
    model.conv3.set_weights((np.load(wdir + 'conv3.npy'), np.load(wdir + 'conv3b.npy')))
    model.conv4a.set_weights((np.load(wdir + 'conv4_a.npy'), np.load(wdir + 'conv4b_a.npy')))
    model.conv5a.set_weights((np.load(wdir + 'conv5_a.npy'), np.load(wdir + 'conv5b_a.npy')))
    model.conv4b.set_weights((np.load(wdir + 'conv4_b.npy'), np.load(wdir + 'conv4b_b.npy')))
    model.conv5b.set_weights((np.load(wdir + 'conv5_b.npy'), np.load(wdir + 'conv5b_b.npy')))
    model.dense1.set_weights((np.load(wdir + 'dense1.npy'), np.load(wdir + 'dense1b.npy')))
    model.dense2.set_weights((np.load(wdir + 'dense2.npy'), np.load(wdir + 'dense2b.npy')))
    model.dense3.set_weights((np.load(wdir + 'dense3.npy'), np.load(wdir + 'dense3b.npy')))


def load_trained_model():
    model = MyModel()
    model(np.zeros((1, HEIGHT, WIDTH, CHANNELS)))
    load_weights(model)
    return model


def generate_random_image():
    img = np.random.rand(HEIGHT, WIDTH, CHANNELS) * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGBA')
    return process_image(img)


def process_image(image):
    im = image.resize([HEIGHT, WIDTH])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]
    I = np.flip(I, 2)
    I = I - np.mean(I, axis=(0, 1), keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I


def normalize_image(image):
    im = image - np.min(image)
    im = im / np.max(im)
    return im


def show_image(image, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(image)


def plot(x, y1, y2=None, title='', xlabel='', ylabel='', label1='', label2=''):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2) if y2 is not None else None
    plt.legend() if label1 != '' or label2 != '' else None


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


def visualize_activation(model, layer_index, neuron_index, reg_coeff=1e-4, optimization_steps=10000, learning_rate=0.1):
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


def main():
    layer_index = 13
    neuron_index = 6
    model = load_trained_model()
    result, loss, norm = visualize_activation(model, layer_index, neuron_index)
    print("Target classification: {}\nActual classification: {}".format(classes[neuron_index], classes[np.argmax(model(result.reshape((1, ) + result.shape)))]))
    result = normalize_image(result)
    show_image(result)
    x = [i for i in range(len(loss))]
    plot(x, loss, title='loss')
    plot(x, norm, title='norm')
    plt.show()


if __name__ == '__main__':
    main()
