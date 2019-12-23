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

    def layer_4(self, x):
        s = x.get_shape()
        y = tf.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], 48])
        y1 = tf.slice(x, [0, 0, 0, 48], [s[0], s[1], s[2], s[3] - 48])
        a = self.conv2a(y)
        b = self.conv2b(y1)
        c = tf.concat((a, b), 3)
        return self.relu(c)

    def layer_5(self, x):
        return self.local_response_normalization(x)

    def layer_6(self, x):
        return self.maxpool(x)

    def layer_7(self, x):
        return self.relu(self.conv3(x))

    def layer_8(self, x):
        s = x.get_shape()
        y = tf.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], 192])
        y1 = tf.slice(x, [0, 0, 0, 192], [s[0], s[1], s[2], s[3] - 192])
        a = self.conv4a(y)
        b = self.conv4b(y1)
        c = tf.concat((a, b), 3)
        return self.relu(c)

    def layer_9(self, x):
        s = x.get_shape()
        y = tf.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], 192])
        y1 = tf.slice(x, [0, 0, 0, 192], [s[0], s[1], s[2], s[3] - 192])
        a = self.conv5a(y)
        b = self.conv5b(y1)
        c = tf.concat((a, b), 3)
        return self.relu(c)

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
    model(tf.zeros((1, HEIGHT, WIDTH, CHANNELS)))
    load_weights(model)
    return model


def read_image(path='./AlexnetWeights/poodle.png'):
    img = Image.open(path)
    return process_image(img)


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
    img = normalize_image(image)
    plt.figure()
    plt.title(title)
    plt.imshow(img)


def plot(x, y1, y2=None, title='', xlabel='', ylabel='', label1='', label2=''):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2) if y2 is not None else None
    plt.legend() if label1 != '' or label2 != '' else None


@tf.function
def step(image, model, loss_function, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index):
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


def visualize_activation(model, layer_index, neuron_index, reg_coeff=1e-4, optimization_steps=25000, learning_rate=0.01):
    image = tf.Variable(read_image())

    loss_function = tf.keras.losses.CategoricalCrossentropy()
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # todo: maybe pass as a parameter for us to be able to change it for each layer
    image_norm = tf.keras.metrics.Mean(name='input_norm')  # todo: maybe remove?
    losses = list()
    norms = list()  # todo: might remove
    for i in range(1, optimization_steps + 1):
        print('step {}'.format(i)) if i in [1, optimization_steps] or not i % (optimization_steps // 10) else None
        step(image, model, loss_function, training_loss, image_norm, optimizer, reg_coeff, layer_index, neuron_index)
        losses.append(training_loss.result())
        norms.append(image_norm.result())
    return np.asarray(image[0]), losses, norms


@tf.function
def optimization_step(mode, layerIndex, neuronIndex, regCoeff, optimizer, lossFunction, trainingLoss):
    pass


def visualize_neuron(model, layerIndex, neuronIndex, regCoeff=1e-5, optimization_steps=10000, learning_rate=0.01):
    image = tf.Variable(generate_random_image())
    lossFunction = None  # set as negative


def get_random_target_activation_for_model(layer_index):
    """
    Generates a random activation of a layer in the model.
    (20.12.19 - At this points, it is mainly used as a way to determine that the algorithm works in general,
    since generating actual target activations for hidden layers that make sense is complicated)
    :param layer_index: The index of the layer in which the activation is requested (assumes the index is valid).
    :return: A random activation with the shape of the output of the model when running up to the specific layer (including).
    """
    if layer_index == 0:
        # for this, the model simply returns the input layer.
        return read_image()
    elif layer_index in [1, 2]:
        # layer 1 is a regular convolution layer followed by a relu activation function.
        # layer 2 is a local response normalization layer.
        return np.random.rand(1, 56, 56, 96)
    elif layer_index == 3:
        # layer 3 is a max pool layer.
        return np.random.rand(1, 27, 27, 96)
    elif layer_index in [4, 5]:
        # layer 4 splits the input into two (1, 27, 27, 48)-shaped tensors, preforms a different convolution
        # on each into a (1, 27, 27, 128)-shaped tensor, concatenates them, and preforms a relu activation on the result.
        # layer 5 is a local normalization layer.
        return np.random.rand(1, 27, 27, 256)
    elif layer_index == 6:
        # layer 6 is a max pool layer
        return np.random.rand(1, 13, 13, 256)
    elif layer_index in [7, 8]:
        # layer 7 is a regular convolution layer.
        # layer 8 splits the input into two (1, 13, 13, 192)-shaped tensors, preforms a different convolution
        # on each into a (1, 13, 13, 192)-shaped tensor, concatenates them, and preforms a relu activation on the result.
        return np.random.rand(1, 13, 13, 384)
    elif layer_index == 9:
        # layer 9 splits the input into two (1, 13, 13, 192)-shaped tensors, preforms a different convolution
        # on each into a (1, 13, 13, 192)-shaped tensor, concatenates them, and preforms a relu activation on the result.
        return np.random.rand(1, 13, 13, 256)
    elif layer_index == 10:
        # layer 10 is a max pool layer.
        return np.random.rand(1, 6, 6, 256)
    elif layer_index in [11, 12]:
        # layer 11 is flattening the input into a vector with shape (1, 9216), passes the vector through a dense
        # layer with 4096 neurons, and then preforms relu activation.
        # layer 12 is a dense layer with 4096 neurons, and a relu activation afterwards.
        return np.random.rand(1, 4096)
    elif layer_index == 13:
        # layer 13 is a dense layer with 4096 inputs and 1000 outputs, and a softmax activation on the output vector afterwards.
        activation = np.zeros((1000, ))
        activation[np.random.randint(0, 1000)] = 1
        return activation


def main():
    layer_index = 2
    neuron_index = 3
    model = load_trained_model()
    image, losses, norms = visualize_activation(model, layer_index, neuron_index)
    show_image(image, 'Optimized Image After {} Steps'.format(len(losses)))
    plot([i for i in range(len(losses))], losses, title='Loss', xlabel='iteration', ylabel='loss')
    plot([i for i in range(len(norms))], norms, title='Norm', xlabel='iteration', ylabel='norm')
    plt.show()


if __name__ == '__main__':
    main()
