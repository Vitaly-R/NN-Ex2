import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


WIDTH = 224  # width of the input image to the network
HEIGHT = 224  # height of the input image to the network
CHANNELS = 3  # number of color channels in the input image of the network
wdir = './AlexnetWeights/'  # directory of the weights of the network


class MyModel(Model):
    """
    A network with the architecture of Alexnet.
    """
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
        # A list of the layers of the network in order, allows us to get the output of a a specific layer.
        self.model_layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5, self.layer_6,
                             self.layer_7, self.layer_8, self.layer_9, self.layer_10, self.layer_11, self.layer_12,
                             self.layer_13, self.activate_prediction]

    @staticmethod
    def local_response_normalization(x):
        """
        A local response normalization layer.
        """
        return tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    # Network definition
    def __call__(self, x, **kwargs):
        """
        Runs the given input through the entire network.
        :param x: Input batch of images of shape (batch, HEIGHT, WIDTH, CHANNELS) where batch is the number of images.
        :param kwargs: ---
        :return: The prediction of the network for the images.
        """
        for layer in self.model_layers:
            x = layer(x)
        return x

    def run_up_to(self, layer_index, x):
        """
        Runs the network up to the layer given by the index.
        :param layer_index: Index of the layer.
        :param x: Input batch of images of shape (batch, HEIGHT, WIDTH, CHANNELS) where batch is the number of images.
        :return: The output of the last layer through which the batch was passed.
        """
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
        return self.dense3(x)

    def activate_prediction(self, x):
        return self.softmax(x)


def load_weights(model):
    """
    Loads the weights of the pre-trained Alexnet network into the given model.
    Assumes the model has the same architecture as Alexnet.
    :param model: The model into which to load the weights.
    """
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
    """
    Loads a pre-trained Alexnet model.
    """
    model = MyModel()
    model(np.zeros((1, HEIGHT, WIDTH, CHANNELS)))
    load_weights(model)
    return model


def generate_random_image():
    """
    Generates a random image of size (HEIGHT, WIDTH, CHANNELS) of type float32 with values in range [-128, 128].
    """
    img = np.random.rand(HEIGHT, WIDTH, CHANNELS) * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGBA')
    return process_image(img)


def read_image(path):
    """
    Reads an image from the given path and processes it.
    :param path: Path of the image.
    :return: A processed image.
    """
    img = Image.open(path)
    return process_image(img)


def process_image(image):
    """
    Processes the given image into a batch of one image of type float32 with values in range [-128, 128].
    :param image: A RGB image of type uint8.
    """
    im = image.resize([HEIGHT, WIDTH])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]
    I = np.flip(I, 2)
    I = I - np.mean(I, axis=(0, 1), keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I


def normalize_image(image):
    """
    Normalizes an image of to range [0, 1]
    """
    im = image - np.min(image)
    im = im / np.max(im)
    return im


def show_image(image, title='', cmap=None):
    """
    Shows the given image in a figure with the given title.
    """
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap) if cmap is not None else plt.imshow(image)


def plot(x, y, title='', xlabel='', ylabel=''):
    """
    Plots the given y as a function of x.
    :param x: x axis values
    :param y: y axis values
    :param title: title of the plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
