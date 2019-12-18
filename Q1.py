from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
from classes import classes
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
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
        # self.dropout = Dropout(0.4)
        self.softmax = Activation('softmax')
        self.local_response = MyModel.local_response_normalization
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
                             self.layer_7, self.layer_8, self.layer_9, self.layer_10, self.layer_11, self.layer_12]

    @staticmethod
    def local_response_normalization(x):
        return tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    def concat1(self, x):
        return tf.concat((self.conv2a(x[:, :, :, :48]), self.conv2b(x[:, :, :, 48:])), 3)

    def concat2(self, x):
        return tf.concat((self.conv4a(x[:, :, :, :192]), self.conv4b(x[:, :, :, 192:])), 3)

    def concat3(self, x):
        return tf.concat((self.conv5a(x[:, :, :, :192]), self.conv5b(x[:, :, :, 192:])), 3)

    # Network definition
    def __call__(self, x, **kwargs):
        for layer in self.model_layers:
            x = layer(x)
        return self.softmax(x)

    def layer_1(self, x):
        x = self.conv1(x)
        return self.relu(x)

    def layer_2(self, x):
        return self.local_response(x)

    def layer_3(self, x):
        return self.relu(self.concat1(self.maxpool(x)))

    def layer_4(self, x):
        return self.local_response(x)

    def layer_5(self, x):
        return self.maxpool(x)

    def layer_6(self, x):
        return self.relu(self.conv3(x))

    def layer_7(self, x):
        return self.relu(self.concat2(x))

    def layer_8(self, x):
        return self.relu(self.concat3(x))

    def layer_9(self, x):
        return self.maxpool(x)

    def layer_10(self, x):
        return self.relu(self.dense1(self.flatten(x)))

    def layer_11(self, x):
        return self.relu(self.dense2(x))

    def layer_12(self, x):
        return self.dense3(x)

    def run_up_to(self, layer_index, x):
        for layer in self.model_layers[: layer_index]:
            x = layer(x)
        return x


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


def process_image(image):
    im = image.resize([HEIGHT, WIDTH])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]
    I = np.flip(I, 2)
    I = I - np.mean(I, axis=(0, 1), keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I


def main():
    steps = 200
    l = 1e-4  # regularization rate
    # Random image
    img = np.random.rand(HEIGHT, WIDTH, CHANNELS) * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGBA')
    img = process_image(img)

    to_show = (img[0] - np.min(img[0]))
    to_show = to_show / np.max(to_show)
    plt.figure()
    plt.title('initial random image')
    plt.imshow(to_show)

    # Loading pre-trained model
    model = MyModel()
    model(img)
    load_weights(model)

    print(np.argmax(model(img)[0]))

    label = tf.Variable(tf.constant(6))  # desired classification
    im = tf.Variable(img)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=10)

    @tf.function
    def train_step(I, labels):
        with tf.GradientTape() as tape:
            prediction = model(I)[0]
            loss = loss_function(labels, prediction) + tf.constant(l) * tf.square(tf.norm(I))
        gradients = tape.gradient(loss, [I, ])
        optimizer.apply_gradients(zip(gradients, [I, ]))

    for i in range(1, steps + 1):
        print('step {}'.format(i)) if i == 1 or not i % (steps // 10) else None
        train_step(im, [label])

    print(np.argmax(model(im)[0]))

    to_show_2 = im[0] - np.min(im[0])
    to_show_2 = to_show_2 / np.max(to_show_2)
    plt.figure()
    plt.title('after {} steps'.format(steps))
    plt.imshow(to_show_2)
    plt.show()


if __name__ == '__main__':
    main()
