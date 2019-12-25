import matplotlib.pyplot as plt
from Q1 import q1
from Q2 import q2
from Q3 import q3
from Q4 import q4

"""
Main file of Ex2 in Introduction to Neural Networks

---------------- Network Architecture ----------------
The network itself is a pre-trained Alexnet network.
For the purpose of the exercise, we split the layers of the network in the following way:
input layer: A batch of 1 RGB image. Layer output shape - (1, 224, 224, 3).
layer 1: A Convolution layer followed by a ReLU activation with an output of shape (1, 56, 56, 96).
layer 2: A Local Response Normalization layer with an output of shape (1, 56, 56, 96).
layer 3: A Max Pool layer with an output shape of (1, 27, 27, 96).
layer 4: A Concat of two Convolution layers (the input is split along the last axis into two tensors 
         with shape (1, 27, 27, 48)) followed by a ReLU activation with an output o shape (1, 27, 27, 256).
layer 5: A Local Response Normalization layer with an output of shape (1, 27, 27, 256).
layer 6: A Max Pool layer with an output shape of (1, 13, 13, 256).
layer 7: A Convolution layer followed by a ReLU activation with an output of shape (1, 13, 13, 384).
layer 8: A Concat of two Convolution layers (the input is split along the last axis into two tensors 
         with shape (1, 13, 13, 192)) followed by a ReLU activation with an output o shape (1, 13, 13, 384).
layer 9: A Concat of two Convolution layers (the input is split along the last axis into two tensors 
         with shape (1, 13, 13, 192)) followed by a ReLU activation with an output o shape (1, 13, 13, 256).
layer 10: A Max Pool layer with an output shape of (1, 6, 6, 256).
layer 11: A Flattening layer (into shape (1, 9216)), followed by a Dense layer of 4096 neurons, followed by a ReLU activation. Output shape is (1, 4096).
layer 12: A Dense layer of 4096 neurons followed by a ReLU activation. Output shape is (1, 4096).
layer 13: A Dense layer of 1000 neurons. Output shape is (1, 1000).
layer 13a: A Softmax activation layer with output shape (1, 1000).

"""


def main():
    # q1(layer_index=14, neuron_index=84, show_plots=True, num_steps=5000)
    # q2(layer_index=1, neuron_index=50, show_plots=True, num_steps=5000)
    # q3(neuron_index=97, show_plots=True, num_steps=5000)
    q4(class_index=267)
    plt.show()


if __name__ == '__main__':
    main()
