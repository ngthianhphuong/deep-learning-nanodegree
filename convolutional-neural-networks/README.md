# Part II: Convolutional Neural Networks
This part applies PyTorch in 8 mini-projects and 1 final project for image classification task.
## 1. [Multi-Layer Perceptron, MNIST](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/mnist-mlp-exercise)
This notebook trains an MLP to classify images from the [MNIST](http://yann.lecun.com/exdb/mnist/) hand-written digit database. The process is broken down into the following steps:
- Load and visualize the data.
- Define a neural network using PyTorch nn module, specify loss function and optimizer.
- Train the network using training and validation datasets, save the best model.
- Evaluate the performance of trained model on a test dataset.

## 2. [Finding Edges and Custom Kernels](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/finding-edges)
This notebook walks through coding convolutional kernels/filters, applying them to an image, defining filters that detect horizontal or vertical edges and visualizing the outcome.

## 3. [Layer Visualization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/layer-visualization)
This mini-project defines four filters of a convolutional layer and a max-pooling layer, then visualizes the output of each filter and the pooling layer.

## 4. [CNN for CIFAR 10 Classification](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/cifar-10-classification)
This mini-project trains a CNN to classify images from the CIFAR-10 database. It consists of multiple steps:
- Load and prepare the data, split it into train, validation and test set.
- Define the CNN architecture and feedforward behavior.
- Define loss function and optimizer.
- Train the network using GPU and save the model with lowest validation loss.
- Test the network and analyze the results.

## 5. [Transfer Learning on flower images](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/transfer-learning)
This notebook uses VGGNet trained on the ImageNet dataset as a feature extractor and replace final classification layer by a customized one to classify flower images into 5 different categories. It consists of multiple steps:
- Load and transform data.
- Define the model by loading pretrained VGG16, freezing all parameters and replace last layer.
- Train the model using GPU.
- Test the model and analyze the outcome.

## 6. [Weight Initialization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/weight-initialization)
This mini-project walk through different methods of weight initialization for a neural network in order to get the model come to the solution quicker. This notebook uses [Fashion-MNIST database](https://github.com/zalandoresearch/fashion-mnist). In each step, two different initial weights of the same model are instantiated. We observe how the training loss decreases over time and compare the model behaviors. Different weights are initialized as follows:
- All zeros or ones weights.
- Uniform Distribution: general rule vs centered rule.
- Uniform rule vs Normal Distribution.
- Automatic Initialization vs Uniform rule vs Normal Distribution.

## 7. [Autoencoders](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder)
This mini-project is composed of 3 parts, each part corresponds to a notebook:
* [Linear Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/simple-autoencoder) builds a simple autoencoder to compress the MNIST dataset. Input data is passed through an encoder that makes a compressed representation of the input. Then, this representation is passed through a decoder to reconstruct the input data. The encoder and decoder are built with neural networks, then trained on example data.
* [Convolutional Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/convolutional-autoencoder) improves the autoencoder's performance using convolutional layers to compress the MNIST dataset. The encoder portion is made of convolutional and pooling layers and the decoder is made of transpose convolutional layers that learn to "upsample" a compressed representation.
* [De-noising Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/denoising-autoencoder) defines and trains a convolutional autoencoder to de-noise the noisy input images, with target is the non noisy image data.

## 8. [Style Transfer](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/convolutional-neural-networks/style-transfer)
This mini-project recreates a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch. Given one content image and one style image, using VGG19 network, we aim to create a new, target image which should contain the desired content and style components: objects and their arrangement are similar to that of the content image; style, colors, and textures are similar to that of the style image.

## 9. [PROJECT: Dog Breed Classifier using CNN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/dog-breed)

This project aims to build a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm should be able to identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
Along with exploring state-of-the-art CNN models for classification, this project aims to make important design decisions about the user experience of the final app. The goal is to understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. 
The project is composed of multiple steps:
- Pre-process images
- Build helper functions such as human detector and dog detector
- Experiment with building CNN classifiers from scratch
- Train the classifier using transfer learning
- Predict and analyze the results
