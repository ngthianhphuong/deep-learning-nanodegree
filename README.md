# Deep Learning Nanodegree
This repository contains coding exercises, Jupyter notebooks and Python scripts made during Udacity's Deep Learning Nanodegree.

## Table of Contents
The Udacity's Deep Learning Nanodegree is composed of 5 parts:
* [Neural Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master#neural-networks)
* [Convolutional Neural Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master#convolutional-neural-networks)
* [Recurrent Neural Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master#recurrent-neural-networks)
* [Generative Adversarial Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master#generative-adversarial-networks)
* [Deployment with AWS Sagemaker](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master#deployment-with-aws-sagemaker)



### Neural Networks
This first part contains various mini-projects and a final project.

#### 1. [Sentiment Classification with Neural Network from Scratch](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/sentiment-analysis) tackles 6 different tasks using movie review dataset and pure Python as tool:
- Curating the dataset, developing a predictive theory and identifying correlation between input and output data.
- Transforming text to numbers, creating the input and output data.
- Building neural network using pure Python.
- Understanding noise and making learning faster by reducing noise.
- Analyzing inefficiencies in network and making network train and run faster.
- Further reducing noise by strategically reducing the vocabulary.

#### 2. [Deep Learning with PyTorch](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/neural-networks/intro-to-pytorch): How to use PyTorch for building deep learning models.
* [Part 1 - Tensors in PyTorch](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%201%20-%20Tensors%20in%20PyTorch%20(Exercises).ipynb) is a gentle introduction to PyTorch, a framework for building and training neural networks. Several concepts are presented in this notebook such as tensors, math operations on tensors and how to convert tensor to numpy and back.
* [Part 2 - Neural Network in PyTorch](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%202%20-%20Neural%20Networks%20in%20PyTorch%20(Exercises).ipynb) shows how to build neural network with PyTorch to solve a (formerly) difficult problem, identifying text in an image based on MNIST dataset. Various ways to build a network were introduced: define a class with [nn module](https://pytorch.org/docs/master/nn.html), using torch.nn.functional and nn.Sequential. 
* [Part 3 - Training Neural Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%203%20-%20Training%20Neural%20Networks%20(Exercises).ipynb) shows how to backpropagate, how to define a loss function and how autograd works in PyTorch. Then describes how to train a neural network and how to make prediction on MNIST dataset.
* [Part 4 - Fashion-MNIST](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%204%20-%20Fashion-MNIST%20(Exercises).ipynb) builds and trains a neural network using PyTorch's nn module and so classifies [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
* [Part 5 - Inference and Validation](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%205%20-%20Inference%20and%20Validation%20(Exercises).ipynb) implements validation step after training to avoid overfitting. Then introduces how to add Dropout to the model and how to switch between training and validation mode. The model after validation is then used to make predictions on [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
* [Part 6 - Saving and Loading Models](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb) shows how to save trained models, how to load previously trained models to use in making predictions or to continue training on new data.
* [Part 7 - Loading Image Data](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Exercises).ipynb) uses a [dataset of cat and dog photos](https://www.kaggle.com/c/dogs-vs-cats) available from Kaggle to show how to load image data using datasets.ImageFolder from [torchvision](https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder), how to apply transformation on data and how to create dataloader. Then dives into how to augment data using several transformation techniques.
* [Part 8 - Transfer Learning](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/neural-networks/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Exercises).ipynb) shows how to use [pre-trained networks on ImageNet available from torchvision](https://pytorch.org/docs/0.3.0/torchvision/models.html) to solve challenging problems in computer vision. This notebook uses DenseNet and VGG19 pre-trained models to classify cat and dog images, the accuracy was more than 90%.

#### 3. [PROJECT: Predict Bike Sharing Pattern](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/neural-networks/predict-bike-sharing-pattern) builds a neural network to predict daily bike rental ridership.
Bike sharing companies want to predict how many bikes are needed for the near future from historical data in order to avoid losing money from potential riders if there are too few or from too many bikes that are not rented.
This project consists of multiples steps:
* Loading and preparing the data: creating dummy variables, scaling target variables, splitting the data into training, testing, and validation sets
* Implementing a neural network using Python classes.
* Training the network by choosing the number of iterations, the learning rate and the number of hidden nodes.
* Analyzing predictions and thinking about how well the model makes predictions and where and why it fails.

### Convolutional Neural Networks
This part uses PyTorch in 8 mini-projects and 1 final project of image classification.
#### 1. [Multi-Layer Perceptron, MNIST](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/mnist-mlp-exercise)
This notebook trains an MLP to classify images from the [MNIST](http://yann.lecun.com/exdb/mnist/) hand-written digit database. The process is broken down into the following steps:
- Load and visualize the data.
- Define a neural network using PyTorch nn module, specify loss function and optimizer.
- Train the network using training and validation datasets, save the best model.
- Evaluate the performance of trained model on a test dataset.

#### 2. [Finding Edges and Custom Kernels](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/finding-edges)
This notebook walks through coding convolutional kernels/filters, applying them to an image, defining filters that detect horizontal or vertical edges and visualizing the outcome.

#### 3. [Layer Visualization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/layer-visualization)
This mini-project defines four filters of a convolutional layer and a max-pooling layer, then visualizes the output of each filter and the pooling layer.

#### 4. [CNN for CIFAR 10 Classification](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/cifar-10-classification)
This mini-project trains a CNN to classify images from the CIFAR-10 database. It consists of multiple steps:
- Load and prepare the data, split it into train, validation and test set.
- Define the CNN architecture and feedforward behavior.
- Define loss function and optimizer.
- Train the network using GPU and save the model with lowest validation loss.
- Test the network and analyze the results.

#### 5. [Transfer Learning on flower images](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/transfer-learning)
This notebook uses VGGNet trained on the ImageNet dataset as a feature extractor and replace final classification layer by a customized one to classify flower images into 5 different categories. It consists of multiple steps:
- Load and transform data.
- Define the model by loading pretrained VGG16, freezing all parameters and replace last layer.
- Train the model using GPU.
- Test the model and analyze the outcome.

#### 6. [Weight Initialization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/weight-initialization)
This mini-project walk through different methods of weight initialization for a neural network in order to get the model come to the solution quicker. This notebook uses [Fashion-MNIST database](https://github.com/zalandoresearch/fashion-mnist). In each step, two different initial weights of the same model are instantiated. We observe how the training loss decreases over time and compare the model behaviors. Different weights are initialized as follows:
- All zeros or ones weights.
- Uniform Distribution: general rule vs centered rule.
- Uniform rule vs Normal Distribution.
- Automatic Initialization vs Uniform rule vs Normal Distribution.

#### 7. [Autoencoders](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder)
This mini-project is composed of 3 parts, each part corresponds to a notebook:
* [Linear Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/simple-autoencoder) builds a simple autoencoder to compress the MNIST dataset. Input data is passed through an encoder that makes a compressed representation of the input. Then, this representation is passed through a decoder to reconstruct the input data. The encoder and decoder are built with neural networks, then trained on example data.
* [Convolutional Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/convolutional-autoencoder) improves the autoencoder's performance using convolutional layers to compress the MNIST dataset. The encoder portion is made of convolutional and pooling layers and the decoder is made of transpose convolutional layers that learn to "upsample" a compressed representation.
* [De-noising Autoencoder](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/autoencoder/denoising-autoencoder) defines and trains a convolutional autoencoder to de-noise the noisy input images, with target is the non noisy image data.

#### 8. [Style Transfer](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/convolutional-neural-networks/style-transfer)
This mini-project recreates a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch. Given one content image and one style image, using VGG19 network, we aim to create a new, target image which should contain the desired content and style components: objects and their arrangement are similar to that of the content image; style, colors, and textures are similar to that of the style image.
#### 9. [PROJECT: Dog Breed Classifier using CNN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/convolutional-neural-networks/dog-breed)

This project aims to build a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm should be able to identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
Along with exploring state-of-the-art CNN models for classification, this project aims to make important design decisions about the user experience of the final app. The goal is to understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. 
The project is composed of multiple steps:
- Pre-process images
- Build helper functions such as human detector and dog detector
- Experiment with building CNN classifiers from scratch
- Train the classifier using transfer learning
- Predict and analyze the results

### [Recurrent Neural Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks)
This part contains 5 mini-projects and 1 final project. It aims to apply PyTorch to implement RNN and LSTM in various tasks relating to Natural Language Processing.
#### 1. [Time Series](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/time-series)
This is a walkthrough of code to give an idea of how PyTorch represents RNNs and how to represent memory in code.
#### 2. [Character-level RNN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/character-level-rnn-exercise)
This mini-project is about character-level text prediction with an LSTM, using Anna Karenina text file. Character data, after being pre-processed and encoded as integers, is fed into a LSTM that predicts the next character when given an input sequence. The LTSM model is then used to generate new text.
#### 3. [Skip-Gram](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/skip-gram)
This mini-project is about implementing the Word2Vec model using the SkipGram architecture and Negative Sampling. This is a way to create word embedding for use in natural language processing.
#### 4. [Sentiment Prediction](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/sentiment-prediction-lstm)
This mini-project uses a dataset of movie reviews, accompanied by sentiment labels: positive or negative, and implements a LSTM that performs sentiment analysis.
#### 5. [Attention Model](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/recurrent-neural-networks/attention/)
This notebook shows how attention model is implemented, in isolation from a larger model.
#### 6. [PROJECT: Generate TV Scripts](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/tv-script)
In this project, a RNN is built and trained on part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons to generate a new, "fake" TV script, based on patterns it recognizes in the training data.
The project is broken down to multiple steps:
- Get and preprocess the text data
- Build the RNN
- Set hyperparameters and discuss how to choose them
- Train the RNN
- Generate new scripts based on a prime word

### [Generative Adversarial Networks](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan)
This part on GAN contains several mini-projects and a final project on image generation.

#### 1. [MNIST-GAN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/mnist-gan)
This mini-project aims to define and train a GAN. The notebook is about generating new images of handwritten digits that look as if they've come from the original MNIST training data. To generate new images, a Generator must learn about the features that make up images of digits and combine those features in new ways; imagining totally new images of digits.

#### 2. [Batch Normalization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/batch-norm)
This notebook shows how a simple MNIST classification model improves with the addition of batch normalization layers.

#### 3. [Deep Convolutional GAN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/dcgan)
This mini-project implements a deep convolutional GAN, first explored in 2016 in this [original paper](https://arxiv.org/pdf/1511.06434.pdf), capable of generating more complex images that look like house addresses from the [Google Streetview dataset, SVHN](http://ufldl.stanford.edu/housenumbers/).

#### 4. [CycleGAN, Image-to-Image Translation](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/gan/cycle-gan/)
In this mini-project, a CycleGAN is defined and trained to read in an image from a set of images of Yosemite national park taken either during the summer or winter and transform it so that it looks as if it belongs in the other season set.

#### 5. [PROJECT: Generating Faces](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/gan/face-generation/)
In this project, a DCGAN is defined and trained on [CelebFaces Attributes Dataset (CelebA) dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The final goal is to get a generator network to generate new images of faces that look as realistic as possible. At the end of the project, we'll be able to visualize the results of trained Generator to see how it performs; the generated samples should look like fairly realistic faces with small amounts of noise.
The project is broken down into a series of tasks:
- Get and preprocess the image data
- Create a DataLoader in PyTorch
- Define the model with a discriminator and a generator
- Initialize the network weights
- Define discriminator and generator loss functions
- Choose an optimizer
- Training
- Visualize generated images and highlight possible improvements.

### Deployment with AWS Sagemaker

