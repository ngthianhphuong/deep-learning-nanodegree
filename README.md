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
This part uses PyTorch in various mini-projects of image classification tasks.
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









### Recurrent Neural Networks
### Generative Adversarial Networks
### Deployment with AWS Sagemaker

