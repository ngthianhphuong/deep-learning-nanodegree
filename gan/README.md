# Part IV: Generative Adversarial Networks
This part on GAN contains several mini-projects and a final project on image generation.

## 1. [MNIST-GAN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/mnist-gan)
This mini-project aims to define and train a GAN. The notebook is about generating new images of handwritten digits that look as if they've come from the original MNIST training data. To generate new images, a Generator must learn about the features that make up images of digits and combine those features in new ways; imagining totally new images of digits.

## 2. [Batch Normalization](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/batch-norm)
This notebook shows how a simple MNIST classification model improves with the addition of batch normalization layers.

## 3. [Deep Convolutional GAN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/gan/dcgan)
This mini-project implements a deep convolutional GAN, first explored in 2016 in this [original paper](https://arxiv.org/pdf/1511.06434.pdf), capable of generating more complex images that look like house addresses from the [Google Streetview dataset, SVHN](http://ufldl.stanford.edu/housenumbers/).

## 4. [CycleGAN, Image-to-Image Translation](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/gan/cycle-gan/)
In this mini-project, a CycleGAN is defined and trained to read in an image from a set of images of Yosemite national park taken either during the summer or winter and transform it so that it looks as if it belongs in the other season set.

## 5. [PROJECT: GENERATE HUMAN FACES WITH GAN MODELS](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/gan/face-generation/)
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
