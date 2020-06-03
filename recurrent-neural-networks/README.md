# Recurrent Neural Networks
This part contains 5 mini-projects and 1 final project. It aims to apply PyTorch to implement RNN and LSTM in various tasks relating to Natural Language Processing.
## 1. [Time Series](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/time-series)
This is a walkthrough of code to give an idea of how PyTorch represents RNNs and how to represent memory in code.
## 2. [Character-level RNN](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/character-level-rnn-exercise)
This mini-project is about character-level text prediction with an LSTM, using Anna Karenina text file. Character data, after being pre-processed and encoded as integers, is fed into a LSTM that predicts the next character when given an input sequence. The LTSM model is then used to generate new text.
## 3. [Skip-Gram](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/skip-gram)
This mini-project is about implementing the Word2Vec model using the SkipGram architecture and Negative Sampling. This is a way to create word embedding for use in natural language processing.
## 4. [Sentiment Prediction](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/sentiment-prediction-lstm)
This mini-project uses a dataset of movie reviews, accompanied by sentiment labels: positive or negative, and implements a LSTM that performs sentiment analysis.
## 5. [Attention Model](https://github.com/ngthianhphuong/deep-learning-nanodegree/blob/master/recurrent-neural-networks/attention/)
This notebook shows how attention model is implemented, in isolation from a larger model.
## 6. [PROJECT: Generate TV Scripts](https://github.com/ngthianhphuong/deep-learning-nanodegree/tree/master/recurrent-neural-networks/tv-script)
In this project, a RNN is built and trained on part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons to generate a new, "fake" TV script, based on patterns it recognizes in the training data.
The project is broken down to multiple steps:
- Get and preprocess the text data
- Build the RNN
- Set hyperparameters and discuss how to choose them
- Train the RNN
- Generate new scripts based on a prime word
