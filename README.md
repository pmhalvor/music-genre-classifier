# music-genre-classifier
An experiment aimed to compare a range of ML-based music classifiers

We test three different types of machine learning classifiers (traditional, simple neural networks, and pre-trained models) on the [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). 


# Classifier Types
We want to measure performance between complex models and simpler ones. The experiment aims to  ultimately show how much compute resources are needed for good classifications. 

## Traditional
We train and briefly tune 3 traditional machine learning models: random forest, support vector machines, and k-nearest neighbors. 
Then measure the performance based on their macro F-1 scores. We also review category misclassification rate per category using a confusion matrix. 

## Neural networks
We then train three simple neural networks: feed-forward (mlp), cnn, rnn. This is mainly to establish a lower limit for performance of neural architectures. Minimal fine tuning should be done here to save resources and maintain the baseline score. 

## Pretrained models 
There exists many open-source audio/music classifiers. We will select one publicly available through HuggingFace. When choosing a model here, it is important to have a reproducible preprocessing step. Once this preprocessing is set, then we can re-run the previous experiments, but with the same format of data as used here, for a fair comparison. 
