## Incremental Learning Sentiment Analysis
This project demonstrates an approach to sentiment analysis using incremental learning. The focus is on processing a large dataset of Amazon reviews and building a sentiment classification model using Support Vector Machines (SVM) with word embeddings.

###Table of Contents
1. Introduction
2. Dataset
3. Word Embeddings
4. Data Preprocessing
4. Model Training
5. Evaluation
6. Usage
6. Results
   
#### Introduction
Sentiment analysis is the process of determining the emotional tone behind a series of words. It is used to gain an understanding of the attitudes, opinions, and emotions expressed in text. This project uses a large dataset of Amazon reviews to train a sentiment classification model incrementally.

#### Dataset
The dataset used in this project is the Amazon Reviews dataset, the dataset consist of about 4 million samples (train and test) which is downloaded from Kaggle. It contains millions of reviews labeled as positive or negative.

#### Word Embeddings
Word embeddings are used to convert text data into numerical format. In this project, we use the GloVe (Global Vectors for Word Representation) model, specifically the glove-twitter-50 pre-trained model. Word embeddings capture semantic relationships between words, making them valuable for NLP tasks like sentiment analysis.
