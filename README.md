## Incremental Learning Sentiment Analysis
This project demonstrates an approach to sentiment analysis using incremental learning. The focus is on processing a large dataset of Amazon reviews and building a sentiment classification model using Support Vector Machines (SVM) and performing incremental learning using SGD classifiers with word embeddings.

### Table of Contents
1. ###### Introduction
2. ###### Dataset
3. ###### Word Embeddings
4. ###### Data Preprocessing
4. ###### Model Training
5. ###### Evaluation
6. ###### Usage
6. ###### Results
   
#### Introduction
Sentiment analysis is the process of determining the emotional tone behind a series of words. It is used to gain an understanding of the attitudes, opinions, and emotions expressed in text. This project uses a large dataset of Amazon reviews to train a sentiment classification model incrementally.

#### Dataset
The dataset used in this project is the Amazon Reviews dataset, the dataset consist of about 4 million samples (train and test) which is downloaded from Kaggle. It contains millions of reviews labeled as positive or negative.

#### Word Embeddings
Word embeddings are utilized to transform textual data into numerical representations. In this project, we employ the GloVe (Global Vectors for Word Representation) model, specifically using the pre-trained glove-twitter-50 model. These embeddings encode semantic relationships among words, proving invaluable for NLP tasks such as sentiment analysis.

In traditional machine learning algorithms, inputs typically consist of fixed-size vectors. However, in NLP tasks, data often varies in length, presenting a significant challenge. To address this, we computed word embeddings for each word in our task and then generated fixed-length vectors by averaging these embeddings for each input.  

Here are several methods commonly used to create fixed-length embeddings from variable-length inputs in natural language processing (NLP):

#### Averaging Word Embeddings:
   ##### Description: 
      Compute the average of all word embeddings in the input text.
   ##### Advantages: 
      Simple and computationally efficient. Preserves some semantic information from the original text.
   #### Example: 
      If the input text is "The quick brown fox", compute the average of the embeddings for "the", "quick", "brown", and "fox".

#### Summing Word Embeddings:
   ##### Description: 
       Sum all word embeddings in the input text.
   ##### Advantages: 
      Simple and straightforward.
   ##### Example: 
   For the input "The quick brown fox", sum the embeddings for "the", "quick", "brown", and "fox".
   
#### Using a pooling mechanism (Max or Min Pooling):
   ##### Description: 
      Apply max or min pooling over the embeddings of all words in the input text.
   ##### Advantages: 
      Captures the most salient features of the input text.
   ##### Example: 
      For max pooling, select the maximum value from each dimension across all word embeddings in the input.
      
#### Concatenation of Word Embeddings:
   ##### Description: 
      Concatenate embeddings of individual words into a single vector representation.
   ##### Advantages: 
      Preserves sequential information of the input text.
   ##### Example: 
      Concatenate the embeddings of "the", "quick", "brown", and "fox" into a single vector.
   
#### Using Recurrent Neural Networks (RNNs) or Transformers:
   ##### Description: 
      Utilize RNNs or Transformer models that inherently handle variable-length sequences and produce fixed-length representations (e.g., through the final hidden state of an RNN or the [CLS] token in Transformers like BERT).
   ##### Advantages: 
      Captures context and dependencies between words effectively.
   ##### Example: 
      For RNNs, use the final hidden state after processing the entire sequence. For Transformers, use the [CLS] token output.

#### Dimensionality Reduction (PCA, t-SNE):
   ##### Description: 
      Apply dimensionality reduction techniques like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of word embeddings.
   ##### Advantages: 
      Reduces computational complexity and noise in the data.
   ##### Example: 
      Project high-dimensional word embeddings into a lower-dimensional space while preserving semantic relationships.
