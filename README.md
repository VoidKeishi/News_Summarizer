# **News Summarizer desktop app with self-build sentiment analysis model**

## Table of Contents
### Introduction
### Installation
### Results

## I. Introduction
### 1) Sentiment analysis model using Logistic Regression

  #### a) Data
  - I'm using the Twitter Sentiment Analysis dataset on Kaggle, here is the link: [Data source](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

  #### b) Procedure
  - Extract Data: The notebook starts by importing the necessary libraries and loading the training and validation datasets. The datasets are then preprocessed to     prepare the text data for analysis.
  - Preprocessing Text: The text data in the datasets is preprocessed by converting it to lowercase, removing special characters using regular expressions, and    performing tokenization, stemming, and removal of stop words.
  - Apply Logistic Regression: The preprocessed text is used to train a logistic regression model. The dataset is split into training and testing sets, and the text is transformed into numerical features using TF-IDF vectorization. The logistic regression model is trained on the transformed features, and its performance is evaluated using classification metrics.
  - Results: The classification reports for the testing and validation sets are displayed, showing the precision, recall, and F1-score for each sentiment class. The trained logistic regression model is saved to a file for future use.

### 2) Desktop app
