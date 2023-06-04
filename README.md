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
#### Libraries I'm using
- tkinter : A library for creating basic Graphical User Interfaces (GUI).
- pickle : A library for creating basic Graphical User Interfaces (GUI).
- newspaper : A library for extracting and parsing data from news articles provided a URL
- re : A library for working with regular expressions, used for text processing.
- nltk : The Natural Language Toolkit, which provides various tools and resources for natural language processing tasks such as tokenization, stemming, and stop words removal.
- sklearn : The scikit-learn library, which provides a collection of machine learning algorithms and tools for data preprocessing, modeling, and evaluation.

## II. Installation
1. Clone the repository or download the source code files.
2. Install the required libraries as mentioned above.
```
  pip install tkinter
  pip install pickle
  pip install newspaper3k
  pip install nltk
  pip install scikit-learn
```
Additionally, the code downloads the required NLTK resources using the nltk.download function. If you haven't downloaded these resources before, you can run the following lines of code once:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
3. Run the news_summarizer.py file using Python.
```
py News_Summarizer.py
```
## III. Result
The News Summarizer desktop app allows you to summarize news articles provided a URL and analyze the sentiment of the article's summary. The app utilizes a pre-trained logistic regression model for sentiment analysis.   
### How to use the app:   
  1. Launch the app by running the news_summarizer.py file.
  2. Enter the URL of a news article in the provided input field.
  3. Click the "Start" button to initiate the summarization process.
  4. The app will extract the article's title, authors, publication date, and summary.
  5. The sentiment of the summary will be analyzed using the pre-trained logistic regression model.
  6. The results, including the title, authors, publication date, summary, and sentiment, will be displayed in the respective fields.
  Note: If an invalid URL is entered or an error occurs during the process, an error message will be displayed.

  Feel free to explore the code and modify it as per your requirements. Enjoy summarizing news articles and analyzing their sentiment with the News Summarizer desktop app!    

## Sample image
![image](https://github.com/VoidKeishi/News_Summarizer/assets/118616093/410f18ee-9076-4705-9aea-ba6816c89e76)
## Video

[Video](https://drive.google.com/file/d/11hXwGjeYVIAAvCfQZzIQM1h7UmqQXNeP/view?usp=drive_link)


