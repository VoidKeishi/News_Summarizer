import pickle
import streamlit as st
from newspaper import Article
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load the logistic regression model
filename = 'logistic_regression_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the TF-IDF vectorizer
filename2 = 'tfidf_vectorizer.pkl'
tfidf_vectorizer = pickle.load(open(filename2, 'rb'))

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Load the set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess the text by tokenizing, removing stopwords, and performing stemming.
    """
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words and perform stemming
    processed_text = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    
    # Join the processed words back into a single string
    processed_text = ' '.join(processed_text)
    
    return processed_text


def summarize(url):
    # Retrieve the news article from the given URL, display its details, summary, and sentiment analysis.
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        # Display the article details
        st.subheader('Article Details')
        st.write('Title:', article.title)
        st.write('Authors:', ', '.join(article.authors))
        st.write('Publication Date:', str(article.publish_date))

        # Display the article summary
        st.subheader('Summary')
        st.write(article.summary)

        # Perform sentiment analysis on the article summary
        st.subheader('Sentiment Analysis')
        preprocessed_text = preprocess_text(article.summary)  # Preprocess the news summary
        X_new_tfidf = tfidf_vectorizer.transform([preprocessed_text])  # Transform the preprocessed text using TF-IDF vectorizer
        y_pred = loaded_model.predict(X_new_tfidf)  # Use the loaded model to make predictions
        st.write('Sentiment:', y_pred[0])

    except:
        # Display "Invalid URL" message
        st.error('Invalid URL')


# Create the main Streamlit app
def main():
    # Set the page title
    st.title("News Summarizer")

    # URL Input
    url = st.text_input('Enter the URL of the news article')

    # Summarize Button
    if st.button('Summarize'):
        if url:
            summarize(url)
        else:
            st.warning('Please enter a URL.')

if __name__ == '__main__':
    main()
