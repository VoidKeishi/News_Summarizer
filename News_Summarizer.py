import tkinter as tk
import pickle
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
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


def summarize():

    # Retrieve the news article from the given URL, display its details, summary, and sentiment analysis.
    url = url_input.get("1.0", "end").strip()

    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        # Display the article details
        title.config(state="normal")
        title.delete("1.0", "end")
        title.insert("1.0", article.title)
        title.config(state="disabled")

        authors.config(state="normal")
        authors.delete("1.0", "end")
        authors.insert("1.0", ", ".join(article.authors))
        authors.config(state="disabled")

        publication_date.config(state="normal")
        publication_date.delete("1.0", "end")
        publication_date.insert("1.0", str(article.publish_date))
        publication_date.config(state="disabled")

        # Display the article summary
        summary.config(state="normal")
        summary.delete("1.0", "end")
        summary.insert("1.0", article.summary)
        summary.config(state="disabled")

        # Perform sentiment analysis on the article summary
        sentiment.config(state="normal")
        sentiment.delete("1.0", "end")
        
        preprocessed_text = preprocess_text(article.summary)  # Preprocess the news summary
        X_new_tfidf = tfidf_vectorizer.transform([preprocessed_text])  # Transform the preprocessed text using TF-IDF vectorizer
        y_pred = loaded_model.predict(X_new_tfidf)  # Use the loaded model to make predictions
        sentiment.insert("1.0", y_pred[0])
        sentiment.config(state="disabled")

        error_message.config(text="")  # Clear error message

    except:
        # Display "Invalid URL" message
        error_message.config(text="Invalid URL") 


        # Clear URL input
        url_input.delete("1.0", "end")
        url_input.focus_set()

        # Clear other fields
        title.config(state="normal")
        title.delete("1.0", "end")
        title.config(state="disabled")

        authors.config(state="normal")
        authors.delete("1.0", "end")
        authors.config(state="disabled")

        publication_date.config(state="normal")
        publication_date.delete("1.0", "end")
        publication_date.config(state="disabled")

        summary.config(state="normal")
        summary.delete("1.0", "end")
        summary.config(state="disabled")

        sentiment.config(state="normal")
        sentiment.delete("1.0", "end")
        sentiment.config(state="disabled")


# Create the main window
root = tk.Tk()
root.title("News Summarizer")
root.geometry("800x600")
root.resizable(False, False)
root.configure(bg='#F0F0F0')  # Set background color to white

# Create a style for the widgets
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), foreground="black", background='#F0F0F0')  # Set label font and colors
style.configure("TButton", font=("Arial", 12))

# Header
header_label = ttk.Label(root, text="News Summarizer", font=("Arial", 18, "bold"))
header_label.pack(pady=20)

# URL Input
url_label = ttk.Label(root, text="URL", font=("Arial", 14, "bold"))
url_label.pack()

input_frame = ttk.Frame(root)
input_frame.pack()

url_input = ScrolledText(input_frame, height=1, width=80, wrap="word", font=("Arial", 12))
url_input.pack(side="left")

# Summarize Button
button = ttk.Button(input_frame, text="Start", command=summarize, style="TButton")
button.pack(side="right", padx=5)

# Error Message
error_message = ttk.Label(root, text="", font=("Arial", 12, "bold"), foreground="red")
error_message.pack()

# Result Section
result_frame = ttk.Frame(root)
result_frame.pack(pady=10)

# Title
title_label = ttk.Label(result_frame, text="Title", font=("Arial", 14, "bold"))
title_label.pack()

title = ScrolledText(result_frame, height=1, width=100, state="disabled", wrap="word", font=("Arial", 12), bg='white', fg="black")
title.pack()

# Authors, Publication Date and Sentiment
info_frame = ttk.Frame(result_frame)
info_frame.pack(pady=10)

authors_label = ttk.Label(info_frame, text="Authors", font=("Arial", 12, "bold"))
authors_label.grid(row=0, column=0, padx=5)

authors = ScrolledText(info_frame, height=1, width=15, state="disabled", wrap="word", font=("Arial", 12), bg='white', fg="black")
authors.grid(row=0, column=1, padx=5)

publication_date_label = ttk.Label(info_frame, text="Publication\n      Date", font=("Arial", 12, "bold"))
publication_date_label.grid(row=0, column=2, padx=5)

publication_date = ScrolledText(info_frame, height=1, width=15, state="disabled", wrap="word", font=("Arial", 12), bg='white', fg="black")
publication_date.grid(row=0, column=3, padx=5)

sentiment_label = ttk.Label(info_frame, text="Sentiment", font=("Arial", 12, "bold"))
sentiment_label.grid(row=0, column=4, padx=5)

sentiment = ScrolledText(info_frame, height=1, width=15, state="disabled", wrap="word", font=("Arial", 12), bg='white', fg="black")
sentiment.grid(row=0, column=5, padx=5)

# Summary
summary_label = ttk.Label(result_frame, text="Summary", font=("Arial", 14, "bold"))
summary_label.pack()

summary = ScrolledText(result_frame, height=12, width=100, state="disabled", wrap="word", font=("Arial", 12), bg='white', fg="black")
summary.pack()

# Start the main event loop
root.mainloop()
