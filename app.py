import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

import nltk

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # converting all text in lowercase
    text = nltk.word_tokenize(text)  # tokenizing on words

    y = []
    for i in text:
        if i.isalnum():  # checking for special character
            y.append(i)

    text = y[:]  # cloning the list
    y.clear()
    for i in text:  # this code is for removing stopwords and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_msg = st.text_area('Enter the message')

if st.button('Predict'):

    # 1. preprocess
    transformed_msg = transform_text(input_msg)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_msg])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header('Spam')

    else:
        st.header('Not a spam')
