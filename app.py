import streamlit as st
import spacy
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import requests  
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
import pytz

nltk.download('stopwords')

# Load dataset
data = pd.read_excel('./IES_dataset(1) final.xlsx')
train_texts, test_texts, train_labels, test_labels = train_test_split(data['Input'], data['Label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Convert text data to TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()
X_test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

# Convert labels to one-hot encoded vectors
train_labels_onehot = to_categorical(train_labels)
test_labels_onehot = to_categorical(test_labels)

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    tweet_sentences = ' '.join(tweets_clean)
    return tweet_sentences

def predict_class(text):
    functions_list = {
        0: 'CreateInventory',
        1: 'AddEstimate',
        2: 'NewSales',
        3: 'AddExpense',
        4: 'GetExpense',
        5: 'GetUserReport'
    }

    loaded_model = tf.keras.models.load_model("./text_classification_model.h5")

    processed_text = process_tweet(text)

    text_vectorized = tfidf_vectorizer.transform([processed_text]).toarray()

    predictions = loaded_model.predict(text_vectorized)

    predicted_class_index = np.argmax(predictions[0])
    return functions_list[predicted_class_index]

def load_model():
    return spacy.load("./model-best")

def perform_ner(text, nlp_model, predicted_class):
    doc = nlp_model(text)
    entity_labels = {"price": "sellingPrice", "amount": "quantity"}  # Mapping from original entity labels to modified labels
    
    if predicted_class == "CreateInventory":
        entity_labels = {"price": "sellingPrice", "amount": "quantity"}
    elif predicted_class == "AddExpense":
        entity_labels = {"price": "amount"}
        
    modified_entities = {}
    for ent in doc.ents:
        label = ent.label_.lower()
        text = ent.text
        modified_label = entity_labels.get(label, label)  # Use modified label if available, otherwise use original label
        if modified_label not in modified_entities:
            if modified_label == "quantity":
                # Extract numerical value from the text
                numerical_value = re.findall(r'\d+', text)
                if numerical_value:
                    text = int(numerical_value[0])
            if modified_label == "name":
                modified_label = "product"  # Update label to "product" if it's "name"
            modified_entities[modified_label] = text

    return modified_entities

def ask_follow_up_questions(predicted_class):
    follow_up_data = {}
    if predicted_class == 'CreateInventory':
        name = st.text_input("Please enter the product name:", key="name_input")
        quantity = st.number_input("Please enter the quantity:", key="quantity_input")
        selling_price = st.text_input("Please enter the selling price:", key="selling_price_input")
        if name and quantity and selling_price:
            # Extract numerical value from the selling price
            numerical_price = re.findall(r'\d+', selling_price)
            if numerical_price:
                follow_up_data = {"name": name, "quantity": quantity, "sellingPrice": int(numerical_price[0])}
                api_response = post_data_to_api(follow_up_data)  
                st.write("API Response:", api_response)  
            else:
                st.error("Please enter a valid selling price.")
    return follow_up_data

def post_data_to_api(data):
    api_endpoint = "https://testbackend-u2af.onrender.com/api/v1/inventory/new"  
    bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1ZWYxOGEwNjVjZGRlNzhjZjhmNDJiYyIsImlhdCI6MTcxMDc1NTg2NCwiZXhwIjoxNzExMTg3ODY0fQ.Xsd7FK5tH7m7PjVxnVXb1RgDc8kwfpz66F21pnjpp7Y"  
    
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_endpoint, headers=headers, json=data)
    if response.status_code == 200 or response.status_code == 201:
        return response
    else:
        error_message = response.json().get('error', 'Unknown error occurred.')
        st.error(f"Failed to send data to API. Error {response.status_code}: {error_message}")


def create_expense(data):
    api_endpoint = "https://testbackend-u2af.onrender.com/api/v1/add/expense"  
    bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1ZWYxOGEwNjVjZGRlNzhjZjhmNDJiYyIsImlhdCI6MTcxMDc1NTg2NCwiZXhwIjoxNzExMTg3ODY0fQ.Xsd7FK5tH7m7PjVxnVXb1RgDc8kwfpz66F21pnjpp7Y"  
    
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_endpoint, headers=headers, json=data)
    if response.status_code == 200 or response.status_code == 201:
        return response
    else:
        error_message = response.json().get('error', 'Unknown error occurred.')
        st.error(f"Failed to send data to API. Error {response.status_code}: {error_message}")

def get_start_end_dates(date):
    local_timezone = pytz.timezone('Asia/Kolkata')

    if date == "today":
        start_date = datetime.now(local_timezone).replace(hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_date = datetime.now(local_timezone).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    elif date == "yesterday":
        yesterday = datetime.now(local_timezone) - timedelta(days=1)
        start_date = datetime.combine(yesterday.date(), datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_date = yesterday.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    elif date == "this week":
        now = datetime.now(local_timezone)
        start_of_current_week = now - timedelta(days=now.weekday())
        end_of_current_week = start_of_current_week + timedelta(days=6)
        start_date = start_of_current_week.replace(hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_date = end_of_current_week.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    elif date == "last week":
        now = datetime.now(local_timezone)
        start_of_current_week = now - timedelta(days=now.weekday())
        end_of_current_week = start_of_current_week - timedelta(days=1)
        start_of_previous_week = end_of_current_week - timedelta(days=6)
        end_of_previous_week = end_of_current_week
        start_date = start_of_previous_week.replace(hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_date = end_of_previous_week.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    else:
        start_date = end_date = None

    return start_date, end_date

def get_report(data):
    
    date = data.get("date")
    bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1ZWYxOGEwNjVjZGRlNzhjZjhmNDJiYyIsImlhdCI6MTcxMDc1NTg2NCwiZXhwIjoxNzExMTg3ODY0fQ.Xsd7FK5tH7m7PjVxnVXb1RgDc8kwfpz66F21pnjpp7Y"  
    start_date, end_date = get_start_end_dates(date)
    
    if start_date and end_date:
        api_endpoint = f'https://testbackend-u2af.onrender.com/api/v1/report?start_date={start_date}&end_date={end_date}&type=expense'
        
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(api_endpoint, headers=headers)
        if response.status_code == 200:
            return response.json()  # Assuming the API returns JSON data
        else:
            return f"Failed to fetch report. Error: {response.status_code}"
    else:
        return "Invalid date provided"


def main():
    nlp_model = load_model()

    text_input = st.text_area("Enter text:", "")
    
    if st.button("Submit"):
        if text_input.strip() == "":
            st.error("Please enter some text.")
        else:
            class_predicted = predict_class(text_input)

            if 'data' not in st.session_state:
                st.session_state.data = {"entities": {}}

            modified_entities = perform_ner(text_input, nlp_model, class_predicted)

            for label, text in modified_entities.items():
                if label == "name":
                    label = "product"
                if label == "sellingPrice":
                    # Extract numerical value from the selling price
                    numerical_price = re.findall(r'\d+', text)
                    if numerical_price:
                        text = int(numerical_price[0])
                if label == "amount":
                    # Extract numerical value from the amount
                    numerical_amount = re.findall(r'\d+', text)
                    if numerical_amount:
                        text = int(numerical_amount[0])
                st.session_state.data["entities"][label] = text

            if st.session_state.data["entities"]:
                st.write("Entities found:")
                for label, text in st.session_state.data["entities"].items():
                    st.write(f"- {label}: {text}")
            else:
                st.info("No entities found.")


            if class_predicted == "CreateInventory":
                api_response = post_data_to_api(st.session_state.data["entities"])
                if "error" in api_response:
                    st.error(api_response["error"])
                else:
                    st.success("Inventory created successfully")
                    
                    
            if class_predicted == "AddExpense":
                api_response = create_expense(st.session_state.data["entities"])
                if "error" in api_response:
                    st.error(api_response["error"])
                else:
                    st.success("Expense created successfully")
            
            if class_predicted == "GetUserReport":
                api_response = get_report(st.session_state.data["entities"])
                if "error" in api_response:
                    st.error(api_response["error"])
                else:
                    st.write("Report:", api_response)

            st.write("Data:", st.session_state.data["entities"])
            st.write("Class:", class_predicted)

if __name__ == "__main__":
    main()
