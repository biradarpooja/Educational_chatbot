import os
import json
import datetime
import csv
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Bypass SSL verification for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from JSON file
intents_file_path = os.path.abspath("./main.json")
with open(intents_file_path, "r") as file:
    intents = json.load(file)

# Preprocess data for training
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Vectorizer and Classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
x_train = vectorizer.fit_transform(patterns)
y_train = tags

classifier = LogisticRegression(random_state=0, max_iter=10000)
classifier.fit(x_train, y_train)

# Chatbot response function
def chatbot_response(user_input):
    processed_input = vectorizer.transform([user_input])
    predicted_tag = classifier.predict(processed_input)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Main function for Streamlit app
def main():
    st.title("Educational Chatbot")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Chat with the Educational Chatbot")
        st.write("Ask any educational question or explore topics!")

        # Log conversation if file doesn't exist
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as log_file:
                csv_writer = csv.writer(log_file)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        user_input = st.text_input("You:")
        if user_input:
            response = chatbot_response(user_input)
            st.text_area("Chatbot:", value=response, height=150, max_chars=None)

            # Log conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as log_file:
                csv_writer = csv.writer(log_file)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.subheader("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as log_file:
                csv_reader = csv.reader(log_file)
                next(csv_reader)
                for row in csv_reader:
                    st.write(f"**You:** {row[0]}")
                    st.write(f"**Chatbot:** {row[1]}")
                    st.write(f"**Time:** {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found!")

    elif choice == "About":
        st.subheader("About the Educational Chatbot")
        st.write("""
        This chatbot is designed to assist with educational queries using 
        Natural Language Processing (NLP). It identifies user intents and 
        provides relevant responses.
        """)
        st.write("""
        **Features:**
        - Ask educational questions.
        - Review conversation history.
        - Intuitive interface built with Streamlit.
        """)

if __name__ == "__main__":
    main()
