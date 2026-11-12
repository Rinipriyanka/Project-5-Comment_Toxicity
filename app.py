import tensorflow as tf
import pickle
import json
import numpy as np
import pandas as pd


model = tf.keras.models.load_model(
    "toxicity_model.keras",
    compile=False
)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

print("Model loaded successfully!")
print("Tokenizer loaded successfully!")


with open("config.json", "r") as file:
    config = json.load(file)

print("Configuration loaded successfully!")
print(config)

import re

def clean_text(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(
        r"</?[A-Za-z][A-Za-z0-9]*(?:\s+[^<>]*)?\s*/?>",
        " ",
        text
    )

    text = re.sub(r"&#x20;|&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
sample = "You are <b>great</b>! Visit https://example.com"
print(clean_text(sample))

sample_sequence = tokenizer.texts_to_sequences([clean_text(sample)])

print("Tokenized sequence:", sample_sequence)

from tensorflow.keras.preprocessing.sequence import pad_sequences

sample_padded = pad_sequences(
    sample_sequence,
    maxlen=config["max_len"],
    padding="post",
    truncating="post"
)

print("Padded sequence shape:", sample_padded.shape)
prediction = model.predict(sample_padded)
print("Prediction probabilities:", prediction)

thresholds = config["thresholds"]
predicted_labels = (prediction[0] >= thresholds).astype(int)
print("Predicted labels:", predicted_labels)

results = {}
for i, label in enumerate(config["target_columns"]):
    results[label] = "Yes" if predicted_labels[i] == 1 else "No"
print("Toxicity results:", results)

def predict_comments(comments):

    cleaned_comments = [
        clean_text(comment) for comment in comments
    ]

    sequences = tokenizer.texts_to_sequences(cleaned_comments)

    padded_sequences = pad_sequences(
        sequences,
        maxlen=config["max_len"],
        padding="post",
        truncating="post"
    )

    empty_rows = np.all(padded_sequences == 0, axis=1)
    padded_sequences[empty_rows, 0] = 1

    probabilities = model.predict(padded_sequences, verbose=0)

    predicted_labels = (
        probabilities >= np.array(config["thresholds"])
    ).astype(int)

    return probabilities, predicted_labels

#Streamlit
import streamlit as st
st.title("Comment Toxicity Detection")
train_df = pd.read_csv("train.csv")

page = st.sidebar.radio(
    "Select Option",
    ["Dashboard", "Single Comment Detection", "Bulk CSV Prediction"]
)

if page == "Dashboard":
    st.subheader("Dataset Overview")


    toxic_count = train_df["toxic"].sum()
    non_toxic_count = len(train_df) - toxic_count
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Comments", len(train_df))
    with col2:
        st.metric("Toxic Comments", toxic_count)
    with col3:
        st.metric("Non-Toxic Comments", non_toxic_count)

    st.subheader("Overall Model Performance")
        
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Micro F1-score", "0.73")
    col2.metric("Weighted F1-score", "0.74")
    col3.metric("Macro F1-score", "0.58")
    col4.metric("Micro Recall", "0.76")

    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 26px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.subheader("Toxicity Category Distribution")

    label_counts = train_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].sum()
    st.pyplot(label_counts.plot(kind="bar").get_figure())
    st.subheader("Sample Test Cases")
    sample_comments = [
        "You are wonderful!",
        "Thank you for your help.",
        "You are an idiot!",
        "I hate you."
    ]
    for comment in sample_comments:
        st.write("•", comment)

    if st.button("Test Sample Comments"):

        probabilities, predictions = predict_comments(sample_comments)
        st.subheader("Sample Prediction Results")
        results_df = pd.DataFrame(
            predictions,
            columns=config["target_columns"]
        )
        results_df.insert(0, "Comment", sample_comments)
        results_df = results_df.replace({1: "Yes", 0: "No"})
        st.dataframe(results_df, use_container_width=True)

    
if page == "Single Comment Detection":
    st.subheader("Enter your comment")
    user_comment = st.text_area("Write your comment here:")
    if st.button("Predict Toxicity"):

        if user_comment.strip():

            prediction_prob, predicted_labels = predict_comments(
                [user_comment]
            )

            st.write("Prediction completed!")
            if predicted_labels[0].any():
                st.error("Toxic content detected!")
            else:
                st.success("No toxicity detected!")
            st.subheader("Detailed Toxicity Analysis")

            for i, label in enumerate(config["target_columns"]):

                result = "Yes" if predicted_labels[0][i] == 1 else "No"

                st.write(f"{label.replace('_', ' ').title()}: {result}")

        else:
            st.warning("Please enter a comment.")
elif page == "Bulk CSV Prediction":

    st.subheader("Bulk Comment Toxicity Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )
    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded CSV Preview")
        st.dataframe(df.head())

        if "comment_text" in df.columns:

                comments = df["comment_text"].fillna("").astype(str).tolist()[:100]
                
                probabilities, predictions = predict_comments(comments)

                st.success("Predictions completed!")

                for i, label in enumerate(config["target_columns"]):
                    df.loc[:len(predictions)-1, label] = predictions[:, i]

                st.subheader("Bulk Prediction Results")
                st.dataframe(df.head(10))
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please upload a CSV containing a 'comment_text' column.")