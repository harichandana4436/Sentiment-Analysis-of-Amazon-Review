import streamlit as st
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# OPTIONAL: load label encoder (VERY IMPORTANT if you saved it)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


# App title
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")
st.title("📝 Sentiment Analysis App")

st.write("Enter a product review and get sentiment prediction")

# User input
user_input = st.text_area("Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        # Transform input
        text_vec = vectorizer.transform([user_input])

        # Predict (returns number: 0/1/2)
        prediction = model.predict(text_vec)[0]

        # Convert number back to label
        pred_label = le.inverse_transform([prediction])[0]

        # Display result
        st.subheader("Result:")

        if pred_label == "Positive":
            st.success("😊 Positive Review")

        elif pred_label == "Negative":
            st.error("😠 Negative Review")

        else:
            st.info("😐 Neutral Review")