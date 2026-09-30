import streamlit as st  #for creating interface
from transformers import pipeline   #hugging face pipeline(simple model interface) for sentiment analysis

#congif page 
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

#app title and description
st.title("💬 Sentiment Analyzer")
st.write("Enter a sentence or paragraph below, and the app will analyze the sentiment of the text.")

#load the hugging face model only once
@st.cache_resource     #prevents reloading everytime app runs
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  #loads pretrained sentiment model

#load model
with st.spinner("Loading AI model..."):
    sentiment_model = load_model()

#text input
user_text = st.text_area("Enter your text here:", placeholder="I absolutely love this product!", height=150)  #creates text area box for user input

#analyze button
if st.button("Analyze Sentiment", type = "primary"):     #analyze sentiment button
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = sentiment_model(user_text)[0]     #sends input to ai model

            #retrieves models predicted outputs
            sentiment = result['label']      
            confidence = result['score']

            #display results
            if sentiment == "POSITIVE":
                st.success("Sentiment: Positive 😊")
            else:
                st.error("Sentiment: Negative 😞")
            st.write(f"**Confidence Score:** {confidence:.2%}")
            st.progress(confidence)