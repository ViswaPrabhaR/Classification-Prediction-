from textblob import TextBlob
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

def app():
    st.markdown(
        """
         <style>
    .stTextArea textarea {        
        background-color:#FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Sentiment Analysis using Text Blob")
    userText = st.text_area('Enter the text to be analysed ', placeholder='Input the text here')
    st.text(' ')
    if st.button('Predict'):
        if (userText != ''):
            st.header(":green[Analysis - Result]")
            getSentiment(userText)


# Write user define function getSentiment
def getSentiment(userText):
    polarity, subj, status = getPolarity(userText)
    if (status == "Positive"):
        image = Image.open('data/Sentiment_Analysis/positive.png')
    elif (status == 'Negative'):
        image = Image.open('data/Sentiment_Analysis/negative.png')
    else:
        image = Image.open('data/Sentiment_Analysis/neutral.png')
    st.markdown('''<style>
                [data-testid="column"] 
    {
    box-shadow: rgb(0 0 0 / 30%) 0px 2px 1px -1px, rgb(0 0 0 / 24%) 0px 1px 1px 0px, rgb(0 0 0 / 22%) 0px 1px 3px 0px;
    border-radius: 15px;
    padding: 2% 2% 1% 3%;
    align:center;
} </style>''', unsafe_allow_html=True)
    with st.container ():
        col1, col2, col3, col4= st.columns(4)
        col1.metric("**Polarity**", polarity, None)
        col2.metric("**Subjectivity**", subj, None)
        col3.metric("**Result**", status, None)
        col4.image(image)
def getPolarity(userText):
    tb = TextBlob(userText)
    polarity = round(tb.polarity, 2)
    subj = round(tb.subjectivity, 2)
    if polarity > 0:
        return polarity, subj, "Positive"
    elif polarity == 0:
        return polarity, subj, "Neutral"
    else:
        return polarity, subj, "Negative"
