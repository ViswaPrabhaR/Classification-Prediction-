import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pandas as pd


import NLP, Image_Processing, Sentiment_Analysis,Product_Recommendation,Data,Model_Building,Prediction

st.set_page_config(page_title="Data Analysis", page_icon="chart", layout="wide")

def set_bg_hack(main_bg):
    # set bg name
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:data/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
             
                  }}        
         </style>
         """,
        unsafe_allow_html=True
    )

set_bg_hack('bg.png')
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .st-emotion-cache-16txtl3 
    {
     padding: 0rem 1rem; 
    }
   
    </style>
    """, unsafe_allow_html=True
)
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })


    def run():
        # app = st.sidebar(
        st.sidebar.image("logo.png", use_column_width=True)
        with st.sidebar:
            app = option_menu("Title", ["EDA", 'Model Building','Prediction', 'Image Processing','NLP','Sentiment Analysis','Product Recommendation'],
         icons=['activity', 'bezier', 'person-check', 'image','chat-text','emoji-laughing','diagram-3'], menu_icon="cast", default_index=0,
                           styles={
                               "container": {"padding": "2!important", "background-color": "#fafafa" ,"border-width": "5px","border-color": "#000000",},
                               "icon": {"font-size": "14px"},
                               "menu-title":{"font-size": "18px"},
                               "menu-icon": {"font-size": "18px"},
                               "menu-link":{"font-size": "16px"},
                               "nav-link": {"font-size": "14px","text-align": "left",
                                            "--hover-color": "#fff","margin-bottom":"4px"},
                               "nav-link-selected": {"background-color": "#009584"},
                           }
                           )
        if app == "EDA":
            Data.app()
        if app == "Model Building":
            Model_Building.app()
        if app == "Prediction":
            Prediction.app()
        if app == "Image Processing":
            Image_Processing.app()
        if app == "NLP":
            NLP.app()
        if app == "Sentiment Analysis":
            Sentiment_Analysis.app()
        if app == "Product Recommendation":
            Product_Recommendation.app()

    run()
