import streamlit as st
import os
import time
import numpy as np

import pickle

def app():
    st.markdown("""
        <style>
            .st-ax {
                    background-color: lightblue;
            }

            .stTextInput input{
                    background-color: lightblue;
            }

        </style>
        """, unsafe_allow_html=True)
    with open("model_file.pkl", "rb") as mf:
        new_model = pickle.load(mf)
        # Input form
        st.header("Predicition")
        with st.form("user_inputs"):
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    transactionRevenue = st.text_input("Transaction Revenue")
                    st.caption("ranges from (0 to 307221222)")
                with col2:
                    device_operatingSystem = st.text_input("Device Operating System")
                    st.caption("ranges from (0 to 6)")
                with col3:
                    time_on_site = st.text_input("Time On Site")
                    st.caption("ranges from (0 to 26652.75)")
                with col4:
                    sessionQualityDim = st.text_input("Session Quality Time")
                    st.caption("ranges from (1 to 22)")
                with col1:
                    num_interactions = st.text_input("Num Interactions")
                    st.caption("ranges from (20 to 25911)")
                with col2:
                    single_page_rate = st.text_input("Single Page Rate")
                    st.caption("ranges from (0.6 to 1)")
                with col3:
                    avg_session_time_page = st.text_input("Avg Session Time Page")
                    st.caption("ranges from (0 to 339.5)")
                with col4:
                    historic_session_page = st.text_input("Historic Session Page")
                    st.caption("ranges from (0 to 5021.25)")

            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                test_data = np.array([
                    [transactionRevenue,device_operatingSystem,time_on_site,sessionQualityDim,
                     num_interactions,single_page_rate,avg_session_time_page,historic_session_page
                    ]
                ])
                # Prediction
                predicted = new_model.predict(test_data)
                if predicted:
                    predicted_res = "Customer will Convert"
                else:
                    predicted_res = "Customer will Not Convert"

                prediction_proba = new_model.predict_proba(test_data)
                # Display the results
                st.write(":green[**Prediction:**]", predicted_res)
                st.write(":green[**Prediction Probability:**]", prediction_proba)
