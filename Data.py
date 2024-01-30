import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score,recall_score,auc
import seaborn as sns
import streamlit as st
import os
import time

def app():

    st.header("E-commerce Customer Conversion Prediction")
    with (st.container(border=True)):
        st.caption(":green[***Read Dataset***]")
        # Step 1: Load CSV File
        uploaded_file = st.file_uploader("Upload Given Data File", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
             # Save the data in the session state
            with open(os.path.join("data/Prediction_Classifcation", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                st.success(f"{uploaded_file.name} - File Saved Successfully!", icon="✅")

        # Display DataFrame
            if data is not None:
                st.caption(":green[***Data Frame***]")
                st.write(data.sample(5))


                # Data Visualization
                st.subheader("Data Visualization & Pre-Processing")
                st.caption(":green[***Data Geo Visualization***]")
                fig = px.scatter_geo(
                    data, lat="geoNetwork_latitude",
                    lon="geoNetwork_longitude", projection="natural earth")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(":green[***Avg_Session_Time***]")
                st.line_chart(data.groupby('geoNetwork_region')['avg_session_time'].mean(), y='avg_session_time')


            # Drop Duplicates
                st.caption(":green[***Check Duplicate Value***]")
                col1, col2 = st.columns(2)
                with col1:
                    duplicate = data.duplicated().value_counts()
                    st.write(duplicate)
                with col2:
                    dup_counts = data.duplicated().sum()
                    st.write("Overall Data points", data.shape[0])
                    st.write("Duplicated Rows ", dup_counts)
                    data_1 = data.drop_duplicates(inplace=True)
                    st.success("Duplicates values dropped successfully!")

            # Drop NaN Values
                col1 , col2 = st.columns(2)
                with col1:
                    st.caption(":green[***Check NAN Value***]")
                    null_data = (data.isnull().mean() * 100).round(2)
                    st.write(null_data)
                with col2:
                    st.caption(":green[***Check 0 Value***]")
                    # Finding "0"
                    col_with_zero = []
                    for i in data.columns:
                        perc_zero = (data[i] == 0).mean() * 100
                        col_with_zero.append((i, perc_zero))
                        z_df = pd.DataFrame(col_with_zero,
                                            columns=["column_name", "zero_perc"]).sort_values(
                                                       "zero_perc",ascending=False)
                    st.write(z_df)
                st.write(data.describe().transpose())

                #category Variables Check
                st.subheader("Values Counting Data in Categoricals")
                cats = ['channelGrouping', 'device_browser', 'device_operatingSystem',
                        'device_deviceCategory', 'geoNetwork_region','earliest_source','latest_source','earliest_medium','latest_medium']
                features = cats
                col1, col2 =st.columns(2)
                fig = plt.figure(figsize=(9, 7))
                with col1:
                    for col in cats:
                        st.caption(f''':green[***Value count Colums {col}:***]''')
                        st.write(data[col].value_counts())
                        st.write()
                with col2:
                    for col in cats:
                        data_plot = data[col].value_counts()
                        color = sns.color_palette("husl", len(data_plot))
                        data_plot.plot(kind="bar", color=color)
                        st.pyplot(fig)



                # correlation
                st.subheader("Stat Analysis")
                st.caption(":green[***Correlation***]")
                corr = data.corr(numeric_only=True)
                st.write(corr)
                data.drop(["latest_visit_id", "last_visitId","target_date","visitId_threshold"], axis=1, inplace=True)

                fig = plt.figure(figsize=(15, 10))
                data.boxplot(vert=0)
                st.pyplot(fig)
                st.write("From the above heatmap we can find the highly correlated variables and drop the highly correlated variable for feature selection")

                # Outlier Detection
                st.caption(":green[***Outlier Detection***]")
                num = []
                for i in data.columns:
                    if data[i].dtype != "object":
                        num.append(i)
                fig = plt.figure(figsize=(10, 20))
                for i in range(0, len(num)):
                    plt.subplot(8, 4, i + 1)
                    sns.boxplot(data[num[i]], color='skyblue', orient='v')
                    plt.xlabel(num[i])
                    plt.tight_layout()
                st.pyplot(fig)
                st.write("Dropping the column which are are no use")
                data.drop(["youtube", "days_since_last_visit"], axis=1, inplace=True)
                st.write(data.shape)

                #Changing the datatypes
                data["latest_isTrueDirect"] = data["latest_isTrueDirect"].astype("object")
                data["earliest_isTrueDirect"] = data["earliest_isTrueDirect"].astype("object")
                data["device_isMobile"] = data["device_isMobile"].astype("object")
                data["totals_newVisits"] = data["totals_newVisits"].astype("object")

                st.caption(":green[***Treating Outliers***]")
                outliers = ["transactionRevenue", "time_on_site", "bounces", "num_interactions", "bounce_rate",
                            "visits_per_day", "days_since_first_visit", "avg_visit_time", "latest_visit_number",
                            "earliest_visit_number", "sessionQualityDim", "earliest_visit_id", "single_page_rate",
                            "avg_session_time_page", "avg_session_time", "historic_session_page", "historic_session",
                            "geoNetwork_longitude", "geoNetwork_latitude", "count_hit", "count_session"]

                def remove_outlier(col):
                    sorted(col)
                    Q1, Q3 = np.percentile(col, [25, 75])
                    IQR = Q3 - Q1
                    lower_range = Q1 - (1.5 * IQR)
                    upper_range = Q3 + (1.5 * IQR)
                    return lower_range, upper_range

                for column in outliers:
                    lr, ur = remove_outlier(data[column])
                    data[column] = np.where(data[column] > ur, ur, data[column])
                    data[column] = np.where(data[column] < lr, lr, data[column])

                fig = plt.figure(figsize=(10, 20))
                for i in range(0, len(outliers)):
                    plt.subplot(8, 4, i + 1)
                    sns.boxplot(data[outliers[i]], palette='Greens', orient='v')
                    plt.xlabel(outliers[i])
                    plt.tight_layout()
                st.pyplot(fig)

                for x in data.columns:
                    if data[x].dtype == "object":
                        data[x] = pd.Categorical(data[x]).codes

                #Target Variable
                st.caption(":green[***Target Variable***]")
                data.drop(["products_array"], axis=1, inplace=True)
                fig = plt.figure(figsize=(20, 5))
                sns.barplot(data=data, x=data["device_browser"], y=data["has_converted"])
                st.pyplot(fig)


                st.caption(":green[***Normalization***]")
                col_name = data.columns
                fig1 = plt.figure(figsize=(10, 20))
                for i in range(0, len(col_name)):
                    plt.subplot(10, 4, i + 1)
                    sns.kdeplot(data = data[col_name[i]], fill=True, color='g')
                    plt.xlabel(col_name[i])
                    plt.tight_layout()
                st.pyplot(fig1)
                st.write("We converted all datatypes into numerical format. Next Feature Selection and Model Building")

                if data not in st.session_state:
                    # Get the data if you haven't
                    df = data
                    # Save the data to session state
                    st.session_state.df = df


























