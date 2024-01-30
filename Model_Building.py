import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, accuracy_score,recall_score,auc,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd



def app():

    st.header("Feature Selection")

    data = st.session_state.df
    data_data = data.drop("has_converted", axis=1)
    lab = data['has_converted']
    st.caption(":green[***RandomForestClassifier***]")
    rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
    rf_model.fit(data_data, lab)
    rf_model.feature_importances_ * 100
    fs= pd.DataFrame({
        "columns": data_data.columns,
        "imp_scor": rf_model.feature_importances_ * 100
    }).sort_values("imp_scor", ascending=False).head(8)['columns'].to_list()
    col1,col2 = st.columns([2,3])
    with col1:
        st.write(fs)

    data_1 = data[['transactionRevenue',
 'device_operatingSystem',
 'time_on_site',
 'sessionQualityDim',
 'num_interactions',
 'single_page_rate',
 'avg_session_time_page',
 'historic_session_page', 'has_converted']]
    with col2:
        st.write(data_1.head(5))
    st.caption(":green[***Standard Scaler***]")
    scaler = StandardScaler(with_mean=True)
    to_scale = data_1[['transactionRevenue',
 'device_operatingSystem',
 'time_on_site',
 'sessionQualityDim',
 'num_interactions',
 'single_page_rate',
 'avg_session_time_page',
 'historic_session_page']
]
    scaler.fit(to_scale)
    scaled_features = scaler.transform(to_scale)
    y = data['has_converted']
    data_feat = pd.DataFrame(scaled_features, columns=['transactionRevenue',
 'device_operatingSystem',
 'time_on_site',
 'sessionQualityDim',
 'num_interactions',
 'single_page_rate',
 'avg_session_time_page',
 'historic_session_page']
)
    st.write(data_feat.head())
    x = abs(data.drop('has_converted', axis=1))  # Data
    y = data['has_converted']  # Target

    st.caption(":green[***SelectKBest***]")

    selection = SelectKBest(chi2, k=10)
    data_select = selection.fit_transform(x, y)
    best = pd.DataFrame({
        "columns": x.columns,
        "chi-sq-value": selection.scores_
    }).sort_values('chi-sq-value', ascending=False).head(8)
    col1,col2=st.columns([2,3])
    with col2:
        st.write(best)
    with col1:
        st.write(list(best['columns']))

    st.header("Model Building")
    st.subheader(":green[***Random Forest***]")
    X = data[['transactionRevenue',
 'device_operatingSystem',
 'time_on_site',
 'sessionQualityDim',
 'num_interactions',
 'single_page_rate',
 'avg_session_time_page',
 'historic_session_page']
]
    y = data['has_converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    rf_model = RandomForestClassifier(n_estimators=501)
    rf_model.fit(X_train.values, y_train.values)


    rf_train_predict = rf_model.predict(X_train.values)
    rf_accuracy = accuracy_score(y_train.values, rf_train_predict)
    rf_Precision = precision_score(y_train.values, rf_train_predict)
    rf_recall = recall_score(y_train.values, rf_train_predict)
    rf_f1 = f1_score(y_train.values, rf_train_predict)

    rf_test_predict = rf_model.predict(X_test.values)
    rf_test_accuracy = accuracy_score(y_test.values, rf_test_predict)
    rf_test_Precision = precision_score(y_test.values, rf_test_predict)
    rf_test_recall = recall_score(y_test.values, rf_test_predict)
    rf_test_f1 = f1_score(y_test.values, rf_test_predict)

    with (st.container(border=True)):
        col1, col2 = st.columns(2)

        with col1:
            # Display Random Forest Model results
            st.caption(":green[***Training Data Classification Report:***]")
            st.write(f"Accuracy  : {rf_accuracy:.2f}")
            st.write(f"Precision : {rf_Precision:.2f}")
            st.write(f"Recall    : {rf_recall:.2f}")
            st.write(f"F1_score  : {rf_f1:.2f}")
            st.caption(":green[***Training Data Confusion Report:***]")
            st.write(confusion_matrix(y_train, rf_train_predict))

        with col2:
            # Display Random Forest Model results
            st.caption(":green[***Testing Data Classification Report:***]")
            st.write(f"Accuracy  : {rf_test_accuracy:.2f}")
            st.write(f"Precision : {rf_test_Precision:.2f}")
            st.write(f"Recall    : {rf_test_recall:.2f}")
            st.write(f"F1_score  : {rf_test_f1:.2f}")

            st.caption(":green[***Testing Data Confusion Report:***]")
            st.write(confusion_matrix(y_test,rf_test_predict))

    st.subheader(":green[***Logistic Regression***]")
    X = data_feat
    y = data['has_converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logm = LogisticRegression()
    logm.fit(X_train, y_train)
    train_pred = logm.predict(X_train)
    test_pred = logm.predict(X_test)

    logm_train_predict = logm.predict(X_train.values)
    logm_accuracy = accuracy_score(y_train.values, logm_train_predict)
    logm_Precision = precision_score(y_train.values, logm_train_predict)
    logm_recall = recall_score(y_train.values, logm_train_predict)
    logm_f1 = f1_score(y_train.values, logm_train_predict)

    logm_test_predict = logm.predict(X_test.values)
    logm_test_accuracy = accuracy_score(y_test.values, logm_test_predict)
    logm_test_Precision = precision_score(y_test.values, logm_test_predict)
    logm_test_recall = recall_score(y_test.values, logm_test_predict)
    logm_test_f1 = f1_score(y_test.values, logm_test_predict)
    with (st.container(border=True)):
        col1, col2 = st.columns(2)

        with col1:
            # Display LOGISTIC REGRESSION Model results
            st.caption(":green[***Training Data Classification Report:***]")
            st.write(f"Accuracy  : {logm_accuracy:.2f}")
            st.write(f"Precision : {logm_Precision:.2f}")
            st.write(f"Recall    : {logm_recall:.2f}")
            st.write(f"F1_score  : {logm_f1:.2f}")
            st.caption(":green[***Training Data Confusion Report:***]")
            st.write(confusion_matrix(y_train, rf_train_predict))

        with col2:
             # Display LOGISTIC REGRESSION Model results
            st.caption(":green[***Testing Data Classification Report:***]")
            st.write(f"Accuracy  : {logm_test_accuracy:.2f}")
            st.write(f"Precision : {logm_test_Precision:.2f}")
            st.write(f"Recall    : {logm_test_recall:.2f}")
            st.write(f"F1_score  : {logm_test_f1:.2f}")
            st.caption(":green[***Testing Data Confusion Report:***]")
            st.write(confusion_matrix(y_test, rf_test_predict))

    st.subheader(":green[***KNeighbors***]")
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    knn_train_pred = knn.predict(X_train)
    knn_test_pred = knn.predict(X_test)

    knn_train_predict = knn.predict(X_train.values)
    knn_accuracy = accuracy_score(y_train.values, knn_train_pred)
    knn_Precision = precision_score(y_train.values, knn_train_pred)
    knn_recall = recall_score(y_train.values, knn_train_pred)
    knn_f1 = f1_score(y_train.values, knn_train_pred)

    knn_test_predict = knn.predict(X_test.values)
    knn_test_accuracy = accuracy_score(y_test.values, knn_test_predict)
    knn_test_Precision = precision_score(y_test.values, knn_test_predict)
    knn_test_recall = recall_score(y_test.values, knn_test_predict)
    knn_test_f1 = f1_score(y_test.values, knn_test_predict)

    with (st.container(border=True)):
        col1, col2 = st.columns(2)

        with col1:
            # Display KNeighbors Model results
            st.caption(":green[***Training Data Classification Report:***]")
            st.write(f"Accuracy  : {knn_accuracy:.2f}")
            st.write(f"Precision : {knn_Precision:.2f}")
            st.write(f"Recall    : {knn_recall:.2f}")
            st.write(f"F1_score  : {knn_f1:.2f}")
            st.caption(":green[***Training Data Confusion Report:***]")
            st.write(confusion_matrix(y_train, knn_train_predict))

        with col2:
            # Display KNeighbors Model results
            st.caption(":green[***Testing Data Classification Report:***]")
            st.write(f"Accuracy  : {knn_test_accuracy:.2f}")
            st.write(f"Precision : {knn_test_Precision:.2f}")
            st.write(f"Recall    : {knn_test_recall:.2f}")
            st.write(f"F1_score  : {knn_test_f1:.2f}")

            st.caption(":green[***Testing Data Confusion Report:***]")
            st.write(confusion_matrix(y_test, knn_test_predict))

    st.subheader("***Conclusion***")
    # Display results in a table
    results_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'KNeighbors'],
        'Accuracy': [rf_accuracy, logm_accuracy, knn_accuracy],
        'Precision': [rf_Precision, logm_Precision, knn_Precision],
        'Recall': [rf_recall, logm_recall, knn_recall],
        'F1_score': [rf_f1, logm_f1, knn_f1]
    }

    results_table = st.table(results_data)
    st.write("We achieved a good performance in **Random Forest Model**. "
             "KNN model  performs well with training data and testing data with a accuracy of 93%.")