## **E-commerce Customer Conversion Prediction**

**Problem Statement**

We have an E-commerce dataset and our goal is to predict whether a visitor will convert into a customer or not. The dataset contains various features, and the target variable we are interested in is the "has_converted" column. We will perform classification to determine whether a user will convert into a customer or not.

**Introduction :**

A dataset to perform exploratory data analysis (EDA), preprocessing, statistical analysis, feature selection, and model building. We will also make predictions for live stream data and display the precision, recall, accuracy, and F1-score for three different models.

We start by importing the necessary libraries: pandas, numpy, matplotlib, seaborn, streamlit, and scikit-learn modules for data analysis, visualization, and model building.

# Reading the dataset
We need to read the dataset using the pd.read_csv() function, providing the URL of the dataset.

# Exploratory Data Analysis (EDA)
Perform multiple EDA with plots and  other visualization  using  Streamlit.
We can use it to create visualizations, display data, and perform various analysis tasks in detail .

# Preprocessing
After performing EDA, we move on to preprocessing the data. This step involves handling missing values, encoding categorical variables, scaling features, and other necessary preprocessing steps.

# Statistical Analysis
We next perform statistical analysis to gain insights into the data, such as calculating descriptive statistics or performing hypothesis testing or conducting other statistical tests.

# Feature Selection
Feature selection is an important step in model building. It involves selecting the most relevant features for the classification task. One common technique is to use the SelectKBest method, it is important to scale the features using the StandardScaler to ensure all features have the same scale.

# Train-Test Split
Before training the models, We split the data into training and testing sets using the train_test_split() function from scikit-learn.

# Model Building
We build  3 models using different algorithms such as Logistic Regression, KNN, and Random Forest.These models will be used for prediction and evaluation.

# Model Evaluation
After training the models, we evaluate their performance using precision, recall, accuracy, and F1-score. These metrics provide insights into how well the models are performing.

# Live Stream Data Prediction
	Once the models are trained and evaluated, we can use them to make predictions on live stream data. This allows us to predict whether a customer will convert or not in real-time.

# Display Model Metrics
	Finally, we calculate the precision, recall, accuracy, and F1-score for each model. Display the results using Streamlit.

