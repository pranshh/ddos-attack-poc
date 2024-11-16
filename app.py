import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import necessary ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GRU, Dropout
from keras.optimizers import Adam

# Streamlit App Starts
st.set_page_config(page_title="DDOS Attack Analysis", layout="wide")

st.title("DDOS Attack Analysis")
st.write("Upload a CSV dataset, and this app will process it, generate visualizations, and evaluate multiple ML models.")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV File", type="csv")

if uploaded_file:
    # Load the dataset
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    # Dataset Overview
    st.write("### Dataset Overview")
    st.text(f"Shape: {df.shape}")
    st.text("Column Information:")
    st.text(df.info())

    st.write("### Missing Values")
    st.text(df.isnull().sum())

    st.write("### Descriptive Statistics")
    st.write(df.describe())

    # Categorical Column Visualization
    st.write("### Categorical Column Visualization")
    categorical_column = st.selectbox("Select a categorical column to visualize:", df.select_dtypes(include=['object']).columns)
    if categorical_column:
        with st.spinner(f"Plotting distribution for {categorical_column}..."):
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette="viridis")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Distribution of {categorical_column}")
            st.pyplot(plt)

    # Numerical Columns Visualization
    st.write("### Numerical Column Distribution")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_numerical_col = st.selectbox("Select a numerical column to visualize:", numerical_columns)
    if selected_numerical_col:
        with st.spinner(f"Plotting distribution for {selected_numerical_col}..."):
            plt.figure(figsize=(6, 4))
            sns.histplot(df[selected_numerical_col], kde=True, bins=50, color="blue")
            plt.title(f"Distribution of {selected_numerical_col}")
            st.pyplot(plt)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    with st.spinner("Generating heatmap..."):
        plt.figure(figsize=(16, 10))
        correlation = df.select_dtypes(include=['float64', 'int64']).corr()
        sns.heatmap(correlation, annot=False, cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap")
        st.pyplot(plt)

    # Data Preprocessing
    st.write("### Data Preprocessing")
    with st.spinner("Encoding categorical columns..."):
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    target_column = st.selectbox("Select the target column:", df.columns)
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        st.write("Splitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("Scaling the data...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Run Models
    st.write("### Model Evaluation")

    def evaluate_model(name, model, X_train, y_train, X_test, y_test):
        with st.spinner(f"Training and evaluating {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            st.write(f"#### {name} Results")
            st.text(f"Accuracy: {accuracy:.4f}")
            st.text(f"F1 Score: {f1:.4f}")
            st.text(f"Recall: {recall:.4f}")
            st.text(f"Precision: {precision:.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    # Random Forest
    evaluate_model("Random Forest", RandomForestClassifier(random_state=42), X_train, y_train, X_test, y_test)

    # KNN
    evaluate_model("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test)

    # Logistic Regression
    evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train, y_train, X_test, y_test)

    # Decision Tree
    evaluate_model("Decision Tree", DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test)

    # Naive Bayes
    evaluate_model("Naive Bayes", GaussianNB(), X_train, y_train, X_test, y_test)

    # XGBoost
    evaluate_model("XGBoost", xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y_train)), random_state=42), X_train, y_train, X_test, y_test)

    # Neural Network
    with st.spinner("Training Neural Network..."):
        nn_model = Sequential([
            Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(64, activation="relu"),
            Dense(len(set(y_train)), activation="softmax")
        ])
        nn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        nn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)  # Fewer epochs for simplicity
        loss, accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
        st.write("#### Neural Network Results")
        st.text(f"Accuracy: {accuracy:.4f}")
        y_pred_nn = nn_model.predict(X_test).argmax(axis=1)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_nn))
