import autogen
from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import traceback

# OpenRouter API Configuration
base_url = "https://openrouter.ai/api/v1"
api_key = "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"

config_list = [{"model": "openai/gpt-3.5-turbo", "api_key": api_key, "base_url": base_url}]
llm_config = {"config_list": config_list}

UPLOAD_PATH = "uploaded_dataset.csv"  # Path where uploaded file will be saved

# ----------------------- DATA FUNCTIONS -----------------------

def summarize_data(dataset_path):
    try:
        df = pd.read_csv(dataset_path, encoding="latin1")
        columns = df.columns.tolist()
        summary = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(columns)}."
        return summary, df
    except Exception as e:
        return f"Error loading dataset: {e}", None

def preprocess_data(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

def select_best_model(df):
    target_column = df.columns[-1]
    y_unique = df[target_column].nunique()
    
    if y_unique <= 10:
        return [LogisticRegression(), RandomForestClassifier(), SVC()]
    else:
        return [LinearRegression(), RandomForestRegressor(), SVR()]

def train_and_test(df):
    df = preprocess_data(df)
    target_column = df.columns[-1]
    X, y = df.iloc[:, :-1], df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = select_best_model(df)
    best_model, best_score = None, -np.inf

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if df[target_column].nunique() <= 10:
            score = accuracy_score(y_test, predictions)
        else:
            score = -mean_squared_error(y_test, predictions)

        if score > best_score:
            best_score, best_model = score, model

    return best_model, best_score

# ----------------------- GRAPH GENERATION -----------------------

def generate_and_display_graphs(df):
    """Generate and display user-selected graphs."""
    
    st.write("### Choose Graph Type and Axes for Visualization:")

    # Dropdown for selecting graph type
    graph_type = st.selectbox("Select Graph Type:", ["Scatter Plot", "Bar Chart", "Line Plot", "Histogram"])

    # Dropdowns for selecting X and Y axes
    x_axis = st.selectbox("Select X axis:", df.columns)
    y_axis = None

    if graph_type != "Histogram":
        y_axis = st.selectbox("Select Y axis:", df.columns)

    if st.button("Generate Graph"):
        plt.figure(figsize=(10, 5))

        if graph_type == "Scatter Plot":
            sns.scatterplot(x=df[x_axis], y=df[y_axis])
        elif graph_type == "Bar Chart":
            sns.barplot(x=df[x_axis], y=df[y_axis])
        elif graph_type == "Line Plot":
            sns.lineplot(x=df[x_axis], y=df[y_axis])
        elif graph_type == "Histogram":
            sns.histplot(df[x_axis], kde=True)

        plt.xlabel(x_axis)
        if y_axis:
            plt.ylabel(y_axis)
        plt.title(f"{graph_type} of {x_axis} {'vs ' + y_axis if y_axis else ''}")
        st.pyplot(plt)

# ----------------------- STREAMLIT UI -----------------------

def generate_streamlit_ui(df, best_model):
    st.title("AI Predictor")
    st.write("### Choose Column to Predict:")

    target_column = st.selectbox("Select target column for prediction:", df.columns)
    
    st.write(f"### Enter Feature Values for Prediction (excluding {target_column}):")

    feature_names = df.columns[df.columns != target_column]
    user_input = []

    for feature in feature_names:
        min_val, max_val = df[feature].min(), df[feature].max()
        value = st.number_input(f"{feature} ({min_val} to {max_val})", min_value=min_val, max_value=max_val, value=min_val)
        user_input.append(value)

    if st.button("Predict"):
        try:
            user_input = np.array(user_input, dtype=float).reshape(1, -1)
            prediction = best_model.predict(user_input)
            st.success(f"Predicted Output: {prediction[0]}")
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------- AUTOGEN AGENTS -----------------------

data_analyzer = autogen.AssistantAgent(
    name="DataAnalyzer",
    llm_config=llm_config,
    function_map={"summarize_data": summarize_data}
)

streamlit_ui_agent = autogen.AssistantAgent(
    name="StreamlitUIAgent",
    llm_config=llm_config,
    function_map={"generate_streamlit_ui": generate_streamlit_ui}
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    code_execution_config={"use_docker": False},
    function_map={"summarize_data": summarize_data, "generate_and_display_graphs": generate_and_display_graphs}
)

groupchat = autogen.GroupChat(
    agents=[data_analyzer, streamlit_ui_agent, user_proxy],
    messages=[],
    max_round=10,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# ----------------------- EXECUTION -----------------------

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    with open(UPLOAD_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    summary, df = summarize_data(UPLOAD_PATH)
    if df is not None:
        st.write(f"### Dataset Summary: {summary}")
        generate_and_display_graphs(df)
        best_model, best_score = train_and_test(df)
        st.write(f"### Best Model: {best_model._class.name_}")
        st.write(f"### Model Score: {best_score}")
        generate_streamlit_ui(df, best_model)
