
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Load Saved Models
# =====================
with open("best_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("best_regressor.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

st.set_page_config(page_title="Customer Analytics App", layout="wide")

st.title("📊 Customer Analytics Web Application")

st.sidebar.header("Upload / Input Data")

# =====================
# Data Upload Section
# =====================
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
manual_input = st.sidebar.checkbox("Enter data manually")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
elif manual_input:
    st.sidebar.write("Enter values below:")
    num_features = st.sidebar.number_input("Number of features", min_value=1, max_value=20, value=5)
    user_input = {}
    for i in range(num_features):
        user_input[f"feature_{i+1}"] = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
    df = pd.DataFrame([user_input])
    st.success("Manual input recorded!")
else:
    st.warning("Please upload a CSV or enter values manually to proceed.")
    df = None

# =====================
# Analysis Section
# =====================
if df is not None:
    st.subheader("Preview of Input Data")
    st.dataframe(df.head())

    # ---- Classification (Conversion Prediction) ----
    st.subheader("🔮 Conversion Prediction (Classification)")
    try:
        y_pred_class = classifier.predict(df)
        st.write("**Predicted Conversion:**")
        st.write(y_pred_class)

        st.bar_chart(pd.Series(y_pred_class).value_counts())
    except Exception as e:
        st.error(f"Classification failed: {e}")

    # ---- Regression (Revenue Estimation) ----
    st.subheader("💰 Revenue Estimation (Regression)")
    try:
        y_pred_reg = regressor.predict(df)
        st.write("**Predicted Revenue:**")
        st.write(y_pred_reg)

        st.line_chart(y_pred_reg)
    except Exception as e:
        st.error(f"Regression failed: {e}")

    # ---- Clustering (Customer Segments) ----
    st.subheader("👥 Customer Segmentation (Clustering)")
    try:
        clusters = kmeans.predict(df)
        df["Cluster"] = clusters
        st.write("Cluster Assignments:")
        st.dataframe(df)

        # Plot cluster counts
        cluster_counts = df["Cluster"].value_counts()
        st.bar_chart(cluster_counts)

        # Scatter plot for first 2 features
        if df.shape[1] >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df["Cluster"], palette="Set2", ax=ax)
            plt.title("Cluster Visualization (First 2 Features)")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

    # ---- Additional Visualizations ----
    st.subheader("📈 Data Visualizations")
    try:
        st.write("Histogram of Numeric Features")
        st.bar_chart(df.select_dtypes(include=[np.number]).iloc[:, 0])
    except:
        st.warning("No numeric columns available for histogram.")
