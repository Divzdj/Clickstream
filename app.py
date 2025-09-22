import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)

# Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from collections import Counter
import warnings

# Suppress warnings for cleaner output in the app
warnings.filterwarnings("ignore")

# --- Dummy Data to Simulate Training ---
def create_dummy_data():
    """
    Creates a small, representative dataset to simulate the original training data.
    This allows the models and one-hot encoder to be fitted on consistent data.
    """
    data = {
        'session_id': [1, 1, 2, 2, 2, 3],
        'page': [1, 2, 1, 2, 3, 1],
        'order': [1, 2, 1, 2, 3, 1],
        'page1_main_category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'page2_clothing_model': ['M1', 'M2', 'M1', 'M3', 'M4', 'M5'],
        'price': [100.0, 120.0, 110.0, 90.0, 150.0, 80.0],
        'price_2': [1, 2, 1, 2, 2, 1]
    }
    dummy_train = pd.DataFrame(data)
    return dummy_train

# --- Data Preprocessing and Feature Engineering function ---
def preprocess_data(df, ohe):
    """
    Applies the same feature engineering and one-hot encoding steps from the notebook
    to new input data, ensuring feature consistency. This function does not handle scaling or
    feature selection as these are handled by the fitted pipelines.
    """
    df_processed = df.copy()

    # Feature Engineering (mirroring the notebook)
    df_processed['session_length'] = df_processed.groupby('session_id')['order'].transform('max')
    df_processed['num_clicks'] = df_processed.groupby('session_id')['order'].transform('count')
    df_processed['time_per_category'] = df_processed.groupby(['session_id', 'page1_main_category'])['order'].transform('count')
    df_processed['unique_categories'] = df_processed.groupby('session_id')['page1_main_category'].transform('nunique')
    df_processed['is_bounce'] = (df_processed['session_length'] == 1).astype(int)
    df_processed['is_revisit'] = df_processed.groupby(['session_id', 'page2_clothing_model'])['page'].transform(lambda x: int(x.duplicated().any()))

    # One-Hot Encoding
    categorical_cols = ['page1_main_category', 'page2_clothing_model']
    
    # Explicitly convert to string to prevent TypeError with np.isnan
    df_processed[categorical_cols] = df_processed[categorical_cols].astype(str)
    
    df_cat_ohe = ohe.transform(df_processed[categorical_cols])
    df_ohe = pd.DataFrame(df_cat_ohe, columns=ohe.get_feature_names_out(categorical_cols), index=df_processed.index)
    df_processed = pd.concat([df_processed.drop(columns=categorical_cols, errors='ignore'), df_ohe], axis=1)

    # Drop non-feature columns
    df_processed = df_processed.drop(columns=['session_id', 'price', 'price_2', 'order', 'page'], errors='ignore')

    return df_processed


# --- Train Models on dummy data (This replaces loading from .pkl files) ---
@st.cache_resource
def train_and_load_models():
    """
    Simulates the entire notebook pipeline to create and cache the trained models.
    """
    st.write("Training models on sample data...")
    dummy_train = create_dummy_data()

    # Preprocessing and Feature Engineering
    # This loop needs to happen BEFORE OneHotEncoding,
    # as it requires the original categorical columns.
    df = dummy_train
    df['session_length'] = df.groupby('session_id')['order'].transform('max')
    df['num_clicks'] = df.groupby('session_id')['order'].transform('count')
    df['time_per_category'] = df.groupby(['session_id', 'page1_main_category'])['order'].transform('count')
    df['unique_categories'] = df.groupby('session_id')['page1_main_category'].transform('nunique')
    df['is_bounce'] = (df['session_length'] == 1).astype(int)
    df['is_revisit'] = df.groupby(['session_id', 'page2_clothing_model'])['page'].transform(lambda x: int(x.duplicated().any()))

    categorical_cols = dummy_train.select_dtypes(include=['object']).columns.tolist()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = ohe.fit_transform(dummy_train[categorical_cols])
    train_ohe = pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_cols), index=dummy_train.index)
    dummy_train = pd.concat([dummy_train.drop(columns=categorical_cols), train_ohe], axis=1)

    # Define feature sets
    non_feature_cols = ['session_id', 'price', 'price_2']
    X_class = dummy_train.drop(columns=non_feature_cols, errors='ignore')
    y_class = dummy_train['price_2'].astype(int)
    X_reg = dummy_train.drop(columns=non_feature_cols, errors='ignore')
    y_reg = dummy_train['price']
    X_clu = dummy_train.drop(columns=non_feature_cols, errors='ignore')

    # Classification Pipeline
    y_class_mapped = y_class.map({1: 0, 2: 1})
    Xc = X_class.copy()
    to_drop_class = ['order', 'page']
    Xc = Xc.drop(columns=to_drop_class, errors='ignore')
    Xc_train, _, yc_train, _ = train_test_split(
        Xc, y_class_mapped, test_size=0.2, random_state=42, stratify=y_class_mapped
    )
    best_classifier = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(mutual_info_classif, k=min(100, Xc_train.shape[1]))),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    best_classifier.fit(Xc_train, yc_train)

    # Regression Pipeline
    Xr = X_reg.copy()
    to_drop_reg = ['order', 'page']
    Xr = Xr.drop(columns=to_drop_reg, errors='ignore')
    best_regressor = ImbPipeline([
        ("select", SelectKBest(f_regression, k=min(100, Xr.shape[1]))),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    best_regressor.fit(Xr, y_reg)

    # Clustering Model (on a sample for performance)
    X_clu_sample = X_clu.sample(n=min(10000, len(X_clu)), random_state=42)
    scaler_clu = StandardScaler().fit(X_clu_sample)
    X_clu_scaled = scaler_clu.transform(X_clu_sample)
    pca_clu = PCA(n_components=2, random_state=42).fit(X_clu_scaled)
    optimal_k = 5
    best_clusterer = KMeans(n_clusters=optimal_k, random_state=42)
    best_clusterer.fit(X_clu_scaled)

    return best_classifier, best_regressor, best_clusterer, ohe, scaler_clu, pca_clu

# --- Main Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Clickstream Prediction App")

st.title("Clickstream Analysis and Prediction")
st.markdown(
    """
    This application uses machine learning models trained on clickstream data to provide
    real-time predictions. You can upload a CSV file or use the provided sample data to
    get insights on customer behavior.
    """
)

# Load models and preprocessing components with caching
with st.spinner("Training models on a sample dataset..."):
    best_classifier, best_regressor, best_clusterer, ohe, scaler_clu, pca_cluster = train_and_load_models()
st.success("Models are ready!")

# Sidebar for input method selection
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose your input method:", ("Upload CSV", "Use Sample Data"))

df_to_predict = None
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df_to_predict = pd.read_csv(uploaded_file)
        st.info("CSV file loaded successfully!")
elif input_method == "Use Sample Data":
    st.sidebar.info("Click 'Run Predictions' to analyze a sample clickstream session.")
    sample_data = {
        'session_id': [1, 1, 1],
        'page': [1, 2, 3],
        'order': [1, 2, 3],
        'page1_main_category': ['A', 'B', 'B'],
        'page2_clothing_model': ['M10', 'M11', 'M11'],
        'price': [150.0, 200.0, 200.0]
    }
    df_to_predict = pd.DataFrame(sample_data)

if df_to_predict is not None:
    st.header("Predictions")

    # Preprocess the input data
    df_processed = preprocess_data(df_to_predict, ohe)

    # Get the feature names used for training from the fitted pipelines
    classifier_features = best_classifier.named_steps['select'].get_feature_names_out()
    regressor_features = best_regressor.named_steps['select'].get_feature_names_out()
    cluster_features = scaler_clu.feature_names_in_

    # Ensure the columns in the input data match the training data and are in the same order
    df_class = df_processed.reindex(columns=classifier_features, fill_value=0)
    df_reg = df_processed.reindex(columns=regressor_features, fill_value=0)
    df_clu = df_processed.reindex(columns=cluster_features, fill_value=0)

    # Make predictions
    class_pred = best_classifier.predict(df_class)
    reg_pred = best_regressor.predict(df_reg)
    
    # The clusterer was fitted on a separate scaled dataset, so we must manually scale the new data
    cluster_labels = best_clusterer.predict(scaler_clu.transform(df_clu))

    # Add predictions back to the original dataframe for visualization
    df_to_predict['conversion_prediction'] = class_pred
    df_to_predict['revenue_estimation'] = reg_pred
    df_to_predict['cluster'] = cluster_labels
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Conversion Prediction")
        pred_text = "Yes, likely to convert." if class_pred.mean() > 0.5 else "No, not likely to convert."
        st.markdown(f"**Prediction:** `{pred_text}`")
    with col2:
        st.subheader("Revenue Estimation")
        st.markdown(f"**Estimated avg. revenue:** `${reg_pred.mean():.2f}`")
    with col3:
        st.subheader("Customer Segment")
        st.markdown(f"**Cluster ID:** `{cluster_labels[0]}`")

    st.header("Visualizations")
    
    # Clustering Visualization
    st.subheader("User Segments in Reduced Space")
    st.markdown("A scatter plot of users in a 2D space, colored by their assigned cluster.")
    
    # Use the pre-fitted PCA to reduce the data for plotting
    df_for_pca = scaler_clu.transform(df_clu)
    pca_data = pca_cluster.transform(df_for_pca)
    df_pca = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
    df_pca['cluster'] = cluster_labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_pca, palette='viridis', style='cluster', s=100, ax=ax)
    plt.title("User Segments based on Behavior")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Main Category Distribution")
        fig_cat, ax_cat = plt.subplots(figsize=(6, 4))
        sns.countplot(y='page1_main_category', data=df_to_predict)
        plt.title("Distribution of Main Categories Visited")
        plt.tight_layout()
        st.pyplot(fig_cat)
    
    with col2:
        st.subheader("Estimated Revenue Histogram")
        fig_rev, ax_rev = plt.subplots(figsize=(6, 4))
        sns.histplot(df_to_predict['revenue_estimation'], kde=True, bins=10)
        plt.title("Estimated Revenue Distribution")
        plt.xlabel("Estimated Revenue")
        st.pyplot(fig_rev)
    
    st.subheader("Raw Data with Predictions")
    st.write(df_to_predict)

else:
    st.info("Please upload a CSV file or use the sample data from the sidebar to run predictions.")
