import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

#  1. Load Trained Artifacts 

MODELS_LOADED = False
# Load models saved as .pkl files
try:
    
    best_clf_pipeline = joblib.load('best_clf_pipeline_final.pkl')
    best_reg_pipeline = joblib.load('best_reg_pipeline_final.pkl')
    km_model = joblib.load('customer_segmenter_kmeans.pkl')
    MODELS_LOADED = True
except FileNotFoundError:
    st.error("Error loading models. Please ensure 'best_clf_pipeline_final.pkl', 'best_reg_pipeline_final.pkl', and 'customer_segmenter_kmeans.pkl' are in the same directory as app.py.")

# 2. Define Helper Variables 
CONVERSION_THRESHOLD = 0.35
FEATURE_COLS = ['total_clicks', 'model_photography', 'country_2', 'country_3', 'country_4', 'country_5', 'country_6', 'country_7', 'country_8', 'country_9', 'country_10', 'country_11', 'country_12', 'country_13', 'country_14', 'country_15', 'country_16', 'country_17', 'country_18', 'country_19', 'country_20', 'country_21', 'country_22', 'country_23', 'country_24', 'country_25', 'country_26', 'country_27', 'country_28', 'country_29', 'country_30', 'country_31', 'country_32', 'country_33', 'country_34', 'country_35', 'country_36', 'country_37', 'country_38', 'country_39', 'country_41', 'country_42', 'country_43', 'country_44', 'country_45', 'country_46', 'country_47', 'main_category_mode_2', 'main_category_mode_3', 'main_category_mode_4', 'colour_2', 'colour_3', 'colour_4', 'colour_5', 'colour_6', 'colour_7', 'colour_8', 'colour_9', 'colour_10', 'colour_11', 'colour_12', 'colour_13', 'colour_14']

#  3. Feature Transformation Functions 

def transform_input(input_df, feature_cols):
    """
    Transforms and aligns the input data (single row or batch) to match the training feature format.
    """
    if input_df.empty:
        return pd.DataFrame()

    # CRITICAL: Ensure categorical columns are strings for OHE
    categorical_cols = ['country', 'main_category_mode', 'colour']
    for col in categorical_cols:
        if col in input_df.columns:
            # Ensure type is string for OHE
            input_df[col] = input_df[col].astype(str)
        else:
            # Fails if a core required column is missing
            return pd.DataFrame() 

    # 1. Label Encoding (Must match preprocessing steps)
    if 'model_photography' in input_df.columns:
        # Robust Mapping for model_photography: handle case issues and NaNs
        input_df['model_photography'] = input_df['model_photography'].astype(str).str.title()
        
        # Map 'Yes' to 1 and 'No' to 0
        input_df['model_photography'] = input_df['model_photography'].map({'Yes': 1, 'No': 0})
        
        # Handle any remaining NaNs (e.g., blank cells or unrecognized input like 'nan') by treating them as 'No' (0)
        input_df['model_photography'] = input_df['model_photography'].fillna(0).astype(int) 
    
    # 2. One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # 3. Align with Training Features 
    # Create an empty DataFrame with all expected features
    aligned_df = pd.DataFrame(0, index=input_df.index, columns=feature_cols)
    
    # Copy the values from the input to the aligned DataFrame
    common_cols = list(set(input_df.columns) & set(aligned_df.columns))
    aligned_df[common_cols] = input_df[common_cols]
            
    return aligned_df

def run_prediction(X_input):
    """
    Applies all three models (CLF, REG, CLUST) to the prepared feature DataFrame.
    Returns a DataFrame with the original features plus predictions.
    """
    # Check if models were loaded
    if X_input.empty or not MODELS_LOADED:
        return pd.DataFrame()
    
    # Check if feature count matches expected feature count (CRITICAL CHECK)
    if X_input.shape[1] != len(FEATURE_COLS):
        st.error(f"Feature count mismatch! Expected {len(FEATURE_COLS)} features, but got {X_input.shape[1]}. Prediction halted. Check FEATURE_COLS list.")
        return pd.DataFrame()

    try:
        # 4.1 Classification Prediction (Conversion Likelihood)
        clf_proba = best_clf_pipeline.predict_proba(X_input)[:, 1]
        clf_pred = np.where(clf_proba >= CONVERSION_THRESHOLD, 1, 0)

        # 4.2 Regression Prediction (Estimated Revenue)
        reg_pred = best_reg_pipeline.predict(X_input)

        # 4.3 Clustering Prediction (Customer Segment)
        # Get the scaler from the CLF pipeline for consistency
        try:
            scaler = best_clf_pipeline.named_steps['scaler']
        except KeyError:
            st.error("Could not find 'scaler' step in classification pipeline. Check pipeline construction.")
            return pd.DataFrame()
            
        X_scaled_for_clustering = scaler.transform(X_input)
        cluster_label = km_model.predict(X_scaled_for_clustering)
    except Exception as e:
        st.error(f"Prediction execution failed, likely due to feature or scaling issues: {e}")
        return pd.DataFrame()


    # Combine results
    results_df = X_input.copy()
    results_df['Conversion_Prob'] = clf_proba
    results_df['Conversion_Prediction'] = clf_pred
    results_df['Estimated_Revenue'] = reg_pred
    results_df['Cluster_ID'] = cluster_label
    
    # Add readable segment name
    segment_map = {
        0: "High-Engagement Shopper",
        1: "Price-Sensitive Explorer",
        2: "Quick Decision Maker",
        3: "General Browser" 
        
    }
    results_df['Customer_Segment'] = results_df['Cluster_ID'].map(segment_map).fillna("Unknown Segment")

    return results_df


#  4. Streamlit UI and Prediction Logic

st.set_page_config(page_title="Clickstream Predictive Analytics", layout="wide")

st.title(" E-commerce Session Predictive Analytics")
st.markdown("Predict the likelihood of a high-price item view, estimate potential session revenue, and identify customer segments using trained machine learning models.")

if not MODELS_LOADED:
    st.stop()

# --- Tabbed Interface ---
tab_manual, tab_batch, tab_viz = st.tabs(["Manual Session Prediction", "Batch File Prediction", "Analysis & Visualization"])



# TAB 1: Manual Session Prediction

with tab_manual:
    st.header("Single Session Analysis")
    st.markdown("Enter the parameters for a single user session to get real-time predictions.")

    with st.sidebar:
        st.header("Session Input Parameters")
        
        # INPUT FIELDS 
        total_clicks = st.slider("Total Clicks (Session Length)", 1, 100, 15)
        country = st.selectbox("Country", ['1', '2', '3', '4']) 
        main_category = st.selectbox("Main Category Mode (Most Viewed)", ['1', '2', '3', '4', '5'])
        colour = st.selectbox("Favorite Colour Mode", ['1', '2', '3', '4', '5'])
        model_photography = st.selectbox("Model Photography Used?", ['Yes', 'No'])

        # Prepare Input Data (as a DataFrame)
        input_data = {
            'total_clicks': total_clicks,
            'country': country,
            'main_category_mode': main_category,
            'colour': colour,
            'model_photography': model_photography
        }
        input_df = pd.DataFrame([input_data])
    
    # Transform input and align features
    X_input = transform_input(input_df, FEATURE_COLS)

    # Run Prediction
    results_df = run_prediction(X_input)

    # --- Display Results ---
    st.subheader("Results")
    
    if not results_df.empty:
        
        clf_proba = results_df['Conversion_Prob'].iloc[0]
        clf_pred = results_df['Conversion_Prediction'].iloc[0]
        reg_pred = results_df['Estimated_Revenue'].iloc[0]
        cluster_label = results_df['Cluster_ID'].iloc[0]
        segment_desc = results_df['Customer_Segment'].iloc[0]

        col1, col2, col3 = st.columns(3)

        # Classification Output
        with col1:
            st.header("High-Price View")
            st.metric(label="Likelihood", value=f"{clf_proba * 100:.1f}%")
            status = "High Likelihood" if clf_pred == 1 else "Low Likelihood"
            st.success(f"Prediction: **{status}**")

        # Regression Output
        with col2:
            st.header("Revenue Estimation")
            st.metric(label="Estimated Session Revenue", value=f"${reg_pred:.2f}") 
            st.info("Goal: Revenue Optimization")
            
        # Clustering Output
        with col3:
            st.header("Customer Segment")
            st.metric(label="Cluster ID", value=f"Cluster {cluster_label}")
            st.warning(f"Description: **{segment_desc}**")

        st.subheader("Model Inputs (Aligned Features)")
        st.dataframe(X_input)


 
# TAB 2: Batch File Prediction

with tab_batch:
    st.header("Batch File Prediction")
    st.markdown("Upload a CSV file containing multiple sessions for bulk analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # 1. Define explicit dtypes for robust reading
            dtype_map = {
                'total_clicks': np.int32,
                'country': str, 
                'main_category_mode': str,
                'colour': str, 
                'model_photography': str 
            }
            
            # Read CSV with explicit dtypes
            batch_df = pd.read_csv(uploaded_file, dtype=dtype_map)
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head())
            
            # Check if required columns are present
            required_cols = ['total_clicks', 'country', 'main_category_mode', 'colour', 'model_photography']
            
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"The uploaded CSV is missing required columns: {required_cols}")
            else:
                st.info("Running batch prediction...")
                
                with st.spinner('Preprocessing and predicting...'):
                    # Transform the entire batch, selecting only required columns
                    X_batch_input = transform_input(batch_df[required_cols], FEATURE_COLS)
                    
                    # Run Prediction
                    batch_results = run_prediction(X_batch_input)
                
                # 2. Check for successful results before saving to state
                if not batch_results.empty:
                    st.success("Batch prediction complete!")
                    st.session_state['batch_results'] = batch_results # Save for the Viz tab
                    
                    st.subheader("Batch Prediction Results (Sample)")
                    st.dataframe(batch_results.head(10))
                    
                    # Download button for results
                    csv_export = batch_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results CSV",
                        data=csv_export,
                        file_name='clickstream_predictions.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("Prediction failed. The results DataFrame is empty. Check data for unseen categories.")

        except Exception as e:
            st.error(f"Error processing file. This may be due to incompatible data types or values: {e}")


# TAB 3: Analysis & Visualization

with tab_viz:
    st.header("Analysis & Visualization")
    st.markdown("Visual summary of the last processed batch prediction (Classification, Regression, and Clustering results).")

    if 'batch_results' in st.session_state and not st.session_state['batch_results'].empty:
        results_df = st.session_state['batch_results']

        
        try:
            results_df['Conversion_Prediction'] = results_df['Conversion_Prediction'].astype(int)
            results_df['Customer_Segment'] = results_df['Customer_Segment'].astype(str)
        except Exception as e:
            st.error(f"Data type conversion error before plotting: {e}")


        try:
            st.subheader("Conversion Prediction Summary (Classification)")
            
            
            conversion_counts = results_df['Conversion_Prediction'].value_counts().reset_index()
            conversion_counts.columns = ['Conversion_Prediction', 'Count']
            
            # Map 0/1 to meaningful labels
            conversion_counts['Label'] = conversion_counts['Conversion_Prediction'].map({
                0: 'Not Converted (0)', 
                1: 'Converted (1)'
            })
            
            if conversion_counts.empty:
                 st.warning("Conversion prediction data is empty. Cannot generate the chart.")
            else:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                sns.barplot(x='Label', y='Count', data=conversion_counts, ax=ax, palette=['darkred', 'mediumseagreen'])
                
                ax.set_title("Total Sessions Predicted Converted vs. Not Converted") 
                ax.set_ylabel('Number of Sessions')
                ax.set_xlabel('Prediction Outcome')
                
                # Add text labels on bars for clarity
                for container in ax.containers:
                    ax.bar_label(container, fmt='%d')
                    
                st.pyplot(fig)
                plt.close(fig) 
        except Exception as e:
            st.error(f"Error generating Conversion Summary chart: {e}")
            
        try:
            st.subheader("Revenue Distribution (Regression)")
            
            # 2. Histogram: Estimated Revenue Distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(results_df['Estimated_Revenue'], bins=30, kde=True, ax=ax, color='skyblue')
            ax.set_title('Distribution of Estimated Session Revenue')
            ax.set_xlabel('Estimated Revenue ($)')
            st.pyplot(fig)
            plt.close(fig) 
        except Exception as e:
            st.error(f"Error generating Revenue Distribution chart: {e}")
            
        try:
            st.subheader("Customer Segmentation Overview")
            
            # 3. Pie Chart: Segment Distribution
            fig, ax = plt.subplots(figsize=(8, 8))
            segment_counts = results_df['Customer_Segment'].value_counts()
            ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'}, colors=sns.color_palette('pastel'))
            ax.axis('equal') 
            ax.set_title('Customer Segment Distribution')
            st.pyplot(fig)
            plt.close(fig) 
        except Exception as e:
            st.error(f"Error generating Segment Distribution chart: {e}")

    else:
        st.info("Please successfully upload and process a CSV file in the 'Batch File Prediction' tab to generate visualizations.")