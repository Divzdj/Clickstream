#  Customer Conversion Analysis for Online Shopping

##  Project Overview
This project is an end-to-end Machine Learning solution developed to analyze customer clickstream data from an e-commerce platform. The goal is to provide actionable insights into browsing behavior to enhance customer engagement and drive sales.

The solution is deployed as an interactive web application where businesses can predict customer purchases, forecast revenue, and segment users for targeted marketing.

###  **[View Live Application](https://clickstream-9hwsn37uny9nwqrcpfebvz.streamlit.app/)**

---

##  Business Objectives
The application solves three critical business problems:

1.  **Customer Conversion Prediction (Classification):** * *Goal:* Predict whether a user will complete a purchase (Conversion) or not based on their session activity.
    * *Model:* Random Forest / XGBoost (Binary Classification).
2.  **Revenue Forecasting (Regression):**
    * *Goal:* Estimate the potential revenue a customer is likely to generate to optimize ad spend.
    * *Model:* Linear Regression / Gradient Boosting.
3.  **Customer Segmentation (Clustering):**
    * *Goal:* Group users into distinct personas (e.g., "Window Shoppers," "High-Value Customers") for personalized recommendations.
    * *Model:* K-Means Clustering.

---

##  Dataset Details
The dataset consists of **clickstream data** capturing user interactions within a single session. Key features include:

* **Session Info:** `SESSION ID`, `ORDER` (click sequence), `DAY`, `MONTH`, `YEAR`.
* **User Demographics:** `COUNTRY` (47 categories).
* **Product Interaction:** `PAGE 1` (Main Category), `PAGE 2` (Product Code), `COLOUR`, `PRICE`, `PRICE 2` (Above Avg Price?), `MODEL PHOTOGRAPHY`.
* **Page Context:** `LOCATION` (Screen area of the clicked item), `PAGE` (Page number).

*Source: UCI Machine Learning Repository - Clickstream Data*

---

##  Technical Architecture

### 1. Data Preprocessing
* **Cleaning:** Handling missing values and removing duplicates.
* **Aggregation:** Transforming "click-level" data into "session-level" data (grouping by `SESSION ID`).
* **Encoding:** One-Hot Encoding for categorical variables (e.g., `COUNTRY`, `PAGE 1`) and Label Encoding for binary features.

### 2. Exploratory Data Analysis (EDA)
* Visualizing sales funnels, session duration distributions, and price sensitivity.
* Correlation heatmaps to identify key drivers of conversion.

### 3. Model Pipeline
* **Balancing:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance in conversion data.
* **Scaling:** `StandardScaler` applied to numerical features like `PRICE` and `ORDER`.
* **Training:** Multi-model approach validated using Accuracy, F1-Score (Classification), RMSE (Regression), and Silhouette Score (Clustering).

### 4. Deployment
* Built with **Streamlit** for a user-friendly frontend.
* Supports CSV upload and manual parameter input for real-time inference.

---

##  Installation & Local Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/customer-conversion-analysis.git](https://github.com/your-username/customer-conversion-analysis.git)
    cd customer-conversion-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---
