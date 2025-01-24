# ml_analysis_page.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def run_clustering(df, algorithm, params=None):
    """
    Run clustering algorithms on the provided DataFrame.
    
    Args:
        df: pandas DataFrame
        algorithm: str, name of the clustering algorithm
        params: dict, algorithm parameters (optional)
    """
    # Prepare the data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None, "No numeric columns found in the dataset"
    
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the algorithm
    if params is None:
        params = {}
        
    try:
        if algorithm == "kmeans":
            model = KMeans(n_clusters=params.get("n_clusters", 3), random_state=42)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=params.get("eps", 0.5), min_samples=params.get("min_samples", 5))
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=params.get("n_clusters", 3))
        else:
            return None, f"Unknown algorithm: {algorithm}"
            
        # Fit the model
        labels = model.fit_predict(X_scaled)
        
        # Add cluster labels to the original dataframe
        result_df = df.copy()
        result_df['Cluster'] = labels
        
        # Calculate basic statistics
        stats = {
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
        
        return result_df, stats
        
    except Exception as e:
        return None, str(e)

def run_regression(df, algorithm, target_column, params=None):
    """
    Run regression algorithms on the provided DataFrame.
    
    Args:
        df: pandas DataFrame
        algorithm: str, name of the regression algorithm
        target_column: str, name of the target column
        params: dict, algorithm parameters (optional)
    """
    if target_column not in df.columns:
        return None, f"Target column '{target_column}' not found in dataset"
    
    # Prepare the data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[numeric_cols != target_column]
    
    if len(numeric_cols) == 0:
        return None, "No numeric features found in the dataset"
    
    X = df[numeric_cols]
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the algorithm
    if params is None:
        params = {}
        
    try:
        if algorithm == "linear":
            model = LinearRegression()
        elif algorithm == "ridge":
            model = Ridge(alpha=params.get("alpha", 1.0))
        elif algorithm == "lasso":
            model = Lasso(alpha=params.get("alpha", 1.0))
        else:
            return None, f"Unknown algorithm: {algorithm}"
            
        # Fit and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        return metrics, feature_importance
        
    except Exception as e:
        return None, str(e)

def sklearn_page():
    """Scikit-learn page implementation"""
    st.title("ML Analysis with Scikit-learn")
    
    # File upload
    file_path = st.text_input("Enter the path to your CSV file:", value="")
    
    if not file_path:
        st.info("Please enter a file path to proceed")
        return
        
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Algorithm selection
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Clustering", "Regression"]
        )
        
        if analysis_type == "Clustering":
            algorithm = st.selectbox(
                "Select Clustering Algorithm",
                ["kmeans", "dbscan", "hierarchical"]
            )
            
            # Algorithm-specific parameters
            params = {}
            if algorithm in ["kmeans", "hierarchical"]:
                params["n_clusters"] = st.slider("Number of Clusters", 2, 10, 3)
            elif algorithm == "dbscan":
                params["eps"] = st.slider("Epsilon", 0.1, 2.0, 0.5)
                params["min_samples"] = st.slider("Min Samples", 2, 10, 5)
            
            if st.button("Run Clustering Analysis", type="primary"):
                with st.spinner("Running clustering analysis..."):
                    result_df, stats = run_clustering(df, algorithm, params)
                    
                    if isinstance(stats, str):  # Error message
                        st.error(stats)
                    else:
                        st.success("Clustering completed!")
                        
                        # Display results
                        st.subheader("Clustering Results")
                        st.write("Statistics:", stats)
                        
                        st.subheader("Clustered Data")
                        st.dataframe(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="clustering_results.csv",
                            mime="text/csv"
                        )
        
        elif analysis_type == "Regression":
            # Select target variable
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_column = st.selectbox("Select Target Variable", numeric_cols)
            
            algorithm = st.selectbox(
                "Select Regression Algorithm",
                ["linear", "ridge", "lasso"]
            )
            
            # Algorithm-specific parameters
            params = {}
            if algorithm in ["ridge", "lasso"]:
                params["alpha"] = st.slider("Alpha (Regularization Strength)", 0.0, 10.0, 1.0)
            
            if st.button("Run Regression Analysis", type="primary"):
                with st.spinner("Running regression analysis..."):
                    metrics, feature_importance = run_regression(df, algorithm, target_column, params)
                    
                    if isinstance(metrics, str):  # Error message
                        st.error(metrics)
                    else:
                        st.success("Regression analysis completed!")
                        
                        # Display results
                        st.subheader("Regression Metrics")
                        st.write(metrics)
                        
                        st.subheader("Feature Importance")
                        st.dataframe(feature_importance)
                        
                        # Download results
                        csv = feature_importance.to_csv(index=False)
                        st.download_button(
                            label="Download Feature Importance as CSV",
                            data=csv,
                            file_name="regression_results.csv",
                            mime="text/csv"
                        )
                    
    except Exception as e:
        st.error(f"Error reading or processing file: {str(e)}")