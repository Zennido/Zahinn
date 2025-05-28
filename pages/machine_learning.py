import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)

# Set page config
st.set_page_config(
    page_title="Machine Learning | Zahinn",
    page_icon="📊",
    layout="wide"
)

# Check if data is available in session state
if "data" not in st.session_state or st.session_state.data is None:
    st.error("No data loaded. Please upload a file on the main page.")
    st.stop()

# Use transformed data if available, otherwise use original data
if "transformed_data" in st.session_state and st.session_state.transformed_data is not None:
    df = st.session_state.transformed_data
else:
    df = st.session_state.data

# Header
st.title("Machine Learning")
st.write("Apply machine learning techniques to gain insights and make predictions from your data.")

# Initialize ML model in session state if needed
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None
if "ml_features" not in st.session_state:
    st.session_state.ml_features = None
if "ml_target" not in st.session_state:
    st.session_state.ml_target = None
if "ml_predictions" not in st.session_state:
    st.session_state.ml_predictions = None
if "ml_metrics" not in st.session_state:
    st.session_state.ml_metrics = {}
if "ml_type" not in st.session_state:
    st.session_state.ml_type = None

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Configuration")
    
    # Task selection
    ml_task = st.radio(
        "Select Task Type",
        options=["Regression", "Classification"],
        help="Regression predicts continuous values. Classification predicts categories."
    )
    
    # Column selection
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_columns = df.columns.tolist()
    
    if ml_task == "Regression":
        # For regression, target should be numeric
        target_options = numeric_columns
        if not target_options:
            st.error("No numeric columns available for regression target.")
            st.stop()
    else:  # Classification
        # For classification, target can be numeric or categorical
        target_options = all_columns
    
    target_col = st.selectbox(
        "Select Target Variable (what you want to predict)",
        options=target_options
    )
    
    # Feature selection
    feature_options = [col for col in all_columns if col != target_col]
    selected_features = st.multiselect(
        "Select Features (variables used for prediction)",
        options=feature_options,
        default=feature_options[:min(5, len(feature_options))]
    )
    
    # Model selection based on task
    if ml_task == "Regression":
        model_options = [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ]
    else:  # Classification
        model_options = [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "XGBoost Classifier"
        ]
    
    selected_model = st.selectbox(
        "Select Model",
        options=model_options
    )
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        test_size = st.slider(
            "Test Data Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing the model"
        ) / 100
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Seed for random number generation (for reproducibility)"
        )
        
        normalize_features = st.checkbox(
            "Normalize Features",
            value=True,
            help="Scale numeric features to have mean=0 and variance=1"
        )
    
    # Train model button
    train_button = st.button("Train Model", use_container_width=True)
    
    # Reset button
    reset_button = st.button("Reset", use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.markdown("### Model Information")
    
    if ml_task == "Regression":
        st.markdown("""
        **Regression** predicts continuous values like sales, prices, or customer lifetime value.
        
        **Models available:**
        - **Linear Regression**: Simple, interpretable model for linear relationships
        - **Decision Tree**: Handles non-linear relationships, easy to understand
        - **Random Forest**: Ensemble of trees, good accuracy but less interpretable
        """)
    else:
        st.markdown("""
        **Classification** predicts categories like customer churn, purchase decisions, or risk levels.
        
        **Models available:**
        - **Logistic Regression**: Simple, interpretable model with probabilities
        - **Decision Tree**: Visual decision rules, handles non-linear relationships
        - **Random Forest**: Ensemble of trees, good accuracy and robust
        - **XGBoost**: Advanced boosting algorithm, typically highest accuracy
        """)
        
with col2:
    if reset_button:
        # Reset all ML-related session state variables
        st.session_state.ml_model = None
        st.session_state.ml_features = None
        st.session_state.ml_target = None
        st.session_state.ml_predictions = None
        st.session_state.ml_metrics = {}
        st.session_state.ml_type = None
        st.success("Model and results have been reset")
        st.rerun()
    
    if train_button:
        if not selected_features:
            st.error("Please select at least one feature for the model")
        else:
            with st.spinner("Training model..."):
                # Store selections in session state
                st.session_state.ml_features = selected_features
                st.session_state.ml_target = target_col
                st.session_state.ml_type = ml_task
                
                # Prepare features and target
                X = df[selected_features].copy()
                y = df[target_col].copy()
                
                # Handle categorical features
                categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                # Create a container for transformed columns
                X_processed = pd.DataFrame(index=X.index)
                
                # Process categorical features with one-hot encoding
                for cat_col in categorical_features:
                    # Simple one-hot encoding
                    one_hot = pd.get_dummies(X[cat_col], prefix=cat_col, drop_first=True)
                    X_processed = pd.concat([X_processed, one_hot], axis=1)
                
                # Process numeric features
                for num_col in numeric_features:
                    # Add numeric columns directly
                    X_processed[num_col] = X[num_col]
                
                # Fill any remaining NAs with 0
                X_processed = X_processed.fillna(0)
                
                # For classification, handle target if it's categorical
                if ml_task == "Classification" and (y.dtype == 'object' or y.dtype.name == 'category'):
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                    # Store classes for later use
                    st.session_state.target_classes = label_encoder.classes_
                
                # Normalize features if selected
                if normalize_features and len(numeric_features) > 0:
                    scaler = StandardScaler()
                    for num_col in numeric_features:
                        if num_col in X_processed.columns:
                            X_processed[num_col] = scaler.fit_transform(X_processed[[num_col]])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=int(random_state)
                )
                
                # Initialize and train the model
                if selected_model == "Linear Regression":
                    model = LinearRegression()
                elif selected_model == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=int(random_state))
                elif selected_model == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=100, random_state=int(random_state))
                elif selected_model == "Logistic Regression":
                    model = LogisticRegression(random_state=int(random_state), max_iter=1000)
                elif selected_model == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(random_state=int(random_state))
                elif selected_model == "Random Forest Classifier":
                    model = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
                elif selected_model == "XGBoost Classifier":
                    model = xgb.XGBClassifier(random_state=int(random_state))
                
                # Ensure feature names are strings (XGBoost requirement)
                X_train.columns = X_train.columns.astype(str)
                X_test.columns = X_test.columns.astype(str)
                model.fit(X_train, y_train)


                
                # Store the model in session state
                st.session_state.ml_model = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Save predictions for use in the app
                if len(X_test) > 0:
                    test_indices = X_test.index
                    predictions_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred
                    }, index=test_indices)
                    st.session_state.ml_predictions = predictions_df
                
                # Calculate metrics based on task type
                if ml_task == "Regression":
                    # Regression metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
                    
                    st.session_state.ml_metrics = {
                        'MSE': mse,
                        'RMSE': rmse,
                        'R²': r2,
                        'CV R² (mean)': cv_scores.mean(),
                        'CV R² (std)': cv_scores.std()
                    }
                else:
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # For multi-class, we need to specify average
                    n_classes = len(np.unique(y))
                    if n_classes > 2:
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                    else:
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                    
                    # Store confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.session_state.ml_confusion_matrix = cm
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='accuracy')
                    
                    st.session_state.ml_metrics = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'CV Accuracy (mean)': cv_scores.mean(),
                        'CV Accuracy (std)': cv_scores.std()
                    }
                
                # Feature importances (if available)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': list(X_processed.columns),
                        'Importance': importances.ravel()
                    }).sort_values('Importance', ascending=False)
                    st.session_state.ml_feature_importance = feature_importance

                elif hasattr(model, 'coef_'):
                    # For linear models
                    coefficients = model.coef_
                    if coefficients.ndim > 1 and coefficients.shape[0] == 1:
                        coefficients = coefficients[0]  # For binary logistic regression
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X_processed.columns,
                        'Coefficient': np.abs(coefficients)
                    }).sort_values('Coefficient', ascending=False)
                    st.session_state.ml_feature_importance = feature_importance
                
                st.success(f"Successfully trained {selected_model}")
                st.rerun()
    
    # Display results if model is trained
    if st.session_state.ml_model is not None:
        st.subheader("Model Results")
        
        # Display metrics
        metrics_df = pd.DataFrame({
            'Metric': list(st.session_state.ml_metrics.keys()),
            'Value': list(st.session_state.ml_metrics.values())
        })
        
        st.write("### Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Display feature importance if available
        if hasattr(st.session_state, 'ml_feature_importance'):
            st.write("### Feature Importance")
            imp_df = st.session_state.ml_feature_importance.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.barh(imp_df['Feature'], imp_df['Importance' if 'Importance' in imp_df.columns else 'Coefficient'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Display confusion matrix for classification
        if st.session_state.ml_type == "Classification" and hasattr(st.session_state, 'ml_confusion_matrix'):
            st.write("### Confusion Matrix")
            cm = st.session_state.ml_confusion_matrix
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(fig)
        
        # Display actual vs predicted scatter plot for regression
        if st.session_state.ml_type == "Regression" and st.session_state.ml_predictions is not None:
            st.write("### Actual vs Predicted Values")
            pred_df = st.session_state.ml_predictions
            
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(pred_df['Actual'], pred_df['Predicted'], alpha=0.5)
            plt.plot([pred_df['Actual'].min(), pred_df['Actual'].max()], 
                     [pred_df['Actual'].min(), pred_df['Actual'].max()], 
                     'k--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            st.pyplot(fig)
        
        # Show sample predictions
        if st.session_state.ml_predictions is not None:
            st.write("### Sample Predictions")
            st.dataframe(st.session_state.ml_predictions.head(10), use_container_width=True)
        
        # Add predictions to the dataset
        if st.button("Add Predictions to Dataset", use_container_width=True):
            # Create full predictions for the entire dataset
            X_full = df[st.session_state.ml_features].copy()
            
            # Process the full dataset the same way as during training
            categorical_features = X_full.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            numeric_features = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            X_processed_full = pd.DataFrame(index=X_full.index)
            
            # Process categorical features with one-hot encoding
            for cat_col in categorical_features:
                one_hot = pd.get_dummies(X_full[cat_col], prefix=cat_col, drop_first=True)
                X_processed_full = pd.concat([X_processed_full, one_hot], axis=1)
            
            # Process numeric features
            for num_col in numeric_features:
                X_processed_full[num_col] = X_full[num_col]
            
            # Fill any remaining NAs with 0
            X_processed_full = X_processed_full.fillna(0)
            
            # Check that we have all needed columns from training
            missing_cols = set(st.session_state.ml_model.feature_names_in_) - set(X_processed_full.columns)
            if missing_cols:
                st.error(f"Missing columns from training: {missing_cols}")
            else:
                # Reorder columns to match training data
                X_processed_full = X_processed_full[st.session_state.ml_model.feature_names_in_]
                
                # Make predictions
                full_preds = st.session_state.ml_model.predict(X_processed_full)
                
                # Add to dataframe
                prediction_column_name = f"Predicted_{st.session_state.ml_target}"
                
                # For classification with categorical target, convert back to original labels
                if (st.session_state.ml_type == "Classification" and 
                    hasattr(st.session_state, 'target_classes')):
                    full_preds = np.array([st.session_state.target_classes[i] for i in full_preds])
                
                # Update the transformed dataset
                if "transformed_data" in st.session_state and st.session_state.transformed_data is not None:
                    st.session_state.transformed_data[prediction_column_name] = full_preds
                    st.success(f"Added predictions as column '{prediction_column_name}'")
                    
                    # Add to transformation history
                    if "transformation_history" in st.session_state:
                        st.session_state.transformation_history.append({
                            "type": "Machine Learning",
                            "details": f"Added predictions using {st.session_state.ml_model.__class__.__name__}",
                            "original_shape": (len(df), len(df.columns)),
                            "new_shape": (len(st.session_state.transformed_data), 
                                         len(st.session_state.transformed_data.columns))
                        })
                else:
                    st.error("Could not find transformed data to add predictions")

# Footer
st.markdown("---")
st.caption("Zahinn by NidoData | Machine Learning Module")
