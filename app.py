import pandas as pd
import numpy as np
from pathlib import Path
import os
import streamlit as st

# Local module imports
from utils.data_handling import load_data, validate_file, get_basic_stats
from utils.visualization import preview_data

# Set page config
st.set_page_config(
    page_title="Zahinn by NidoData",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if "data" not in st.session_state:
    st.session_state.data = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "transformed_data" not in st.session_state:
    st.session_state.transformed_data = None
if "transformation_history" not in st.session_state:
    st.session_state.transformation_history = []

# App header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://pixabay.com/get/gd2d81834d2cbce8ece6077511ea3f2ae01013407929c976ec72fd79af9627d61627832bb28a1e5877d5ed03b56150b3ba06f59ae589ec205c32223f5d5ad4525_1280.jpg", width=100)
with col2:
    st.title("Zahinn - Business Intelligence Platform")
    st.subheader("by NidoData")

# Main app tabs
tab1, tab2 = st.tabs(["Upload Data", "Data Overview"])

with tab1:
    st.header("Upload Your Data")
    st.write("Upload CSV or Excel files to start your data analysis journey.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Attempt to load and validate the file
        with st.spinner("Loading data..."):
            is_valid, error_msg = validate_file(uploaded_file)
            
            if is_valid:
                df, file_type = load_data(uploaded_file)
                
                # Store in session state
                st.session_state.data = df
                st.session_state.filename = uploaded_file.name
                st.session_state.file_type = file_type
                st.session_state.transformed_data = df.copy()
                
                st.success(f"Successfully loaded {uploaded_file.name}")
                
                # Show a preview
                st.subheader("Data Preview")
                st.dataframe(df.head(5), use_container_width=True)
                
                # Display basic information
                st.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
            else:
                st.error(f"Error loading file: {error_msg}")

with tab2:
    if st.session_state.data is not None:
        st.header(f"Overview: {st.session_state.filename}")
        
        # Dashboard overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Statistics")
            stats_df = get_basic_stats(st.session_state.data)
            st.dataframe(stats_df, use_container_width=True)
            
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame(
                st.session_state.data.dtypes, 
                columns=["Data Type"]
            )
            st.dataframe(dtypes_df, use_container_width=True)
        
        with col2:
            st.subheader("Data Preview")
            preview_data(st.session_state.data)
            
            # Number of missing values
            st.subheader("Missing Values")
            missing = pd.DataFrame(
                st.session_state.data.isnull().sum(), 
                columns=["Missing Values"]
            )
            missing["Percentage"] = (missing["Missing Values"] / len(st.session_state.data) * 100).round(2)
            st.dataframe(missing, use_container_width=True)
    else:
        st.image("https://pixabay.com/get/g641d262599e7d30dfe5bf2575448384a652b6f3e6aefbadff8232fe4abeeb5346a73466714d7230943a77cd2d2e46efb5909c06899a434ad93975984de7fcd0f_1280.jpg", width=600)
        st.info("Please upload a file in the 'Upload Data' tab to see an overview.")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data loaded: {st.session_state.filename}")
    st.sidebar.info(f"📊 Rows: {len(st.session_state.data)} | Columns: {len(st.session_state.data.columns)}")
    
    # Navigation options
    st.sidebar.markdown("### Analysis Tools")
    if st.sidebar.button("Data Wrangling", use_container_width=True):
        st.switch_page("pages/data_wrangling.py")
    if st.sidebar.button("Visualizations", use_container_width=True):
        st.switch_page("pages/visualization.py")
    if st.sidebar.button("Machine Learning", use_container_width=True):
        st.switch_page("pages/machine_learning.py")
    if st.sidebar.button("Export Data", use_container_width=True):
        st.switch_page("pages/export.py")
    if st.sidebar.button("About", use_container_width=True):
        st.switch_page("pages/about.py")
else:
    st.sidebar.warning("⚠️ No data loaded")
    
    # About button even without data
    if st.sidebar.button("About Zahinn", use_container_width=True):
        st.switch_page("pages/about.py")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Zahinn by NidoData")
st.sidebar.caption("Open Source Business Intelligence")
