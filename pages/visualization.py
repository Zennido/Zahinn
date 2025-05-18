import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Local module imports
from utils.visualization import (
    create_visualization, get_chart_type_options,
    plot_bar_chart, plot_line_chart, plot_scatter_chart,
    plot_histogram, plot_pie_chart, plot_box_plot, plot_heatmap
)

# Set page config
st.set_page_config(
    page_title="Visualization | Zahinn",
    page_icon="📊",
    layout="wide"
)

# Check if data is available in session state
if "data" not in st.session_state or st.session_state.data is None:
    st.error("No data loaded. Please upload a file on the main page.")
    st.stop()

# Header
st.title("Data Visualization")
st.write("Create insightful visualizations from your data.")

# Use transformed data if available, otherwise use original data
if "transformed_data" in st.session_state and st.session_state.transformed_data is not None:
    df = st.session_state.transformed_data
else:
    df = st.session_state.data

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Visualization Settings")
    
    # Chart type selection
    chart_types = get_chart_type_options()
    chart_type = st.selectbox("Chart Type", options=chart_types)
    
    # Column selection based on chart type
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    st.markdown("### Select Data")
    
    if chart_type == "Bar Chart":
        st.info("Bar charts are good for comparing categories.")
        x_column = st.selectbox("X-axis (Categories)", options=all_columns)
        y_column = st.selectbox("Y-axis (Values)", options=numeric_columns) if numeric_columns else None
        color_column = st.selectbox("Color (Optional)", options=["None"] + categorical_columns)
        
        if color_column == "None":
            selected_columns = [x_column, y_column]
        else:
            selected_columns = [x_column, y_column, color_column]
    
    elif chart_type == "Line Chart":
        st.info("Line charts are good for showing trends over time or sequences.")
        x_column = st.selectbox("X-axis", options=all_columns)
        y_columns = st.multiselect("Y-axis (Values)", options=numeric_columns) if numeric_columns else None
        
        selected_columns = [x_column] + y_columns if y_columns else [x_column]
    
    elif chart_type == "Scatter Plot":
        st.info("Scatter plots show the relationship between two numerical variables.")
        x_column = st.selectbox("X-axis", options=numeric_columns) if numeric_columns else None
        y_column = st.selectbox("Y-axis", options=numeric_columns) if numeric_columns else None
        color_column = st.selectbox("Color by (Optional)", options=["None"] + all_columns)
        size_column = st.selectbox("Size by (Optional)", options=["None"] + numeric_columns)
        
        selected_columns = [x_column, y_column]
        if color_column != "None":
            selected_columns.append(color_column)
        if size_column != "None":
            selected_columns.append(size_column)
    
    elif chart_type == "Histogram":
        st.info("Histograms show the distribution of a single numerical variable.")
        column = st.selectbox("Column", options=numeric_columns) if numeric_columns else None
        bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
        
        selected_columns = [column]
    
    elif chart_type == "Pie Chart":
        st.info("Pie charts show the proportion of each category in a total.")
        column = st.selectbox("Column (Categories)", options=categorical_columns) if categorical_columns else None
        
        selected_columns = [column]
    
    elif chart_type == "Box Plot":
        st.info("Box plots show the distribution of a numerical variable, optionally across categories.")
        y_column = st.selectbox("Values", options=numeric_columns) if numeric_columns else None
        x_column = st.selectbox("Group by (Optional)", options=["None"] + categorical_columns)
        
        if x_column == "None":
            selected_columns = [y_column]
        else:
            selected_columns = [y_column, x_column]
    
    elif chart_type == "Heatmap":
        st.info("Heatmaps show the correlation between numerical variables.")
        selected_columns = numeric_columns if numeric_columns else []
    
    # Chart title
    chart_title = st.text_input("Chart Title", f"{chart_type} of {df.columns[0] if not df.empty else ''}")
    
    # Additional chart options
    st.markdown("### Chart Options")
    show_data = st.checkbox("Show data table", value=True)

with col2:
    st.subheader("Visualization Preview")
    
    # Data info
    st.info(f"Data: {len(df)} rows, {len(df.columns)} columns")
    
    # Create visualization
    if all(col in df.columns for col in selected_columns if col is not None):
        with st.spinner("Creating visualization..."):
            create_visualization(df, chart_type, selected_columns, chart_title)
    else:
        missing_cols = [col for col in selected_columns if col is not None and col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            st.error("Please select valid columns for the visualization.")
    
    # Show data table if selected
    if show_data and not df.empty:
        st.markdown("### Data Table")
        st.dataframe(df.head(10), use_container_width=True)
    
    # Show visualization tips
    st.markdown("### Visualization Tips")
    
    if chart_type == "Bar Chart":
        st.markdown("""
        - Bar charts are best for comparing categories
        - Vertical bars work better for fewer categories
        - Consider sorting the bars for better readability
        - Use color to highlight important categories or represent another variable
        """)
    
    elif chart_type == "Line Chart":
        st.markdown("""
        - Line charts are ideal for showing trends over time
        - They work best when the x-axis represents a continuous variable
        - Multiple lines can be compared but avoid too many (5-7 max)
        - Consider using a logarithmic scale if values vary greatly
        """)
    
    elif chart_type == "Scatter Plot":
        st.markdown("""
        - Scatter plots reveal relationships between two numerical variables
        - Look for patterns: linear, clustered, random
        - Use color or size to add additional dimensions
        - Consider adding a trend line for clearer patterns
        """)
    
    elif chart_type == "Histogram":
        st.markdown("""
        - Histograms show the distribution and frequency of a single variable
        - Adjust the number of bins to show more or less detail
        - Look for normal distributions, skewness, or multiple peaks
        - Use to identify outliers or unusual patterns
        """)
    
    elif chart_type == "Pie Chart":
        st.markdown("""
        - Pie charts show proportions of a whole
        - Best when you have fewer than 7 categories
        - Ensure categories add up to 100%
        - Consider a bar chart if precise comparisons are important
        """)
    
    elif chart_type == "Box Plot":
        st.markdown("""
        - Box plots show statistical distribution of data
        - The box shows quartiles (25%, median, 75%)
        - Whiskers show variability outside quartiles
        - Points beyond whiskers are potential outliers
        - Good for comparing distributions across categories
        """)
    
    elif chart_type == "Heatmap":
        st.markdown("""
        - Heatmaps show correlations between numerical variables
        - Darker colors usually indicate stronger correlations
        - Perfect correlation with self (diagonal) is always 1
        - Look for strong positive (close to 1) or negative (close to -1) correlations
        - Use to identify relationships for further analysis
        """)

# Footer
st.markdown("---")
st.caption("Zahinn by NidoData | Visualization Module")
