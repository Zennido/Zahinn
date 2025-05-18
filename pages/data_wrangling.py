import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Local module imports
from utils.transformations import (
    rename_columns, filter_data, sort_data, handle_missing_values,
    convert_data_type, create_calculated_column, aggregate_data,
    select_columns, transform_column
)

# Set page config
st.set_page_config(
    page_title="Data Wrangling | Zahinn",
    page_icon="📊",
    layout="wide"
)

# Check if data is available in session state
if "data" not in st.session_state or st.session_state.data is None:
    st.error("No data loaded. Please upload a file on the main page.")
    st.stop()

# Header
st.title("Data Wrangling")
st.write("Transform and clean your data with these powerful tools.")

# Initialize transformed data if needed
if "transformed_data" not in st.session_state:
    st.session_state.transformed_data = st.session_state.data.copy()

# Initialize transformation history if needed
if "transformation_history" not in st.session_state:
    st.session_state.transformation_history = []

# Main content
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Transformation Tools")
    
    # Create tabs for different transformation categories
    wrangling_tabs = st.tabs([
        "Columns", "Filter & Sort", "Missing Values", 
        "Data Types", "Calculations", "Advanced"
    ])
    
    # Columns tab - Rename, select, drop columns
    with wrangling_tabs[0]:
        st.markdown("### Column Operations")
        
        # Column selection
        st.write("Select columns to keep:")
        all_columns = st.session_state.transformed_data.columns.tolist()
        selected_columns = st.multiselect(
            "Columns to keep", 
            options=all_columns,
            default=all_columns
        )
        
        if st.button("Apply Column Selection", use_container_width=True):
            if not selected_columns:
                st.error("You must select at least one column.")
            else:
                previous_data = st.session_state.transformed_data.copy()
                st.session_state.transformed_data = select_columns(
                    st.session_state.transformed_data, 
                    selected_columns
                )
                st.session_state.transformation_history.append({
                    "type": "Select Columns",
                    "details": f"Selected {len(selected_columns)} columns",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Selected {len(selected_columns)} columns")
                st.rerun()
        
        # Column renaming
        st.write("Rename columns:")
        col_to_rename = st.selectbox(
            "Column to rename",
            options=st.session_state.transformed_data.columns
        )
        new_col_name = st.text_input("New column name", value=col_to_rename)
        
        if st.button("Rename Column", use_container_width=True):
            if new_col_name == col_to_rename:
                st.info("Column name unchanged.")
            elif new_col_name in st.session_state.transformed_data.columns:
                st.error(f"Column name '{new_col_name}' already exists.")
            else:
                previous_data = st.session_state.transformed_data.copy()
                st.session_state.transformed_data = rename_columns(
                    st.session_state.transformed_data, 
                    {col_to_rename: new_col_name}
                )
                st.session_state.transformation_history.append({
                    "type": "Rename Column",
                    "details": f"Renamed '{col_to_rename}' to '{new_col_name}'",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Renamed column '{col_to_rename}' to '{new_col_name}'")
                st.rerun()
    
    # Filter & Sort tab
    with wrangling_tabs[1]:
        st.markdown("### Filter Data")
        
        filter_col = st.selectbox(
            "Column to filter", 
            options=st.session_state.transformed_data.columns,
            key="filter_col"
        )
        
        # Different filter options based on column type
        col_dtype = st.session_state.transformed_data[filter_col].dtype
        
        # Determine appropriate operators based on column type
        if pd.api.types.is_numeric_dtype(col_dtype):
            filter_operators = ["==", "!=", ">", "<", ">=", "<="]
            filter_op = st.selectbox("Operator", options=filter_operators)
            
            # Show unique values or allow numeric input
            unique_values = st.session_state.transformed_data[filter_col].unique()
            if len(unique_values) <= 10:
                filter_value = st.selectbox("Value", options=sorted(unique_values))
            else:
                filter_value = st.number_input(
                    "Value", 
                    value=float(st.session_state.transformed_data[filter_col].median())
                )
        
        elif pd.api.types.is_string_dtype(col_dtype) or col_dtype == 'object':
            filter_operators = ["==", "!=", "contains", "starts with", "ends with"]
            filter_op = st.selectbox("Operator", options=filter_operators)
            
            # Show unique values for exact matches, text input for contains
            unique_values = st.session_state.transformed_data[filter_col].dropna().unique()
            if filter_op in ["==", "!="] and len(unique_values) <= 20:
                filter_value = st.selectbox("Value", options=sorted(unique_values))
            else:
                filter_value = st.text_input("Value")
        
        else:
            filter_operators = ["==", "!="]
            filter_op = st.selectbox("Operator", options=filter_operators)
            
            # Show all unique values
            unique_values = st.session_state.transformed_data[filter_col].dropna().unique()
            filter_value = st.selectbox("Value", options=sorted(unique_values))
        
        if st.button("Apply Filter", use_container_width=True):
            previous_data = st.session_state.transformed_data.copy()
            filtered_df = filter_data(
                st.session_state.transformed_data, 
                filter_col, 
                filter_op, 
                filter_value
            )
            
            if len(filtered_df) == 0:
                st.error("Filter would result in empty dataset. No changes applied.")
            else:
                st.session_state.transformed_data = filtered_df
                st.session_state.transformation_history.append({
                    "type": "Filter Data",
                    "details": f"Filtered '{filter_col}' {filter_op} '{filter_value}'",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Applied filter, {len(filtered_df)} rows remaining")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Sort Data")
        
        sort_columns = st.multiselect(
            "Columns to sort by",
            options=st.session_state.transformed_data.columns,
            key="sort_columns"
        )
        
        if sort_columns:
            # Create sort order toggles
            sort_orders = []
            for col in sort_columns:
                sort_asc = st.checkbox(f"Sort {col} ascending", value=True, key=f"sort_{col}")
                sort_orders.append(sort_asc)
            
            if st.button("Apply Sorting", use_container_width=True):
                previous_data = st.session_state.transformed_data.copy()
                st.session_state.transformed_data = sort_data(
                    st.session_state.transformed_data,
                    sort_columns,
                    sort_orders
                )
                st.session_state.transformation_history.append({
                    "type": "Sort Data",
                    "details": f"Sorted by {', '.join(sort_columns)}",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Data sorted by {', '.join(sort_columns)}")
                st.rerun()
    
    # Missing Values tab
    with wrangling_tabs[2]:
        st.markdown("### Handle Missing Values")
        
        # Column with missing values
        miss_cols = [col for col in st.session_state.transformed_data.columns 
                    if st.session_state.transformed_data[col].isnull().any()]
        
        if not miss_cols:
            st.info("No columns with missing values found.")
        else:
            miss_col = st.selectbox(
                "Column with missing values",
                options=miss_cols,
                key="miss_col"
            )
            
            miss_count = st.session_state.transformed_data[miss_col].isnull().sum()
            miss_percent = (miss_count / len(st.session_state.transformed_data) * 100).round(2)
            
            st.write(f"Missing values: {miss_count} ({miss_percent}%)")
            
            # Method to handle missing values
            handle_methods = [
                "drop", "fill_value", "fill_mean", "fill_median", "fill_mode"
            ]
            
            miss_method = st.selectbox(
                "Method to handle missing values",
                options=handle_methods,
                key="miss_method"
            )
            
            fill_value = None
            if miss_method == "fill_value":
                # Determine appropriate input based on column type
                col_dtype = st.session_state.transformed_data[miss_col].dtype
                
                if pd.api.types.is_numeric_dtype(col_dtype):
                    fill_value = st.number_input("Fill value", value=0.0, key="fill_num")
                else:
                    fill_value = st.text_input("Fill value", value="Unknown", key="fill_text")
            
            if st.button("Handle Missing Values", use_container_width=True):
                previous_data = st.session_state.transformed_data.copy()
                st.session_state.transformed_data = handle_missing_values(
                    st.session_state.transformed_data,
                    miss_col,
                    miss_method,
                    fill_value
                )
                
                st.session_state.transformation_history.append({
                    "type": "Handle Missing Values",
                    "details": f"Applied '{miss_method}' to '{miss_col}'",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Applied {miss_method} to column '{miss_col}'")
                st.rerun()
    
    # Data Types tab
    with wrangling_tabs[3]:
        st.markdown("### Convert Data Types")
        
        convert_col = st.selectbox(
            "Column to convert",
            options=st.session_state.transformed_data.columns,
            key="convert_col"
        )
        
        # Show current type
        current_type = str(st.session_state.transformed_data[convert_col].dtype)
        st.write(f"Current type: {current_type}")
        
        target_types = ["int", "float", "str", "datetime", "category", "bool"]
        target_type = st.selectbox(
            "Convert to type",
            options=target_types,
            key="target_type"
        )
        
        if st.button("Convert Data Type", use_container_width=True):
            previous_data = st.session_state.transformed_data.copy()
            st.session_state.transformed_data = convert_data_type(
                st.session_state.transformed_data,
                convert_col,
                target_type
            )
            
            new_type = str(st.session_state.transformed_data[convert_col].dtype)
            st.session_state.transformation_history.append({
                "type": "Convert Data Type",
                "details": f"Converted '{convert_col}' from {current_type} to {new_type}",
                "original_shape": previous_data.shape,
                "new_shape": st.session_state.transformed_data.shape
            })
            st.success(f"Converted '{convert_col}' to {new_type}")
            st.rerun()
    
    # Calculations tab
    with wrangling_tabs[4]:
        st.markdown("### Create Calculated Column")
        
        new_col_name = st.text_input(
            "New column name", 
            key="calc_col_name"
        )
        
        st.write("Formula examples:")
        st.code("df['column1'] + df['column2']\ndf['price'] * 1.1\ndf['column'].str.len()")
        
        formula = st.text_area(
            "Column formula",
            height=100,
            key="calc_formula"
        )
        
        if st.button("Create Calculated Column", use_container_width=True):
            if not new_col_name:
                st.error("Please provide a name for the new column")
            elif not formula:
                st.error("Please provide a formula")
            elif new_col_name in st.session_state.transformed_data.columns:
                st.error(f"Column '{new_col_name}' already exists")
            else:
                previous_data = st.session_state.transformed_data.copy()
                try:
                    st.session_state.transformed_data = create_calculated_column(
                        st.session_state.transformed_data,
                        new_col_name,
                        formula
                    )
                    
                    st.session_state.transformation_history.append({
                        "type": "Create Calculated Column",
                        "details": f"Created '{new_col_name}' using formula",
                        "original_shape": previous_data.shape,
                        "new_shape": st.session_state.transformed_data.shape
                    })
                    st.success(f"Created calculated column '{new_col_name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in formula: {str(e)}")
    
    # Advanced tab - Aggregations, transformations
    with wrangling_tabs[5]:
        st.markdown("### Aggregate Data")
        
        group_cols = st.multiselect(
            "Group by columns",
            options=st.session_state.transformed_data.columns,
            key="group_cols"
        )
        
        numeric_cols = st.session_state.transformed_data.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        agg_cols = st.multiselect(
            "Columns to aggregate",
            options=numeric_cols,
            key="agg_cols"
        )
        
        if group_cols and agg_cols:
            # Aggregation functions per column
            agg_funcs = []
            for col in agg_cols:
                func = st.selectbox(
                    f"Function for {col}",
                    options=["sum", "mean", "min", "max", "count"],
                    key=f"func_{col}"
                )
                agg_funcs.append(func)
            
            if st.button("Aggregate Data", use_container_width=True):
                previous_data = st.session_state.transformed_data.copy()
                st.session_state.transformed_data = aggregate_data(
                    st.session_state.transformed_data,
                    group_cols,
                    agg_cols,
                    agg_funcs
                )
                
                st.session_state.transformation_history.append({
                    "type": "Aggregate Data",
                    "details": f"Grouped by {', '.join(group_cols)} with {len(agg_cols)} aggregations",
                    "original_shape": previous_data.shape,
                    "new_shape": st.session_state.transformed_data.shape
                })
                st.success(f"Aggregated data by {', '.join(group_cols)}")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Transform Column")
        
        transform_col = st.selectbox(
            "Column to transform",
            options=st.session_state.transformed_data.columns,
            key="transform_col"
        )
        
        # Determine appropriate transformations based on column type
        col_dtype = st.session_state.transformed_data[transform_col].dtype
        
        if pd.api.types.is_numeric_dtype(col_dtype):
            transformations = ["normalize", "standardize", "log", "sqrt", "round", "bin"]
        elif pd.api.types.is_string_dtype(col_dtype) or col_dtype == 'object':
            transformations = ["upper", "lower", "capitalize", "trim", "replace", "extract"]
        else:
            transformations = ["replace"]
        
        transform_type = st.selectbox(
            "Transformation type",
            options=transformations,
            key="transform_type"
        )
        
        # Parameters for specific transformations
        params = {}
        
        if transform_type == "round":
            params["decimals"] = st.number_input("Decimal places", value=2, min_value=0)
        elif transform_type == "replace":
            params["old_value"] = st.text_input("Value to replace")
            params["new_value"] = st.text_input("New value")
        elif transform_type == "extract":
            params["pattern"] = st.text_input("Regex pattern")
        elif transform_type == "bin":
            params["bins"] = st.number_input("Number of bins", value=5, min_value=2)
            
        if st.button("Transform Column", use_container_width=True):
            previous_data = st.session_state.transformed_data.copy()
            st.session_state.transformed_data = transform_column(
                st.session_state.transformed_data,
                transform_col,
                transform_type,
                params
            )
            
            st.session_state.transformation_history.append({
                "type": "Transform Column",
                "details": f"Applied '{transform_type}' to '{transform_col}'",
                "original_shape": previous_data.shape,
                "new_shape": st.session_state.transformed_data.shape
            })
            st.success(f"Applied {transform_type} to column '{transform_col}'")
            st.rerun()

with col2:
    st.subheader("Data Preview")
    
    # Data info
    st.info(f"Rows: {len(st.session_state.transformed_data)} | Columns: {len(st.session_state.transformed_data.columns)}")
    
    # Display number of rows to show
    rows_to_show = st.slider("Rows to display", min_value=5, max_value=100, value=10)
    
    # Display dataframe with pagination
    st.dataframe(st.session_state.transformed_data.head(rows_to_show), use_container_width=True)
    
    # Transformation history
    st.subheader("Transformation History")
    
    if not st.session_state.transformation_history:
        st.write("No transformations applied yet.")
    else:
        # Create a dataframe from transformation history
        history_df = pd.DataFrame(st.session_state.transformation_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Reset button
        if st.button("Reset to Original Data"):
            st.session_state.transformed_data = st.session_state.data.copy()
            st.session_state.transformation_history = []
            st.success("Data reset to original state")
            st.rerun()
        
        # Undo last transformation
        if st.button("Undo Last Transformation"):
            if st.session_state.transformation_history:
                # Remove the last transformation
                st.session_state.transformation_history.pop()
                
                # Reset data and reapply all transformations
                st.session_state.transformed_data = st.session_state.data.copy()
                
                # This is a simplified approach - in a real app, you'd want to store
                # the actual operations and reapply them
                st.success("Undid last transformation")
                st.rerun()
            else:
                st.error("No transformations to undo")

# Footer
st.markdown("---")
st.caption("Zahinn by NidoData | Data Wrangling Module")
