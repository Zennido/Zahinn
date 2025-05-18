import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union

def rename_columns(df: pd.DataFrame, renaming_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a DataFrame
    
    Args:
        df: The pandas DataFrame
        renaming_dict: Dictionary mapping old column names to new column names
    
    Returns:
        DataFrame with renamed columns
    """
    try:
        return df.rename(columns=renaming_dict)
    except Exception as e:
        st.error(f"Error renaming columns: {str(e)}")
        return df

def filter_data(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame:
    """
    Filter a DataFrame based on a condition
    
    Args:
        df: The pandas DataFrame
        column: Column to filter on
        operator: Filter operator ('==', '!=', '>', '<', '>=', '<=', 'contains', 'starts with', 'ends with')
        value: Value to filter against
    
    Returns:
        Filtered DataFrame
    """
    try:
        if column not in df.columns:
            st.error(f"Column '{column}' not found in the data")
            return df
            
        if operator == "==":
            return df[df[column] == value]
        elif operator == "!=":
            return df[df[column] != value]
        elif operator == ">":
            return df[df[column] > value]
        elif operator == "<":
            return df[df[column] < value]
        elif operator == ">=":
            return df[df[column] >= value]
        elif operator == "<=":
            return df[df[column] <= value]
        elif operator == "contains":
            if df[column].dtype == 'object':
                return df[df[column].str.contains(str(value), na=False)]
            else:
                st.error(f"Cannot use 'contains' operator on non-string column '{column}'")
                return df
        elif operator == "starts with":
            if df[column].dtype == 'object':
                return df[df[column].str.startswith(str(value), na=False)]
            else:
                st.error(f"Cannot use 'starts with' operator on non-string column '{column}'")
                return df
        elif operator == "ends with":
            if df[column].dtype == 'object':
                return df[df[column].str.endswith(str(value), na=False)]
            else:
                st.error(f"Cannot use 'ends with' operator on non-string column '{column}'")
                return df
        else:
            st.error(f"Unsupported operator: {operator}")
            return df
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return df

def sort_data(df: pd.DataFrame, columns: List[str], ascending: List[bool]) -> pd.DataFrame:
    """
    Sort a DataFrame by one or more columns
    
    Args:
        df: The pandas DataFrame
        columns: List of columns to sort by
        ascending: List of boolean values indicating sort order for each column
    
    Returns:
        Sorted DataFrame
    """
    try:
        # Validate that all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}")
            return df
        
        # Adjust ascending list if needed
        if len(ascending) < len(columns):
            ascending = ascending + [True] * (len(columns) - len(ascending))
        
        return df.sort_values(by=columns, ascending=ascending)
    except Exception as e:
        st.error(f"Error sorting data: {str(e)}")
        return df

def handle_missing_values(df: pd.DataFrame, column: str, method: str, 
                         value: Optional[Any] = None) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame column
    
    Args:
        df: The pandas DataFrame
        column: Column to process
        method: Method to handle missing values ('drop', 'fill_value', 'fill_mean', 'fill_median', 'fill_mode')
        value: Value to use if method is 'fill_value'
    
    Returns:
        DataFrame with handled missing values
    """
    try:
        df_copy = df.copy()
        
        if column not in df_copy.columns:
            st.error(f"Column '{column}' not found in the data")
            return df_copy
            
        if method == "drop":
            return df_copy.dropna(subset=[column])
        elif method == "fill_value":
            df_copy[column] = df_copy[column].fillna(value)
            return df_copy
        elif method == "fill_mean":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
            else:
                st.error(f"Cannot fill with mean for non-numeric column '{column}'")
            return df_copy
        elif method == "fill_median":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column] = df_copy[column].fillna(df_copy[column].median())
            else:
                st.error(f"Cannot fill with median for non-numeric column '{column}'")
            return df_copy
        elif method == "fill_mode":
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0] if not df_copy[column].mode().empty else None)
            return df_copy
        else:
            st.error(f"Unsupported method: {method}")
            return df_copy
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df

def convert_data_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """
    Convert a column to a different data type
    
    Args:
        df: The pandas DataFrame
        column: Column to convert
        target_type: Target data type ('int', 'float', 'str', 'datetime', 'category', 'bool')
    
    Returns:
        DataFrame with converted column
    """
    try:
        df_copy = df.copy()
        
        if column not in df_copy.columns:
            st.error(f"Column '{column}' not found in the data")
            return df_copy
            
        if target_type == "int":
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce').fillna(0).astype(int)
        elif target_type == "float":
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        elif target_type == "str":
            df_copy[column] = df_copy[column].astype(str)
        elif target_type == "datetime":
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        elif target_type == "category":
            df_copy[column] = df_copy[column].astype('category')
        elif target_type == "bool":
            # Try to handle common boolean representations
            df_copy[column] = df_copy[column].map({
                'true': True, 'True': True, 'TRUE': True, 'yes': True, 'Yes': True, 'YES': True, '1': True, 1: True,
                'false': False, 'False': False, 'FALSE': False, 'no': False, 'No': False, 'NO': False, '0': False, 0: False
            }, na_action='ignore')
            df_copy[column] = df_copy[column].astype('boolean')
        else:
            st.error(f"Unsupported data type: {target_type}")
        
        return df_copy
    except Exception as e:
        st.error(f"Error converting data type: {str(e)}")
        return df

def create_calculated_column(df: pd.DataFrame, new_column: str, 
                            expression: str) -> pd.DataFrame:
    """
    Create a new column based on an expression
    
    Args:
        df: The pandas DataFrame
        new_column: Name for the new column
        expression: Python expression to calculate the new column
    
    Returns:
        DataFrame with new column added
    """
    try:
        df_copy = df.copy()
        
        # Create a safe local environment for evaluation
        local_dict = {'df': df_copy, 'np': np, 'pd': pd}
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, local_dict)
        
        # Add the new column
        df_copy[new_column] = result
        
        return df_copy
    except Exception as e:
        st.error(f"Error creating calculated column: {str(e)}")
        return df

def aggregate_data(df: pd.DataFrame, group_by_columns: List[str], 
                  agg_columns: List[str], agg_functions: List[str]) -> pd.DataFrame:
    """
    Aggregate data by grouping columns
    
    Args:
        df: The pandas DataFrame
        group_by_columns: Columns to group by
        agg_columns: Columns to aggregate
        agg_functions: Aggregation functions to apply
    
    Returns:
        Aggregated DataFrame
    """
    try:
        # Validate columns
        missing_cols = [col for col in group_by_columns + agg_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}")
            return df
            
        # Create aggregation dictionary
        agg_dict = {}
        for col, func in zip(agg_columns, agg_functions):
            if col in agg_dict:
                agg_dict[col].append(func)
            else:
                agg_dict[col] = [func]
        
        # Perform groupby and aggregation
        result = df.groupby(group_by_columns).agg(agg_dict)
        
        # Flatten column names
        result.columns = ['_'.join([col, func]) for col, func in zip(
            [col for col in agg_columns for _ in range(len(agg_dict[col]))],
            [func for funcs in agg_dict.values() for func in funcs]
        )]
        
        # Reset index to make grouped columns regular columns
        result = result.reset_index()
        
        return result
    except Exception as e:
        st.error(f"Error aggregating data: {str(e)}")
        return df

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Select specific columns from DataFrame
    
    Args:
        df: The pandas DataFrame
        columns: List of columns to select
    
    Returns:
        DataFrame with only selected columns
    """
    try:
        # Validate that all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}")
            return df
            
        return df[columns]
    except Exception as e:
        st.error(f"Error selecting columns: {str(e)}")
        return df

def transform_column(df: pd.DataFrame, column: str, 
                    transformation: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """
    Apply a transformation to a column
    
    Args:
        df: The pandas DataFrame
        column: Column to transform
        transformation: Type of transformation
        params: Additional parameters for the transformation
    
    Returns:
        DataFrame with transformed column
    """
    try:
        df_copy = df.copy()
        
        if column not in df_copy.columns:
            st.error(f"Column '{column}' not found in the data")
            return df_copy
            
        if transformation == "log":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                # Handle zeros and negative values
                min_val = df_copy[column].min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    df_copy[column] = np.log(df_copy[column] + offset)
                else:
                    df_copy[column] = np.log(df_copy[column])
            else:
                st.error(f"Cannot apply logarithm to non-numeric column '{column}'")
                
        elif transformation == "sqrt":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                # Handle negative values
                min_val = df_copy[column].min()
                if min_val < 0:
                    offset = abs(min_val)
                    df_copy[column] = np.sqrt(df_copy[column] + offset)
                else:
                    df_copy[column] = np.sqrt(df_copy[column])
            else:
                st.error(f"Cannot apply square root to non-numeric column '{column}'")
                
        elif transformation == "normalize":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                min_val = df_copy[column].min()
                max_val = df_copy[column].max()
                if min_val != max_val:  # Avoid division by zero
                    df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
            else:
                st.error(f"Cannot normalize non-numeric column '{column}'")
                
        elif transformation == "standardize":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                mean = df_copy[column].mean()
                std = df_copy[column].std()
                if std != 0:  # Avoid division by zero
                    df_copy[column] = (df_copy[column] - mean) / std
            else:
                st.error(f"Cannot standardize non-numeric column '{column}'")
                
        elif transformation == "round":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                decimals = params.get('decimals', 0)
                df_copy[column] = df_copy[column].round(decimals)
            else:
                st.error(f"Cannot round non-numeric column '{column}'")
                
        elif transformation == "upper":
            if df_copy[column].dtype == 'object':
                df_copy[column] = df_copy[column].str.upper()
            else:
                st.error(f"Cannot convert non-string column '{column}' to uppercase")
                
        elif transformation == "lower":
            if df_copy[column].dtype == 'object':
                df_copy[column] = df_copy[column].str.lower()
            else:
                st.error(f"Cannot convert non-string column '{column}' to lowercase")
                
        elif transformation == "capitalize":
            if df_copy[column].dtype == 'object':
                df_copy[column] = df_copy[column].str.capitalize()
            else:
                st.error(f"Cannot capitalize non-string column '{column}'")
                
        elif transformation == "trim":
            if df_copy[column].dtype == 'object':
                df_copy[column] = df_copy[column].str.strip()
            else:
                st.error(f"Cannot trim non-string column '{column}'")
                
        elif transformation == "replace":
            old_value = params.get('old_value', '')
            new_value = params.get('new_value', '')
            df_copy[column] = df_copy[column].replace(old_value, new_value)
                
        elif transformation == "extract":
            if df_copy[column].dtype == 'object':
                pattern = params.get('pattern', '')
                df_copy[column] = df_copy[column].str.extract(pattern, expand=False)
            else:
                st.error(f"Cannot extract from non-string column '{column}'")
                
        elif transformation == "bin":
            if pd.api.types.is_numeric_dtype(df_copy[column]):
                bins = params.get('bins', 5)
                labels = params.get('labels', None)
                df_copy[column] = pd.cut(df_copy[column], bins=bins, labels=labels)
            else:
                st.error(f"Cannot bin non-numeric column '{column}'")
                
        else:
            st.error(f"Unsupported transformation: {transformation}")
            
        return df_copy
    except Exception as e:
        st.error(f"Error transforming column: {str(e)}")
        return df
