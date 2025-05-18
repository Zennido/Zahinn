import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, Dict, List, Optional, Union, Any

def validate_file(file) -> Tuple[bool, str]:
    """
    Validates if the uploaded file can be processed
    
    Args:
        file: The uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file size (max 200MB for safety)
        if file.size > 200 * 1024 * 1024:
            return False, "File too large (max 200MB)"
        
        # Check file type
        file_type = get_file_type(file.name)
        if file_type == "unknown":
            return False, "Unsupported file format. Please upload CSV or Excel files."
        
        # Try to read a small sample to validate format
        try:
            # Reset file pointer to beginning
            file.seek(0)
            
            if file_type == "csv":
                pd.read_csv(file, nrows=5)
            elif file_type in ["xlsx", "xls"]:
                pd.read_excel(file, nrows=5)
                
            # Reset file pointer to beginning
            file.seek(0)
            return True, ""
        except Exception as e:
            return False, f"File format error: {str(e)}"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_file_type(filename: str) -> str:
    """
    Determines file type from filename
    
    Args:
        filename: Name of the file
    
    Returns:
        String file type: "csv", "xlsx", "xls", or "unknown"
    """
    lower_filename = filename.lower()
    if lower_filename.endswith('.csv'):
        return "csv"
    elif lower_filename.endswith('.xlsx'):
        return "xlsx"
    elif lower_filename.endswith('.xls'):
        return "xls"
    else:
        return "unknown"

def load_data(file) -> Tuple[pd.DataFrame, str]:
    """
    Loads data from uploaded file into a pandas DataFrame
    
    Args:
        file: The uploaded file object
    
    Returns:
        Tuple of (dataframe, file_type)
    """
    file_type = get_file_type(file.name)
    
    try:
        if file_type == "csv":
            # Try to detect encoding and delimiter
            df = pd.read_csv(file)
        elif file_type in ["xlsx", "xls"]:
            # Let user pick sheet if multiple are available
            xls = pd.ExcelFile(file)
            if len(xls.sheet_names) > 1:
                sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
            else:
                sheet_name = xls.sheet_names[0]
                
            df = pd.read_excel(file, sheet_name=sheet_name)
        
        # Do some basic cleaning
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # Try to convert numeric columns to proper types
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Try to convert to numeric, keeping NaN values
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
                
        return df, file_type
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), file_type

def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes basic statistics for a dataframe
    
    Args:
        df: The pandas DataFrame
    
    Returns:
        DataFrame with statistics
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        # Calculate statistics for numeric columns
        stats = df[numeric_cols].describe().T
        
        # Add additional metrics
        stats['missing'] = df[numeric_cols].isnull().sum()
        stats['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
        
        return stats
    else:
        # Create an empty DataFrame with appropriate columns if no numeric columns
        return pd.DataFrame(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing', 'missing_pct'])

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes columns by their data types
    
    Args:
        df: The pandas DataFrame
    
    Returns:
        Dictionary with column types
    """
    column_types = {
        'numeric': list(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean': list(df.select_dtypes(include=['bool']).columns)
    }
    
    return column_types

def sample_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Return a sample of the dataframe
    
    Args:
        df: The pandas DataFrame
        n_rows: Number of rows to sample (default=5)
    
    Returns:
        Sample DataFrame
    """
    if len(df) <= n_rows:
        return df
    else:
        return df.sample(n_rows)

def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to infer and convert columns to appropriate data types
    
    Args:
        df: The pandas DataFrame
    
    Returns:
        DataFrame with converted types
    """
    df_copy = df.copy()
    
    # Try to convert object columns to datetime
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Check if more than 70% of non-null values can be parsed as dates
        non_null_values = df_copy[col].dropna()
        if len(non_null_values) > 0:
            try:
                pd.to_datetime(non_null_values, errors='raise')
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except:
                pass
    
    # Try to convert object columns to numeric
    for col in df_copy.select_dtypes(include=['object']).columns:
        try:
            numeric_values = pd.to_numeric(df_copy[col], errors='coerce')
            # If at least 90% of values converted successfully, use the conversion
            if numeric_values.notnull().sum() / df_copy[col].count() > 0.9:
                df_copy[col] = numeric_values
        except:
            pass
            
    # Try to convert object columns with few unique values to categorical
    for col in df_copy.select_dtypes(include=['object']).columns:
        unique_count = df_copy[col].nunique()
        if unique_count < 50 and unique_count < len(df_copy) * 0.1:
            df_copy[col] = df_copy[col].astype('category')
    
    return df_copy
