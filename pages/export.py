import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from pathlib import Path
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Export Data | Zahinn",
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
    data_source = "transformed"
else:
    df = st.session_state.data
    data_source = "original"

# Header
st.title("Export Data")
st.write("Export your data in various formats for further use.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Export Settings")
    
    # File format selection
    export_format = st.radio(
        "Select Export Format", 
        options=["CSV", "Excel", "JSON", "HTML"]
    )
    
    # Export options
    st.markdown("### Export Options")
    
    # Common options
    include_index = st.checkbox("Include row indices", value=False)
    
    # Format-specific options
    if export_format == "CSV":
        csv_separator = st.selectbox(
            "Column separator", 
            options=[",", ";", "\\t"],
            format_func=lambda x: {"\\t": "Tab", ",": "Comma", ";": "Semicolon"}[x]
        )
        decimal_separator = st.selectbox(
            "Decimal separator", 
            options=[".", ","]
        )
    
    elif export_format == "Excel":
        sheet_name = st.text_input("Sheet name", value="Data")
    
    elif export_format == "JSON":
        json_orient = st.selectbox(
            "JSON orientation", 
            options=["records", "columns", "index", "split", "table"],
            format_func=lambda x: {
                "records": "Records - List of dictionaries", 
                "columns": "Columns - Dictionary of lists", 
                "index": "Index - Dictionary with index", 
                "split": "Split - Dictionary with index, columns, and data",
                "table": "Table - Table schema"
            }[x]
        )
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Data info
    st.info(f"Exporting {data_source} data with {len(df)} rows and {len(df.columns)} columns")

with col2:
    st.subheader("Download")
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default filename based on source and format
    if "filename" in st.session_state:
        base_filename = os.path.splitext(st.session_state.filename)[0]
    else:
        base_filename = "zahinn_data"
    
    filename = f"{base_filename}_{timestamp}"
    
    # Custom filename
    custom_filename = st.text_input("Custom filename (without extension)", value=filename)
    
    # Generate download button based on format
    if export_format == "CSV":
        def get_csv_download_link(df, filename, separator, decimal, index):
            csv_kwargs = {
                "sep": separator,
                "decimal": decimal,
                "index": index
            }
            
            csv = df.to_csv(**csv_kwargs)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
            return href
        
        # Convert tab separator for CSV export
        actual_separator = "\t" if csv_separator == "\\t" else csv_separator
        
        st.markdown(
            get_csv_download_link(
                df, 
                custom_filename, 
                actual_separator, 
                decimal_separator, 
                include_index
            ), 
            unsafe_allow_html=True
        )
        
        st.info(f"Click the link above to download the data as a CSV file")
    
    elif export_format == "Excel":
        def get_excel_download_link(df, filename, sheet_name, index):
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)
            
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel file</a>'
            return href
        
        st.markdown(
            get_excel_download_link(
                df, 
                custom_filename, 
                sheet_name, 
                include_index
            ), 
            unsafe_allow_html=True
        )
        
        st.info(f"Click the link above to download the data as an Excel file")
    
    elif export_format == "JSON":
        def get_json_download_link(df, filename, orient, index):
            json = df.to_json(orient=orient, index=index)
            b64 = base64.b64encode(json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download JSON file</a>'
            return href
        
        st.markdown(
            get_json_download_link(
                df, 
                custom_filename, 
                json_orient, 
                include_index
            ), 
            unsafe_allow_html=True
        )
        
        st.info(f"Click the link above to download the data as a JSON file")
    
    elif export_format == "HTML":
        def get_html_download_link(df, filename, index):
            html = df.to_html(index=index)
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">Download HTML file</a>'
            return href
        
        st.markdown(
            get_html_download_link(
                df, 
                custom_filename, 
                include_index
            ), 
            unsafe_allow_html=True
        )
        
        st.info(f"Click the link above to download the data as an HTML file")
    
    # Export metadata
    st.markdown("### Export Metadata")
    
    metadata = {
        "Source filename": st.session_state.get("filename", "Unknown"),
        "Export timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Export format": export_format,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Data source": data_source,
        "Column types": ", ".join([f"{col}: {df[col].dtype}" for col in df.columns[:5]]) + 
                        ("..." if len(df.columns) > 5 else "")
    }
    
    for key, value in metadata.items():
        st.text(f"{key}: {value}")
    
    st.markdown("---")
    
    # Show data transformation report if applicable
    if data_source == "transformed" and "transformation_history" in st.session_state:
        st.subheader("Transformation History")
        
        if not st.session_state.transformation_history:
            st.write("No transformations have been applied.")
        else:
            for i, transform in enumerate(st.session_state.transformation_history):
                st.write(f"{i+1}. {transform['type']}: {transform['details']}")
                if 'original_shape' in transform and 'new_shape' in transform:
                    st.caption(f"Changed shape from {transform['original_shape']} to {transform['new_shape']}")

# Footer
st.markdown("---")
st.caption("Zahinn by NidoData | Export Module")
