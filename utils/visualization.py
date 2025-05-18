import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from typing import List, Tuple, Dict, Any, Optional

def preview_data(df: pd.DataFrame, rows: int = 10) -> None:
    """
    Display a preview of the data with pagination
    
    Args:
        df: The pandas DataFrame
        rows: Number of rows to display per page
    """
    # Display interactive dataframe with pagination
    st.dataframe(df.head(rows), use_container_width=True)

def plot_bar_chart(df: pd.DataFrame, x_column: str, y_column: str, 
                  title: str = "Bar Chart", color_column: Optional[str] = None) -> None:
    """
    Create a bar chart visualization
    
    Args:
        df: The pandas DataFrame
        x_column: Column to use for x-axis
        y_column: Column to use for y-axis
        title: Chart title
        color_column: Optional column to use for color encoding
    """
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        st.error("Invalid data or columns for bar chart")
        return
        
    if color_column and color_column in df.columns:
        fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
    else:
        fig = px.bar(df, x=x_column, y=y_column, title=title)
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_line_chart(df: pd.DataFrame, x_column: str, y_columns: List[str], 
                   title: str = "Line Chart") -> None:
    """
    Create a line chart visualization
    
    Args:
        df: The pandas DataFrame
        x_column: Column to use for x-axis
        y_columns: List of columns to plot as lines
        title: Chart title
    """
    if df.empty or x_column not in df.columns or not all(col in df.columns for col in y_columns):
        st.error("Invalid data or columns for line chart")
        return
    
    fig = go.Figure()
    
    for y_col in y_columns:
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df[y_col],
            mode='lines+markers',
            name=y_col
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Values",
        height=500,
        legend_title="Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_chart(df: pd.DataFrame, x_column: str, y_column: str, 
                      title: str = "Scatter Plot", color_column: Optional[str] = None,
                      size_column: Optional[str] = None) -> None:
    """
    Create a scatter plot visualization
    
    Args:
        df: The pandas DataFrame
        x_column: Column to use for x-axis
        y_column: Column to use for y-axis
        title: Chart title
        color_column: Optional column to use for color encoding
        size_column: Optional column to use for point size
    """
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        st.error("Invalid data or columns for scatter plot")
        return
    
    plot_args = {
        "x": x_column,
        "y": y_column,
        "title": title,
    }
    
    if color_column and color_column in df.columns:
        plot_args["color"] = color_column
    
    if size_column and size_column in df.columns:
        plot_args["size"] = size_column
    
    fig = px.scatter(df, **plot_args)
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 20, 
                  title: str = "Histogram") -> None:
    """
    Create a histogram visualization
    
    Args:
        df: The pandas DataFrame
        column: Column to plot
        bins: Number of bins
        title: Chart title
    """
    if df.empty or column not in df.columns:
        st.error("Invalid data or column for histogram")
        return
    
    fig = px.histogram(df, x=column, nbins=bins, title=title)
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(df: pd.DataFrame, column: str, title: str = "Pie Chart") -> None:
    """
    Create a pie chart visualization
    
    Args:
        df: The pandas DataFrame
        column: Column to plot
        title: Chart title
    """
    if df.empty or column not in df.columns:
        st.error("Invalid data or column for pie chart")
        return
    
    # Get value counts for the column
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    fig = px.pie(value_counts, values='count', names=column, title=title)
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> None:
    """
    Create a correlation heatmap visualization
    
    Args:
        df: The pandas DataFrame
        title: Chart title
    """
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        st.error("Not enough numeric columns for correlation heatmap")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=title
    )
    
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_box_plot(df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                 title: str = "Box Plot") -> None:
    """
    Create a box plot visualization
    
    Args:
        df: The pandas DataFrame
        column: Column to plot
        group_by: Optional column to group by
        title: Chart title
    """
    if df.empty or column not in df.columns:
        st.error("Invalid data or column for box plot")
        return
    
    if group_by and group_by in df.columns:
        fig = px.box(df, x=group_by, y=column, title=title)
    else:
        fig = px.box(df, y=column, title=title)
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)

def get_chart_type_options() -> List[str]:
    """
    Returns a list of available chart types
    
    Returns:
        List of chart type options
    """
    return [
        "Bar Chart",
        "Line Chart",
        "Scatter Plot",
        "Histogram",
        "Pie Chart",
        "Box Plot",
        "Heatmap"
    ]

def create_visualization(df: pd.DataFrame, chart_type: str, 
                        columns: List[str], title: str = "") -> None:
    """
    Create a visualization based on the selected chart type
    
    Args:
        df: The pandas DataFrame
        chart_type: Type of chart to create
        columns: Columns to use in the chart
        title: Chart title
    """
    if df.empty:
        st.error("No data available for visualization")
        return
    
    if chart_type == "Bar Chart":
        if len(columns) < 2:
            st.error("Bar chart requires at least 2 columns (x and y)")
        else:
            color_col = columns[2] if len(columns) > 2 else None
            plot_bar_chart(df, columns[0], columns[1], title, color_col)
            
    elif chart_type == "Line Chart":
        if len(columns) < 2:
            st.error("Line chart requires at least 2 columns (x and y values)")
        else:
            plot_line_chart(df, columns[0], columns[1:], title)
            
    elif chart_type == "Scatter Plot":
        if len(columns) < 2:
            st.error("Scatter plot requires at least 2 columns (x and y)")
        else:
            color_col = columns[2] if len(columns) > 2 else None
            size_col = columns[3] if len(columns) > 3 else None
            plot_scatter_chart(df, columns[0], columns[1], title, color_col, size_col)
            
    elif chart_type == "Histogram":
        if len(columns) < 1:
            st.error("Histogram requires at least 1 column")
        else:
            plot_histogram(df, columns[0], title=title)
            
    elif chart_type == "Pie Chart":
        if len(columns) < 1:
            st.error("Pie chart requires at least 1 column")
        else:
            plot_pie_chart(df, columns[0], title)
            
    elif chart_type == "Box Plot":
        if len(columns) < 1:
            st.error("Box plot requires at least 1 column")
        else:
            group_by = columns[1] if len(columns) > 1 else None
            plot_box_plot(df, columns[0], group_by, title)
            
    elif chart_type == "Heatmap":
        plot_heatmap(df, title)
        
    else:
        st.error(f"Unsupported chart type: {chart_type}")
