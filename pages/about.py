import streamlit as st

# Set page config
st.set_page_config(
    page_title="About Zahinn",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("About Zahinn")
st.subheader("Open Source Business Intelligence Platform by NidoData")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## What is Zahinn?
    
    Zahinn is an open-source business intelligence tool that enables users to upload, analyze, transform, and visualize their data with ease. Whether you're a data analyst, business user, or data scientist, Zahinn provides a comprehensive set of tools to help you derive insights from your data.
    
    ## Key Features
    
    - **File Upload**: Upload CSV and Excel files with ease
    - **Data Preview**: Get a quick look at your data and basic statistics
    - **Data Wrangling**: Powerful tools for cleaning and transforming your data
      - Column selection and renaming
      - Data type conversion
      - Handling missing values
      - Filtering and sorting
      - Custom calculations
      - Aggregations
    - **Visualization**: Create beautiful and informative visualizations
      - Bar charts
      - Line charts
      - Scatter plots
      - Histograms
      - Pie charts
      - Box plots
      - Heatmaps
    - **Export**: Save your transformed data for further use
    - **Machine Learning**: Apply predictive analytics to your business data
      - Linear and Logistic Regression
      - Decision Trees
      - Random Forest
      - XGBoost
      - Feature importance analysis
      - Performance metrics and evaluation
    
    ## Open Source
    
    Zahinn is proudly open source, meaning you can use it freely and even contribute to its development. Built with Python and Streamlit, it leverages the power of modern data science libraries like Pandas, NumPy, and Plotly to provide a seamless experience.
    """)

with col2:
    st.image("https://drive.google.com/file/d/1W2hFLd1SdKiA6OilL63RlGC_XY9HIZkv/view?usp=sharing", use_column_width=True)
    
    st.markdown("""
    ## NidoData
    
    NidoData is committed to creating open-source tools that democratize data analytics and business intelligence, making them accessible to everyone.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Get Started
    
    To begin using Zahinn, simply upload a CSV or Excel file from the main page and explore the various features available.
    """)

# Usage section
st.markdown("---")
st.header("How to Use Zahinn")

st.markdown("""
1. **Upload your data**: Start by uploading a CSV or Excel file on the main page.
2. **Explore your data**: View basic statistics and get a feel for your dataset.
3. **Wrangle your data**: Use the data wrangling tools to clean and transform your data.
4. **Visualize your insights**: Create visualizations to better understand your data.
5. **Export your results**: Save your transformed data for further use.
""")

# Footer
st.markdown("---")
st.caption("Zahinn by NidoData | Open Source Business Intelligence")
