import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title('CSV Data Explorer')

uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.success("CSV file loaded successfully!")
    st.write("Data Preview:")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV file to get started.")


st.set_page_config(layout="wide")
st.title('CSV Data Explorer')

# Sidebar for file uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.success("CSV file loaded successfully!")
else:
    st.info("Please upload a CSV file to get started.")

# Main content area with tabs
if 'df' in st.session_state:
    df = st.session_state['df']

    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Descriptive Statistics", "Visualizations", "Data Cleaning"])

    with tab1:
        st.header("Data Overview")
        st.write("### Data Header")
        st.dataframe(df.head())

        st.write("### DataFrame Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("### Column Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

    with tab2:
        st.header("Descriptive Statistics")

        st.write("### Numerical Descriptive Statistics")
        # Select numerical columns and display their descriptive statistics
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            st.dataframe(df[numerical_cols].describe())
        else:
            st.info("No numerical columns found in the dataset.")

        st.write("### Categorical Frequencies")
        # Select categorical columns and display their value counts
        categorical_cols = df.select_dtypes(include='object').columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                st.write(f"#### {col}")
                st.dataframe(df[col].value_counts().reset_index(name='Count').rename(columns={'index': 'Value'}))
        else:
            st.info("No categorical columns found in the dataset.")

    with tab3:
        st.header("Visualizations")
        st.write("Content for Visualizations will go here.")

    with tab4:
        st.header("Data Cleaning")
        st.write("Content for Data Cleaning will go here.")
