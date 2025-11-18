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
