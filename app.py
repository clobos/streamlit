import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            st.dataframe(df[numerical_cols].describe())
        else:
            st.info("No numerical columns found in the dataset.")

        st.write("### Categorical Frequencies")
        categorical_cols = df.select_dtypes(include='object').columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                st.write(f"#### {col}")
                st.dataframe(df[col].value_counts().reset_index(name='Count').rename(columns={'index': 'Value'}))
        else:
            st.info("No categorical columns found in the dataset.")

    with tab3:
        st.header("Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Histogram", "Pie Chart", "Scatter Plot", "Box Plot"]
        )

        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if chart_type == "Bar Chart":
            if categorical_cols:
                x_axis_bar_col = st.selectbox("Select categorical column for X-axis", categorical_cols)
                if x_axis_bar_col:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(data=df, x=x_axis_bar_col)
                    plt.title(f'Bar Chart of {x_axis_bar_col}')
                    plt.xlabel(x_axis_bar_col)
                    plt.ylabel('Count')
                    st.pyplot(plt)
                    plt.clf()
            else:
                st.info("No categorical columns available for a Bar Chart.")

        elif chart_type == "Histogram":
            if numerical_cols:
                hist_col = st.selectbox("Select numerical column for Histogram", numerical_cols)
                if hist_col:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x=hist_col, kde=True)
                    plt.title(f'Histogram of {hist_col}')
                    plt.xlabel(hist_col)
                    plt.ylabel('Frequency')
                    st.pyplot(plt)
                    plt.clf()
            else:
                st.info("No numerical columns available for a Histogram.")

        elif chart_type == "Pie Chart":
            if categorical_cols:
                pie_col = st.selectbox("Select categorical column for Pie Chart", categorical_cols)
                if pie_col:
                    value_counts = df[pie_col].value_counts()
                    labels = value_counts.index
                    sizes = value_counts.values
                    plt.figure(figsize=(8, 8))
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.title(f'Pie Chart of {pie_col}')
                    plt.axis('equal')
                    st.pyplot(plt)
                    plt.clf()
            else:
                st.info("No categorical columns available for a Pie Chart.")

        elif chart_type == "Scatter Plot":
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis_scatter_col = st.selectbox("Select numerical column for X-axis", numerical_cols)
                with col2:
                    y_axis_scatter_col = st.selectbox("Select numerical column for Y-axis", numerical_cols, index=1 if len(numerical_cols)>1 else 0)

                if x_axis_scatter_col and y_axis_scatter_col:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df, x=x_axis_scatter_col, y=y_axis_scatter_col)
                    plt.title(f'Scatter Plot of {x_axis_scatter_col} vs {y_axis_scatter_col}')
                    plt.xlabel(x_axis_scatter_col)
                    plt.ylabel(y_axis_scatter_col)
                    st.pyplot(plt)
                    plt.clf()
            else:
                st.info("Need at least two numerical columns for a Scatter Plot.")

        elif chart_type == "Box Plot":
            if numerical_cols:
                y_axis_box_col = st.selectbox("Select numerical column for Y-axis", numerical_cols)
                if y_axis_box_col:
                    x_axis_options = ["None"] + categorical_cols
                    x_axis_box_col = st.selectbox("Select categorical column for X-axis (optional)", x_axis_options)

                    plt.figure(figsize=(10, 6))
                    if x_axis_box_col != "None":
                        sns.boxplot(data=df, x=x_axis_box_col, y=y_axis_box_col)
                        plt.title(f'Box Plot of {y_axis_box_col} by {x_axis_box_col}')
                        plt.xlabel(x_axis_box_col)
                    else:
                        sns.boxplot(data=df, y=y_axis_box_col)
                        plt.title(f'Box Plot of {y_axis_box_col}')
                    plt.ylabel(y_axis_box_col)
                    st.pyplot(plt)
                    plt.clf()
            else:
                st.info("No numerical columns available for a Box Plot.")

    with tab4:
        st.header("Data Cleaning")
        st.write("Content for Data Cleaning will go here.")
