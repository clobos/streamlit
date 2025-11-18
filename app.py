import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page once
st.set_page_config(layout="wide")
st.title("CSV Data Explorer")

# Initialize session state key
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar file uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("CSV file loaded successfully!")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {e}")

# If no dataframe yet, show info and stop
if st.session_state.df is None:
    st.info("Please upload a CSV file to get started.")
else:
    df = st.session_state.df

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Overview", "Descriptive Statistics", "Visualizations", "Data Cleaning"]
    )

    with tab1:
        st.header("Data Overview")
        st.write("### Data Preview")
        st.dataframe(df.head())

        st.write("### DataFrame Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("### Column Data Types")
        types_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values})
        st.dataframe(types_df)

    with tab2:
        st.header("Descriptive Statistics")

        st.write("### Numerical Descriptive Statistics")
        numerical_cols = df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe())
        else:
            st.info("No numerical columns found in the dataset.")

        st.write("### Categorical Frequencies")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                st.write(f"#### {col}")
                counts = df[col].value_counts(dropna=False).reset_index()
                counts.columns = ["Value", "Count"]
                st.dataframe(counts)
        else:
            st.info("No categorical columns found in the dataset.")

    with tab3:
        st.header("Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type", ["Bar Chart", "Histogram", "Pie Chart", "Scatter Plot", "Box Plot"]
        )

        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if chart_type == "Bar Chart":
            if categorical_cols:
                x_axis = st.selectbox("Select categorical column for X-axis", categorical_cols)
                if x_axis:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=df, x=x_axis, ax=ax)
                    ax.set_title(f"Bar Chart of {x_axis}")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No categorical columns available for a Bar Chart.")

        elif chart_type == "Histogram":
            if numerical_cols:
                hist_col = st.selectbox("Select numerical column for Histogram", numerical_cols)
                if hist_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=df, x=hist_col, kde=True, ax=ax)
                    ax.set_title(f"Histogram of {hist_col}")
                    ax.set_xlabel(hist_col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No numerical columns available for a Histogram.")

        elif chart_type == "Pie Chart":
            if categorical_cols:
                pie_col = st.selectbox("Select categorical column for Pie Chart", categorical_cols)
                if pie_col:
                    value_counts = df[pie_col].value_counts(dropna=False)
                    labels = value_counts.index.astype(str)
                    sizes = value_counts.values
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.set_title(f"Pie Chart of {pie_col}")
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No categorical columns available for a Pie Chart.")

        elif chart_type == "Scatter Plot":
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select numerical column for X-axis", numerical_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox(
                        "Select numerical column for Y-axis",
                        numerical_cols,
                        index=1 if len(numerical_cols) > 1 else 0,
                        key="scatter_y",
                    )

                if x_col and y_col:
                    if x_col == y_col:
                        st.warning("X and Y are the same column — the scatter will show the diagonal.")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Need at least two numerical columns for a Scatter Plot.")

        elif chart_type == "Box Plot":
            if numerical_cols:
                y_col = st.selectbox("Select numerical column for Y-axis", numerical_cols)
                x_options = ["None"] + categorical_cols
                x_col = st.selectbox("Select categorical column for X-axis (optional)", x_options)
                if y_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if x_col != "None":
                        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                        ax.set_title(f"Box Plot of {y_col} by {x_col}")
                        ax.set_xlabel(x_col)
                    else:
                        sns.boxplot(data=df, y=y_col, ax=ax)
                        ax.set_title(f"Box Plot of {y_col}")
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No numerical columns available for a Box Plot.")

    with tab4:
        st.header("Data Cleaning")
        st.write("Basic cleaning helpers:")

        # Example cleaning tools (non-destructive unless user chooses to overwrite)
        missing_summary = df.isnull().sum().sort_values(ascending=False)
        st.write("### Missing values by column")
        st.dataframe(missing_summary[missing_summary > 0])

        if st.button("Show rows with any missing values"):
            st.dataframe(df[df.isnull().any(axis=1)].head())

        if st.button("Reset loaded dataset (clear from session)"):
            st.session_state.df = None
            st.experimental_rerun()
