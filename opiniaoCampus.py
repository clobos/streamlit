import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Explorador de Dados Excel")

if "df" not in st.session_state:
    st.session_state.df = None

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo Excel (XLSX) aqui",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Arquivo Excel carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")

if st.session_state.df is None:
    st.info("Por favor, carregue um arquivo Excel (.xlsx) para começar.")
else:
    df = st.session_state.df.copy()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Visão Geral", "Estatísticas Descritivas", "Visualizações", "Limpeza de Dados"]
    )

    with tab1:
        st.header("Visão Geral dos Dados")
        st.write("### Pré-visualização dos Dados")
        st.dataframe(df.head(), use_container_width=True)

        st.write("### Dimensões do DataFrame")
        st.write(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")

        st.write("### Tipos de Dados das Colunas")
        types_df = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo de Dado": df.dtypes.astype(str).values
        })
        st.dataframe(types_df, use_container_width=True)

    with tab2:
        st.header("Estatísticas Descritivas")

        st.write("### Estatísticas Descritivas Numéricas")
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numerical_cols:
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        else:
            st.info("Nenhuma coluna numérica encontrada no conjunto de dados.")

        st.write("### Frequências Categóricas")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                st.write(f"#### {col}")
                counts = df[col].value_counts(dropna=False).reset_index()
                counts.columns = ["Valor", "Contagem"]
                st.dataframe(counts, use_container_width=True)
        else:
            st.info("Nenhuma coluna categórica encontrada no conjunto de dados.")

    with tab3:
        st.header("Visualizações")

        chart_type = st.selectbox(
            "Selecione o Tipo de Gráfico",
            [
                "Gráfico de Barras",
                "Histograma",
                "Gráfico de Pizza",
                "Gráfico de Dispersão",
                "Box Plot"
            ]
        )

        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        template = st.selectbox(
            "Tema do gráfico",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
        )

        if chart_type == "Gráfico de Barras":
            if categorical_cols:
                x_axis = st.selectbox(
                    "Selecione a coluna categórica para o eixo X",
                    categorical_cols
                )

                if x_axis:
                    counts = (
                        df[x_axis]
                        .fillna("Ausente")
                        .astype(str)
                        .value_counts()
                        .sort_index()
                        .reset_index()
                    )
                    counts.columns = [x_axis, "Contagem"]

                    fig = px.bar(
                        counts,
                        x=x_axis,
                        y="Contagem",
                        title=f"Gráfico de Barras de {x_axis}",
                        text="Contagem",
                        template=template
                    )
                    fig.update_layout(xaxis_tickangle=-90)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhuma coluna categórica disponível para Gráfico de Barras.")

        elif chart_type == "Histograma":
            if numerical_cols:
                hist_col = st.selectbox(
                    "Selecione a coluna numérica para o Histograma",
                    numerical_cols
                )

                nbins = st.slider("Número de classes (bins)", 5, 100, 30)

                fig = px.histogram(
                    df,
                    x=hist_col,
                    nbins=nbins,
                    marginal="box",
                    title=f"Histograma de {hist_col}",
                    template=template
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhuma coluna numérica disponível para Histograma.")

        elif chart_type == "Gráfico de Pizza":
            if categorical_cols:
                pie_col = st.selectbox(
                    "Selecione a coluna categórica para o Gráfico de Pizza",
                    categorical_cols
                )

                if pie_col:
                    value_counts = (
                        df[pie_col]
                        .fillna("Ausente")
                        .astype(str)
                        .value_counts()
                    )

                    if len(value_counts) > 10:
                        st.warning("Mostrando apenas as 10 principais categorias.")
                        value_counts = value_counts.head(10)

                    pie_df = value_counts.reset_index()
                    pie_df.columns = [pie_col, "Contagem"]

                    fig = px.pie(
                        pie_df,
                        names=pie_col,
                        values="Contagem",
                        title=f"Gráfico de Pizza de {pie_col}",
                        template=template
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhuma coluna categórica disponível para Gráfico de Pizza.")

        elif chart_type == "Gráfico de Dispersão":
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    x_col = st.selectbox(
                        "Selecione a coluna numérica para o eixo X",
                        numerical_cols,
                        key="scatter_x"
                    )

                with col2:
                    y_col = st.selectbox(
                        "Selecione a coluna numérica para o eixo Y",
                        numerical_cols,
                        index=1 if len(numerical_cols) > 1 else 0,
                        key="scatter_y"
                    )

                color_col = st.selectbox(
                    "Colorir por coluna categórica (opcional)",
                    ["Nenhuma"] + categorical_cols
                )

                if x_col and y_col:
                    if x_col == y_col:
                        st.warning("X e Y são a mesma coluna — o gráfico mostrará a diagonal.")

                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "Nenhuma" else color_col,
                        title=f"Gráfico de Dispersão de {x_col} vs {y_col}",
                        template=template,
                        opacity=0.75
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("São necessárias pelo menos duas colunas numéricas para um Gráfico de Dispersão.")

        elif chart_type == "Box Plot":
            if numerical_cols:
                y_col = st.selectbox(
                    "Selecione a coluna numérica para o eixo Y",
                    numerical_cols
                )

                x_options = ["Nenhuma"] + categorical_cols
                x_col = st.selectbox(
                    "Selecione a coluna categórica para o eixo X (opcional)",
                    x_options
                )

                if x_col != "Nenhuma":
                    temp_df = df.copy()
                    temp_df[x_col] = temp_df[x_col].fillna("Ausente").astype(str)

                    ordem_box = sorted(temp_df[x_col].unique())

                    fig = px.box(
                        temp_df,
                        x=x_col,
                        y=y_col,
                        category_orders={x_col: ordem_box},
                        title=f"Box Plot de {y_col} por {x_col}",
                        template=template,
                        points="outliers"
                    )
                    fig.update_layout(xaxis_tickangle=-90)
                else:
                    fig = px.box(
                        df,
                        y=y_col,
                        title=f"Box Plot de {y_col}",
                        template=template,
                        points="outliers"
                    )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhuma coluna numérica disponível para Box Plot.")

    with tab4:
        st.header("Limpeza de Dados")
        st.write("Auxiliares básicos de limpeza:")

        missing_summary = df.isnull().sum().sort_values(ascending=False)
        missing_summary = missing_summary[missing_summary > 0]

        st.write("### Valores ausentes por coluna")
        if not missing_summary.empty:
            st.dataframe(
                missing_summary.reset_index().rename(
                    columns={"index": "Coluna", 0: "Valores Ausentes"}
                ),
                use_container_width=True
            )
        else:
            st.success("Não há valores ausentes no conjunto de dados.")

        if st.button("Mostrar linhas com valores ausentes"):
            st.dataframe(df[df.isnull().any(axis=1)].head(), use_container_width=True)

        if st.button("Redefinir conjunto de dados carregado (limpar sessão)"):
            st.session_state.df = None
            st.rerun()
