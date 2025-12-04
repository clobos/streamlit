import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar página uma vez
st.set_page_config(layout="wide")
st.title("Explorador de Dados Excel")

# Inicializar chave de estado da sessão
if "df" not in st.session_state:
    st.session_state.df = None

# Carregador de arquivos na barra lateral
with st.sidebar:
    # Alterado para aceitar .xlsx
    uploaded_file = st.file_uploader("Carregue seu arquivo Excel (XLSX) aqui", type=["xlsx"])
    if uploaded_file is not None:
        try:
            # Alterado para read_excel
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Arquivo Excel carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")

# Se não houver dataframe ainda, mostrar info e parar
if st.session_state.df is None:
    st.info("Por favor, carregue um arquivo Excel (.xlsx) para começar.")
else:
    df = st.session_state.df

    # Área de conteúdo principal com abas
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Visão Geral", "Estatísticas Descritivas", "Visualizações", "Limpeza de Dados"]
    )

    with tab1:
        st.header("Visão Geral dos Dados")
        st.write("### Pré-visualização dos Dados")
        st.dataframe(df.head())

        st.write("### Dimensões do DataFrame")
        st.write(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")

        st.write("### Tipos de Dados das Colunas")
        # Converter tipos para string para evitar problemas de exibição
        types_df = pd.DataFrame({"Coluna": df.columns, "Tipo de Dado": df.dtypes.astype(str).values})
        st.dataframe(types_df)

    with tab2:
        st.header("Estatísticas Descritivas")

        st.write("### Estatísticas Descritivas Numéricas")
        numerical_cols = df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe())
        else:
            st.info("Nenhuma coluna numérica encontrada no conjunto de dados.")

        st.write("### Frequências Categóricas")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                st.write(f"#### {col}")
                counts = df[col].value_counts(dropna=False).reset_index()
                counts.columns = ["Valor", "Contagem"]
                st.dataframe(counts)
        else:
            st.info("Nenhuma coluna categórica encontrada no conjunto de dados.")

    with tab3:
        st.header("Visualizações")

        chart_type = st.selectbox(
            "Selecione o Tipo de Gráfico", ["Gráfico de Barras", "Histograma", "Gráfico de Pizza", "Gráfico de Dispersão", "Box Plot"]
        )

        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if chart_type == "Gráfico de Barras":
            if categorical_cols:
                x_axis = st.selectbox("Selecione a coluna categórica para o eixo X", categorical_cols)
                if x_axis:
                    # --- MODIFICAÇÃO PARA ORDEM ALFABÉTICA ---
                    # 1. Pegamos os valores únicos
                    # 2. Removemos valores nulos (dropna) para não quebrar a ordenação
                    # 3. Usamos sorted() para ordenar alfabeticamente
                    ordem_alfabetica = sorted(df[x_axis].dropna().unique())
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Passamos o parâmetro 'order'
                    sns.countplot(data=df, x=x_axis, ax=ax, order=ordem_alfabetica)
                    
                    ax.set_title(f"Gráfico de Barras de {x_axis} (Ordem Alfabética)")
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel("Contagem")
                    # Rotacionar rótulos se houver muitas categorias
                    plt.xticks(rotation=90)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Nenhuma coluna categórica disponível para Gráfico de Barras.")

        elif chart_type == "Histograma":
            if numerical_cols:
                hist_col = st.selectbox("Selecione a coluna numérica para o Histograma", numerical_cols)
                if hist_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=df, x=hist_col, kde=True, ax=ax)
                    ax.set_title(f"Histograma de {hist_col}")
                    ax.set_xlabel(hist_col)
                    ax.set_ylabel("Frequência")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Nenhuma coluna numérica disponível para Histograma.")

        elif chart_type == "Gráfico de Pizza":
            if categorical_cols:
                pie_col = st.selectbox("Selecione a coluna categórica para o Gráfico de Pizza", categorical_cols)
                if pie_col:
                    value_counts = df[pie_col].value_counts(dropna=False)
                    # Limitar gráfico de pizza às top 10 para evitar poluição visual
                    if len(value_counts) > 10:
                        st.warning("Mostrando apenas as 10 principais categorias.")
                        value_counts = value_counts.head(10)
                    
                    labels = value_counts.index.astype(str)
                    sizes = value_counts.values
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.set_title(f"Gráfico de Pizza de {pie_col}")
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Nenhuma coluna categórica disponível para Gráfico de Pizza.")

        elif chart_type == "Gráfico de Dispersão":
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Selecione a coluna numérica para o eixo X", numerical_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox(
                        "Selecione a coluna numérica para o eixo Y",
                        numerical_cols,
                        index=1 if len(numerical_cols) > 1 else 0,
                        key="scatter_y",
                    )

                if x_col and y_col:
                    if x_col == y_col:
                        st.warning("X e Y são a mesma coluna — o gráfico mostrará a diagonal.")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Gráfico de Dispersão de {x_col} vs {y_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("São necessárias pelo menos duas colunas numéricas para um Gráfico de Dispersão.")

        elif chart_type == "Box Plot":
            if numerical_cols:
                y_col = st.selectbox("Selecione a coluna numérica para o eixo Y", numerical_cols)
                x_options = ["Nenhuma"] + categorical_cols
                x_col = st.selectbox("Selecione a coluna categórica para o eixo X (opcional)", x_options)
                if y_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if x_col != "Nenhuma":
                        # Também ordenamos o Boxplot se tiver eixo X
                        ordem_box = sorted(df[x_col].dropna().unique())
                        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, order=ordem_box)
                        ax.set_title(f"Box Plot de {y_col} por {x_col}")
                        ax.set_xlabel(x_col)
                        plt.xticks(rotation=90)
                    else:
                        sns.boxplot(data=df, y=y_col, ax=ax)
                        ax.set_title(f"Box Plot de {y_col}")
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Nenhuma coluna numérica disponível para Box Plot.")

    with tab4:
        st.header("Limpeza de Dados")
        st.write("Auxiliares básicos de limpeza:")

        # Ferramentas de limpeza de exemplo (não destrutivas a menos que o usuário escolha sobrescrever)
        missing_summary = df.isnull().sum().sort_values(ascending=False)
        st.write("### Valores ausentes por coluna")
        st.dataframe(missing_summary[missing_summary > 0])

        if st.button("Mostrar linhas com valores ausentes"):
            st.dataframe(df[df.isnull().any(axis=1)].head())

        if st.button("Redefinir conjunto de dados carregado (limpar sessão)"):
            st.session_state.df = None
            st.rerun()
