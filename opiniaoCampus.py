import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Histograma por nível de categoria", layout="wide")

st.title("Histograma filtrado por nível de variável categórica")

st.markdown("""
Faça upload de um arquivo **.xlsx**, selecione uma variável categórica,
escolha um nível dessa categoria e gere um histograma com base em uma
variável contínua.
""")

# ---------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------
@st.cache_data
def carregar_excel(arquivo, sheet_name=0):
    return pd.read_excel(arquivo, sheet_name=sheet_name)


def preparar_dados(df):
    df2 = df.copy()

    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = df2[col].replace(r"^\s*$", np.nan, regex=True)

    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = pd.to_numeric(df2[col], errors="ignore")

    return df2


def classificar_variaveis(df):
    numericas = df.select_dtypes(include=np.number).columns.tolist()

    categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datas = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    categoricas = categoricas + [c for c in datas if c not in categoricas]

    return numericas, categoricas


def resumo_variavel(df, var_num):
    return df[var_num].describe().to_frame().T


# ---------------------------------------------------
# Upload do arquivo
# ---------------------------------------------------
arquivo = st.file_uploader("Envie o arquivo XLSX", type=["xlsx"])

if arquivo is not None:

    try:
        xls = pd.ExcelFile(arquivo)
        abas = xls.sheet_names

        aba = st.selectbox("Selecione a aba da planilha", abas, index=0)

        df = carregar_excel(arquivo, sheet_name=aba)
        df = preparar_dados(df)

        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head(20), use_container_width=True)

        numericas, categoricas = classificar_variaveis(df)

        if len(numericas) == 0:
            st.error("Não foram encontradas variáveis numéricas na planilha.")
            st.stop()

        if len(categoricas) == 0:
            st.error("Não foram encontradas variáveis categóricas na planilha.")
            st.stop()

        st.subheader("Configurações do histograma")

        col1, col2, col3 = st.columns(3)

        with col1:
            var_cat = st.selectbox(
                "Escolha a variável categórica",
                categoricas
            )

        niveis = (
            df[var_cat]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        with col2:
            nivel_escolhido = st.selectbox(
                "Escolha o nível da categoria",
                niveis
            )

        with col3:
            var_num = st.selectbox(
                "Escolha a variável contínua",
                numericas
            )

        nbins = st.slider(
            "Número de classes do histograma",
            min_value=5,
            max_value=80,
            value=20
        )

        remover_na = st.checkbox(
            "Remover valores ausentes",
            value=True
        )

        dados = df.copy()

        dados[var_cat] = dados[var_cat].astype(str)

        dados_filtrados = dados[dados[var_cat] == nivel_escolhido]

        if remover_na:
            dados_filtrados = dados_filtrados.dropna(subset=[var_cat, var_num])

        if dados_filtrados.empty:
            st.error("Não há dados disponíveis para o nível selecionado.")
            st.stop()

        st.subheader("Dados filtrados")

        st.write(
            f"Variável categórica selecionada: **{var_cat}**"
        )

        st.write(
            f"Nível selecionado: **{nivel_escolhido}**"
        )

        st.write(
            f"Variável contínua selecionada: **{var_num}**"
        )

        st.write(
            f"Número de observações usadas no histograma: **{len(dados_filtrados)}**"
        )

        st.dataframe(dados_filtrados.head(30), use_container_width=True)

        # ---------------------------------------------------
        # Histograma
        # ---------------------------------------------------
        st.subheader("Histograma")

        fig = px.histogram(
            dados_filtrados,
            x=var_num,
            nbins=nbins,
            title=f"Histograma de {var_num} para {var_cat} = {nivel_escolhido}",
            marginal="box"
        )

        fig.update_layout(
            template="plotly_white",
            height=600,
            xaxis_title=var_num,
            yaxis_title="Frequência"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------
        # Tabela resumo
        # ---------------------------------------------------
        st.subheader("Resumo estatístico da variável contínua")

        tabela_resumo = resumo_variavel(dados_filtrados, var_num)

        st.dataframe(tabela_resumo, use_container_width=True)

        csv = tabela_resumo.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Baixar resumo em CSV",
            data=csv,
            file_name="resumo_histograma_filtrado.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo: {e}")

else:
    st.info("Envie um arquivo .xlsx para começar.")
