import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="App de Gráficos com XLSX", layout="wide")

st.title("App Streamlit para Análise Gráfica de Banco XLSX")

st.markdown("""
Faça upload de um arquivo **.xlsx**.  
O app possui duas abas:

1. **Histograma filtrado por nível de categoria**
2. **Tabela de contingência e gráfico de barras com duas variáveis categóricas**
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


def tabela_frequencia_dupla(df, var_cat1, var_cat2):
    freq = (
        df.groupby([var_cat1, var_cat2], dropna=False)
        .size()
        .reset_index(name="frequencia")
    )

    freq["percentual"] = 100 * freq["frequencia"] / freq["frequencia"].sum()

    return freq


# ---------------------------------------------------
# Upload
# ---------------------------------------------------
arquivo = st.file_uploader("Envie o arquivo XLSX", type=["xlsx"])

if arquivo is not None:

    try:
        xls = pd.ExcelFile(arquivo)
        abas_planilha = xls.sheet_names

        aba_planilha = st.selectbox(
            "Selecione a aba da planilha",
            abas_planilha,
            index=0
        )

        df = carregar_excel(arquivo, sheet_name=aba_planilha)
        df = preparar_dados(df)

        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head(20), use_container_width=True)

        numericas, categoricas = classificar_variaveis(df)

        if len(numericas) == 0:
            st.warning("Não foram encontradas variáveis numéricas.")

        if len(categoricas) == 0:
            st.warning("Não foram encontradas variáveis categóricas.")

        aba1, aba2 = st.tabs([
            "Histograma",
            "Tabela de contingência"
        ])

        # ===================================================
        # ABA 1 — HISTOGRAMA
        # ===================================================
        with aba1:

            st.header("Histograma por nível de variável categórica")

            if len(numericas) == 0 or len(categoricas) == 0:
                st.error(
                    "Para gerar o histograma, é necessário ter pelo menos "
                    "uma variável numérica e uma variável categórica."
                )

            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    var_cat_hist = st.selectbox(
                        "Escolha a variável categórica",
                        categoricas,
                        key="var_cat_hist"
                    )

                niveis = (
                    df[var_cat_hist]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                with col2:
                    nivel_escolhido = st.selectbox(
                        "Escolha o nível da categoria",
                        niveis,
                        key="nivel_hist"
                    )

                with col3:
                    var_num_hist = st.selectbox(
                        "Escolha a variável contínua",
                        numericas,
                        key="var_num_hist"
                    )

                nbins = st.slider(
                    "Número de classes do histograma",
                    min_value=5,
                    max_value=80,
                    value=20,
                    key="nbins_hist"
                )

                remover_na_hist = st.checkbox(
                    "Remover valores ausentes",
                    value=True,
                    key="remover_na_hist"
                )

                dados_hist = df.copy()
                dados_hist[var_cat_hist] = dados_hist[var_cat_hist].astype(str)

                dados_hist = dados_hist[
                    dados_hist[var_cat_hist] == nivel_escolhido
                ]

                if remover_na_hist:
                    dados_hist = dados_hist.dropna(
                        subset=[var_cat_hist, var_num_hist]
                    )

                if dados_hist.empty:
                    st.error("Não há dados disponíveis para o nível selecionado.")

                else:
                    st.write(f"Categoria selecionada: **{var_cat_hist}**")
                    st.write(f"Nível selecionado: **{nivel_escolhido}**")
                    st.write(f"Variável contínua: **{var_num_hist}**")
                    st.write(f"Número de observações: **{len(dados_hist)}**")

                    fig_hist = px.histogram(
                        dados_hist,
                        x=var_num_hist,
                        nbins=nbins,
                        marginal="box",
                        title=f"Histograma de {var_num_hist} para {var_cat_hist} = {nivel_escolhido}"
                    )

                    fig_hist.update_layout(
                        template="plotly_white",
                        height=600,
                        xaxis_title=var_num_hist,
                        yaxis_title="Frequência"
                    )

                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.subheader("Resumo estatístico")

                    tabela_resumo = resumo_variavel(dados_hist, var_num_hist)

                    st.dataframe(tabela_resumo, use_container_width=True)

                    csv_hist = tabela_resumo.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Baixar resumo do histograma em CSV",
                        data=csv_hist,
                        file_name="resumo_histograma.csv",
                        mime="text/csv"
                    )

        # ===================================================
        # ABA 2 — TABELA DE CONTINGÊNCIA
        # ===================================================
        with aba2:

            st.header("Tabela de contingência entre duas variáveis categóricas")

            if len(categoricas) < 2:
                st.error(
                    "Para gerar a tabela de contingência, são necessárias "
                    "pelo menos duas variáveis categóricas."
                )

            else:
                col1, col2 = st.columns(2)

                with col1:
                    var_cat1 = st.selectbox(
                        "Escolha a primeira variável categórica",
                        categoricas,
                        key="var_cat1_cont"
                    )

                opcoes_cat2 = [c for c in categoricas if c != var_cat1]

                with col2:
                    var_cat2 = st.selectbox(
                        "Escolha a segunda variável categórica",
                        opcoes_cat2,
                        key="var_cat2_cont"
                    )

                remover_na_cont = st.checkbox(
                    "Remover valores ausentes",
                    value=True,
                    key="remover_na_cont"
                )

                dados_cont = df.copy()

                if remover_na_cont:
                    dados_cont = dados_cont.dropna(
                        subset=[var_cat1, var_cat2]
                    )

                if dados_cont.empty:
                    st.error("Não há dados disponíveis para as variáveis selecionadas.")

                else:
                    dados_cont[var_cat1] = dados_cont[var_cat1].astype(str)
                    dados_cont[var_cat2] = dados_cont[var_cat2].astype(str)

                    # Tabela de contingência em formato cruzado
                    tabela_contingencia = pd.crosstab(
                        dados_cont[var_cat1],
                        dados_cont[var_cat2],
                        margins=True,
                        margins_name="Total"
                    )

                    st.subheader("Tabela de contingência")

                    st.dataframe(
                        tabela_contingencia,
                        use_container_width=True
                    )

                    # Tabela em formato longo para o gráfico
                    tabela_grafico = tabela_frequencia_dupla(
                        dados_cont,
                        var_cat1,
                        var_cat2
                    )

                    st.subheader("Tabela em formato longo")

                    st.dataframe(
                        tabela_grafico,
                        use_container_width=True
                    )

                    st.subheader("Gráfico de barras")

                    tipo_barra = st.radio(
                        "Tipo de gráfico de barras",
                        [
                            "Barras agrupadas",
                            "Barras empilhadas"
                        ],
                        horizontal=True
                    )

                    barmode = "group" if tipo_barra == "Barras agrupadas" else "stack"

                    fig_bar = px.bar(
                        tabela_grafico,
                        x=var_cat1,
                        y="frequencia",
                        color=var_cat2,
                        text="frequencia",
                        barmode=barmode,
                        title=f"Frequência conjunta de {var_cat1} e {var_cat2}"
                    )

                    fig_bar.update_layout(
                        template="plotly_white",
                        height=600,
                        xaxis_title=var_cat1,
                        yaxis_title="Frequência"
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

                    csv_cont = tabela_contingencia.to_csv().encode("utf-8")

                    st.download_button(
                        label="Baixar tabela de contingência em CSV",
                        data=csv_cont,
                        file_name="tabela_contingencia.csv",
                        mime="text/csv"
                    )

                    csv_grafico = tabela_grafico.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Baixar tabela do gráfico em CSV",
                        data=csv_grafico,
                        file_name="tabela_frequencias_categoricas.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo: {e}")

else:
    st.info("Envie um arquivo .xlsx para começar.")
