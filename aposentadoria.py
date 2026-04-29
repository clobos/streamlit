import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="App de Gráficos com XLSX", layout="wide")

st.title("App Streamlit para Análise Gráfica de Banco XLSX")
st.markdown(
    """
    Faça upload de um arquivo **.xlsx** com variáveis numéricas e categóricas.
    O app permite construir gráficos no Plotly e gerar tabela-resumo por grupos.
    """
)

# ---------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------
@st.cache_data
def carregar_excel(arquivo, sheet_name=0):
    return pd.read_excel(arquivo, sheet_name=sheet_name)

def classificar_variaveis(df):
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # também tratar colunas datetime como categóricas opcionais
    datas = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categoricas = categoricas + [c for c in datas if c not in categoricas]

    return numericas, categoricas

def preparar_dados(df):
    df2 = df.copy()

    # converter strings vazias em NaN
    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = df2[col].replace(r"^\s*$", np.nan, regex=True)

    # tentar converter colunas object para numéricas somente quando fizer sentido
    for col in df2.columns:
        if df2[col].dtype == "object":
            tentativa = pd.to_numeric(df2[col], errors="ignore")
            df2[col] = tentativa

    return df2

def resumo_geral(df, var_num):
    resumo = df[var_num].describe().to_frame().T
    return resumo

def resumo_por_grupo(df, var_num, cat_vars):
    grp = (
        df.groupby(cat_vars, dropna=False)[var_num]
        .agg(
            n="count",
            media="mean",
            desvio_padrao="std",
            mediana="median",
            minimo="min",
            maximo="max"
        )
        .reset_index()
    )
    return grp

def tabela_frequencia(df, cat_vars):
    freq = df.groupby(cat_vars, dropna=False).size().reset_index(name="frequencia")
    freq["percentual"] = 100 * freq["frequencia"] / freq["frequencia"].sum()
    return freq

def limitar_categorias(df, cols, max_niveis=25):
    aviso = []
    for c in cols:
        n = df[c].nunique(dropna=False)
        if n > max_niveis:
            aviso.append(f"{c} ({n} níveis)")
    return aviso

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
            st.warning("Não foram encontradas variáveis categóricas. Alguns gráficos dependem delas.")

        st.subheader("Configurações do gráfico")

        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_grafico = st.selectbox(
                "Tipo de gráfico",
                ["Pontos", "Barras", "Histograma", "Boxplot", "Dispersão", "Pizza"]
            )

        with col2:
            var_num = st.selectbox("Variável numérica", numericas)

        with col3:
            n_cat = st.radio("Número de variáveis categóricas", [1, 2], horizontal=True)

        cat1 = None
        cat2 = None

        if len(categoricas) > 0:
            cat1 = st.selectbox("Variável categórica 1", categoricas, index=0)

            if n_cat == 2:
                opcoes_cat2 = [c for c in categoricas if c != cat1]
                if len(opcoes_cat2) == 0:
                    st.warning("Não há segunda variável categórica diferente disponível.")
                    n_cat = 1
                else:
                    cat2 = st.selectbox("Variável categórica 2", opcoes_cat2, index=0)

        usar_filtro_na = st.checkbox("Remover linhas com NA nas variáveis selecionadas", value=True)

        variaveis_uso = [var_num]
        if cat1 is not None:
            variaveis_uso.append(cat1)
        if cat2 is not None:
            variaveis_uso.append(cat2)

        dados = df.copy()
        if usar_filtro_na:
            dados = dados.dropna(subset=variaveis_uso)

        if dados.empty:
            st.error("Após o filtro de dados faltantes, não restaram observações.")
            st.stop()

        cat_vars = [c for c in [cat1, cat2] if c is not None]

        if len(cat_vars) > 0:
            avisos = limitar_categorias(dados, cat_vars, max_niveis=30)
            if avisos:
                st.warning(
                    "As seguintes variáveis têm muitos níveis e o gráfico pode ficar poluído: "
                    + ", ".join(avisos)
                )

        st.subheader("Gráfico")

        fig = None

        # -----------------------------
        # 1) Pontos
        # -----------------------------
        if tipo_grafico == "Pontos":
            if len(cat_vars) == 0:
                st.warning("Selecione ao menos uma variável categórica.")
            elif len(cat_vars) == 1:
                dados_plot = dados.copy()
                dados_plot["_id"] = np.arange(len(dados_plot))
                fig = px.strip(
                    dados_plot,
                    x=cat_vars[0],
                    y=var_num,
                    color=cat_vars[0],
                    hover_data=dados_plot.columns,
                    stripmode="overlay",
                    title=f"Gráfico de pontos: {var_num} por {cat_vars[0]}"
                )
            else:
                fig = px.strip(
                    dados,
                    x=cat_vars[0],
                    y=var_num,
                    color=cat_vars[1],
                    hover_data=dados.columns,
                    stripmode="overlay",
                    title=f"Gráfico de pontos: {var_num} por {cat_vars[0]} e {cat_vars[1]}"
                )

        # -----------------------------
        # 2) Barras
        # -----------------------------
        elif tipo_grafico == "Barras":
            if len(cat_vars) == 0:
                st.warning("Selecione ao menos uma variável categórica.")
            else:
                resumo = resumo_por_grupo(dados, var_num, cat_vars)
                if len(cat_vars) == 1:
                    fig = px.bar(
                        resumo,
                        x=cat_vars[0],
                        y="media",
                        color=cat_vars[0],
                        text="media",
                        title=f"Média de {var_num} por {cat_vars[0]}"
                    )
                else:
                    fig = px.bar(
                        resumo,
                        x=cat_vars[0],
                        y="media",
                        color=cat_vars[1],
                        barmode="group",
                        text="media",
                        title=f"Média de {var_num} por {cat_vars[0]} e {cat_vars[1]}"
                    )

        # -----------------------------
        # 3) Histograma
        # -----------------------------
        elif tipo_grafico == "Histograma":
            nbins = st.slider("Número de classes", min_value=5, max_value=60, value=20)
            if len(cat_vars) == 0:
                fig = px.histogram(
                    dados,
                    x=var_num,
                    nbins=nbins,
                    title=f"Histograma de {var_num}"
                )
            elif len(cat_vars) == 1:
                fig = px.histogram(
                    dados,
                    x=var_num,
                    color=cat_vars[0],
                    nbins=nbins,
                    barmode="overlay",
                    opacity=0.65,
                    title=f"Histograma de {var_num} por {cat_vars[0]}"
                )
            else:
                fig = px.histogram(
                    dados,
                    x=var_num,
                    color=cat_vars[0],
                    facet_col=cat_vars[1],
                    nbins=nbins,
                    title=f"Histograma de {var_num} por {cat_vars[0]} e {cat_vars[1]}"
                )

        # -----------------------------
        # 4) Boxplot
        # -----------------------------
        elif tipo_grafico == "Boxplot":
            if len(cat_vars) == 0:
                st.warning("Selecione ao menos uma variável categórica.")
            elif len(cat_vars) == 1:
                fig = px.box(
                    dados,
                    x=cat_vars[0],
                    y=var_num,
                    color=cat_vars[0],
                    points="all",
                    title=f"Boxplot de {var_num} por {cat_vars[0]}"
                )
            else:
                fig = px.box(
                    dados,
                    x=cat_vars[0],
                    y=var_num,
                    color=cat_vars[1],
                    points="all",
                    title=f"Boxplot de {var_num} por {cat_vars[0]} e {cat_vars[1]}"
                )

        # -----------------------------
        # 5) Dispersão
        # -----------------------------
        elif tipo_grafico == "Dispersão":
            outras_numericas = [c for c in numericas if c != var_num]
            if len(outras_numericas) == 0:
                st.warning("É necessário pelo menos duas variáveis numéricas para fazer dispersão.")
            else:
                var_num_x = st.selectbox("Variável numérica no eixo X", outras_numericas)
                if len(cat_vars) == 0:
                    fig = px.scatter(
                        dados,
                        x=var_num_x,
                        y=var_num,
                        hover_data=dados.columns,
                        title=f"Dispersão: {var_num} vs {var_num_x}"
                    )
                elif len(cat_vars) == 1:
                    fig = px.scatter(
                        dados,
                        x=var_num_x,
                        y=var_num,
                        color=cat_vars[0],
                        hover_data=dados.columns,
                        title=f"Dispersão: {var_num} vs {var_num_x} por {cat_vars[0]}"
                    )
                else:
                    fig = px.scatter(
                        dados,
                        x=var_num_x,
                        y=var_num,
                        color=cat_vars[0],
                        symbol=cat_vars[1],
                        hover_data=dados.columns,
                        title=f"Dispersão: {var_num} vs {var_num_x} por {cat_vars[0]} e {cat_vars[1]}"
                    )

        # -----------------------------
        # 6) Pizza
        # -----------------------------
        elif tipo_grafico == "Pizza":
            if len(cat_vars) == 0:
                st.warning("Selecione ao menos uma variável categórica.")
            elif len(cat_vars) == 1:
                freq = tabela_frequencia(dados, [cat_vars[0]])
                fig = px.pie(
                    freq,
                    names=cat_vars[0],
                    values="frequencia",
                    title=f"Distribuição de {cat_vars[0]}"
                )
            else:
                st.info("Para pizza com duas categóricas, será usada a combinação das categorias.")
                freq = tabela_frequencia(dados, cat_vars)
                freq["grupo"] = freq[cat_vars[0]].astype(str) + " | " + freq[cat_vars[1]].astype(str)
                fig = px.pie(
                    freq,
                    names="grupo",
                    values="frequencia",
                    title=f"Distribuição conjunta de {cat_vars[0]} e {cat_vars[1]}"
                )

        if fig is not None:
            fig.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------
        # Tabela resumo
        # ---------------------------------------------------
        st.subheader("Tabela resumo")

        if len(cat_vars) == 0:
            tabela = resumo_geral(dados, var_num)
        else:
            tabela = resumo_por_grupo(dados, var_num, cat_vars)

        st.dataframe(tabela, use_container_width=True)

        # ---------------------------------------------------
        # Download da tabela resumo
        # ---------------------------------------------------
        csv = tabela.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar tabela resumo em CSV",
            data=csv,
            file_name="tabela_resumo.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo: {e}")

else:
    st.info("Envie um arquivo .xlsx para começar.")
