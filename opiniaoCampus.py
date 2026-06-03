import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="App de Gráficos com XLSX", layout="wide")

st.title("App Streamlit para Análise Gráfica de Banco XLSX")

st.markdown("""
Faça upload de um arquivo **.xlsx** com variáveis numéricas e/ou categóricas.
O app permite construir gráficos no Plotly e gerar tabelas-resumo.
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


def resumo_geral(df, var_num):
    return df[var_num].describe().to_frame().T


def resumo_por_grupo(df, var_num, cat_vars):
    return (
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


def tabela_frequencia(df, cat_vars):
    freq = df.groupby(cat_vars, dropna=False).size().reset_index(name="frequencia")
    freq["percentual"] = 100 * freq["frequencia"] / freq["frequencia"].sum()
    return freq


def limitar_categorias(df, cols, max_niveis=30):
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

        if len(categoricas) == 0 and len(numericas) == 0:
            st.error("Não foram encontradas variáveis numéricas nem categóricas.")
            st.stop()

        st.subheader("Configurações do gráfico")

        tipo_grafico = st.selectbox(
            "Tipo de gráfico",
            [
                "Frequência 1 categórica",
                "Frequência 2 categóricas",
                "Pizza",
                "Barras com média",
                "Pontos",
                "Histograma",
                "Boxplot",
                "Dispersão"
            ]
        )

        graficos_sem_numerica = [
            "Frequência 1 categórica",
            "Frequência 2 categóricas",
            "Pizza"
        ]

        usa_numerica = tipo_grafico not in graficos_sem_numerica

        var_num = None

        if usa_numerica:
            if len(numericas) == 0:
                st.error("Este tipo de gráfico precisa de pelo menos uma variável numérica.")
                st.stop()

            var_num = st.selectbox("Variável numérica", numericas)

        if len(categoricas) > 0:
            n_cat = st.radio(
                "Número de variáveis categóricas",
                [1, 2],
                horizontal=True
            )

            cat1 = st.selectbox("Variável categórica 1", categoricas, index=0)
            cat2 = None

            if n_cat == 2:
                opcoes_cat2 = [c for c in categoricas if c != cat1]

                if len(opcoes_cat2) == 0:
                    st.warning("Não há segunda variável categórica diferente disponível.")
                    n_cat = 1
                else:
                    cat2 = st.selectbox("Variável categórica 2", opcoes_cat2, index=0)
        else:
            n_cat = 0
            cat1 = None
            cat2 = None

        usar_filtro_na = st.checkbox(
            "Remover linhas com NA nas variáveis selecionadas",
            value=True
        )

        variaveis_uso = []

        if var_num is not None:
            variaveis_uso.append(var_num)

        if cat1 is not None:
            variaveis_uso.append(cat1)

        if cat2 is not None:
            variaveis_uso.append(cat2)

        dados = df.copy()

        if usar_filtro_na and len(variaveis_uso) > 0:
            dados = dados.dropna(subset=variaveis_uso)

        if dados.empty:
            st.error("Após o filtro de dados faltantes, não restaram observações.")
            st.stop()

        cat_vars = [c for c in [cat1, cat2] if c is not None]

        if len(cat_vars) > 0:
            avisos = limitar_categorias(dados, cat_vars)

            if avisos:
                st.warning(
                    "As seguintes variáveis têm muitos níveis e o gráfico pode ficar poluído: "
                    + ", ".join(avisos)
                )

        st.subheader("Gráfico")

        fig = None
        tabela = None

        # ---------------------------------------------------
        # 1) Frequência com uma variável categórica
        # ---------------------------------------------------
        if tipo_grafico == "Frequência 1 categórica":

            if cat1 is None:
                st.warning("Selecione uma variável categórica.")
            else:
                freq = tabela_frequencia(dados, [cat1])

                fig = px.bar(
                    freq,
                    x=cat1,
                    y="frequencia",
                    color=cat1,
                    text="frequencia",
                    title=f"Frequência de {cat1}"
                )

                tabela = freq

        # ---------------------------------------------------
        # 2) Frequência com duas variáveis categóricas
        # ---------------------------------------------------
        elif tipo_grafico == "Frequência 2 categóricas":

            if cat1 is None or cat2 is None:
                st.warning("Selecione duas variáveis categóricas.")
            else:
                freq = tabela_frequencia(dados, [cat1, cat2])

                fig = px.bar(
                    freq,
                    x=cat1,
                    y="frequencia",
                    color=cat2,
                    barmode="group",
                    text="frequencia",
                    title=f"Frequência conjunta de {cat1} e {cat2}"
                )

                tabela = freq

        # ---------------------------------------------------
        # 3) Pizza
        # ---------------------------------------------------
        elif tipo_grafico == "Pizza":

            if cat1 is None:
                st.warning("Selecione uma variável categórica.")
            elif cat2 is None:
                freq = tabela_frequencia(dados, [cat1])

                fig = px.pie(
                    freq,
                    names=cat1,
                    values="frequencia",
                    title=f"Distribuição de {cat1}"
                )

                tabela = freq

            else:
                freq = tabela_frequencia(dados, [cat1, cat2])
                freq["grupo"] = (
                    freq[cat1].astype(str) + " | " + freq[cat2].astype(str)
                )

                fig = px.pie(
                    freq,
                    names="grupo",
                    values="frequencia",
                    title=f"Distribuição conjunta de {cat1} e {cat2}"
                )

                tabela = freq

        # ---------------------------------------------------
        # 4) Barras com média
        # ---------------------------------------------------
        elif tipo_grafico == "Barras com média":

            if len(cat_vars) == 0:
                st.warning("Selecione pelo menos uma variável categórica.")
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

                tabela = resumo

        # ---------------------------------------------------
        # 5) Pontos
        # ---------------------------------------------------
        elif tipo_grafico == "Pontos":

            if len(cat_vars) == 0:
                st.warning("Selecione pelo menos uma variável categórica.")
            elif len(cat_vars) == 1:
                fig = px.strip(
                    dados,
                    x=cat_vars[0],
                    y=var_num,
                    color=cat_vars[0],
                    hover_data=dados.columns,
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

            tabela = resumo_por_grupo(dados, var_num, cat_vars)

        # ---------------------------------------------------
        # 6) Histograma
        # ---------------------------------------------------
        elif tipo_grafico == "Histograma":

            nbins = st.slider(
                "Número de classes",
                min_value=5,
                max_value=60,
                value=20
            )

            if len(cat_vars) == 0:
                fig = px.histogram(
                    dados,
                    x=var_num,
                    nbins=nbins,
                    title=f"Histograma de {var_num}"
                )

                tabela = resumo_geral(dados, var_num)

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

                tabela = resumo_por_grupo(dados, var_num, cat_vars)

            else:
                fig = px.histogram(
                    dados,
                    x=var_num,
                    color=cat_vars[0],
                    facet_col=cat_vars[1],
                    nbins=nbins,
                    title=f"Histograma de {var_num} por {cat_vars[0]} e {cat_vars[1]}"
                )

                tabela = resumo_por_grupo(dados, var_num, cat_vars)

        # ---------------------------------------------------
        # 7) Boxplot
        # ---------------------------------------------------
        elif tipo_grafico == "Boxplot":

            if len(cat_vars) == 0:
                st.warning("Selecione pelo menos uma variável categórica.")
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

            tabela = resumo_por_grupo(dados, var_num, cat_vars)

        # ---------------------------------------------------
        # 8) Dispersão
        # ---------------------------------------------------
        elif tipo_grafico == "Dispersão":

            outras_numericas = [c for c in numericas if c != var_num]

            if len(outras_numericas) == 0:
                st.warning("É necessário ter pelo menos duas variáveis numéricas.")
            else:
                var_num_x = st.selectbox(
                    "Variável numérica no eixo X",
                    outras_numericas
                )

                if len(cat_vars) == 0:
                    fig = px.scatter(
                        dados,
                        x=var_num_x,
                        y=var_num,
                        hover_data=dados.columns,
                        title=f"Dispersão: {var_num} vs {var_num_x}"
                    )

                    tabela = resumo_geral(dados, var_num)

                elif len(cat_vars) == 1:
                    fig = px.scatter(
                        dados,
                        x=var_num_x,
                        y=var_num,
                        color=cat_vars[0],
                        hover_data=dados.columns,
                        title=f"Dispersão: {var_num} vs {var_num_x} por {cat_vars[0]}"
                    )

                    tabela = resumo_por_grupo(dados, var_num, cat_vars)

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

                    tabela = resumo_por_grupo(dados, var_num, cat_vars)

        # ---------------------------------------------------
        # Mostrar gráfico
        # ---------------------------------------------------
        if fig is not None:
            fig.update_layout(
                template="plotly_white",
                height=600
            )

            fig.update_traces(
                texttemplate="%{text:.2s}",
                textposition="outside"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------
        # Mostrar tabela
        # ---------------------------------------------------
        st.subheader("Tabela resumo")

        if tabela is not None:
            st.dataframe(tabela, use_container_width=True)

            csv = tabela.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Baixar tabela resumo em CSV",
                data=csv,
                file_name="tabela_resumo.csv",
                mime="text/csv"
            )
        else:
            st.info("Nenhuma tabela foi gerada para esta configuração.")

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo: {e}")

else:
    st.info("Envie um arquivo .xlsx para começar.")
