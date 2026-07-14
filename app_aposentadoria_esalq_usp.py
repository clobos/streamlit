#---------------------------------------------------
#Para um projeto novo no Windows, eu recomendaria:
#---------------------------------------------------
#py -m venv .venv
#.venv\Scripts\activate.bat
#python -m pip install --upgrade pip
#python -m pip install streamlit pandas numpy openpyxl plotly
#python -m streamlit run app_aposentadoria_esalq_usp.py
#---------------------------------------------------
#Para as próximas vezes, normalmente basta:
#---------------------------------------------------
#.venv\Scripts\activate.bat
#python -m streamlit run app_aposentadoria_esalq_usp.py
#py -m streamlit run app_aposentadoria_esalq_usp.py
#---------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="App de Gráficos com XLSX", layout="wide")

st.title("Aposentadoria ESALQ USP")

st.markdown("""
Faça upload de um arquivo **.xlsx**.  
O app possui seis abas:

1. **Histograma filtrado por nível de categoria**
2. **Tabela de contingência e gráfico de barras com duas variáveis categóricas**
3. **Análise descritiva das variáveis numéricas e categóricas**
4. **Frequências absolutas e relativas de uma variável categórica**
5. **Histogramas comparativos para todos os níveis de uma variável categórica**
6. **Gráfico de barras e tabela de contingência com três variáveis categóricas**
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


def tabela_frequencia_simples(df, variavel, incluir_ausentes=True):
    """Calcula frequências absoluta e relativa de uma variável categórica."""
    serie = df[variavel].copy()

    if incluir_ausentes:
        serie = serie.astype("object").where(serie.notna(), "Ausente")
    else:
        serie = serie.dropna()

    serie = serie.astype(str)

    tabela = (
        serie.value_counts(dropna=False)
        .rename_axis("categoria")
        .reset_index(name="frequencia_absoluta")
    )

    total = tabela["frequencia_absoluta"].sum()
    tabela["frequencia_relativa_%"] = (
        100 * tabela["frequencia_absoluta"] / total
        if total > 0
        else 0.0
    )

    return tabela


def padronizar_fontes_plotly(fig):
    """Padroniza as fontes de todas as figuras Plotly do aplicativo.

    - Títulos das figuras: 18
    - Títulos dos eixos x e y: 20
    - Valores/níveis apresentados nos eixos: 20
    - Itens e títulos das legendas: 20
    - Títulos dos painéis (facetas): 20
    """
    fig.update_layout(
        font=dict(size=20),
        title_font=dict(size=18),
        legend=dict(
            font=dict(size=20),
            title_font=dict(size=20)
        )
    )

    fig.update_xaxes(
        title_font=dict(size=20),
        tickfont=dict(size=20)
    )

    fig.update_yaxes(
        title_font=dict(size=20),
        tickfont=dict(size=20)
    )

    # Nos gráficos com painéis, as anotações representam os níveis do fator.
    fig.update_annotations(font=dict(size=20))

    return fig


def analise_descritiva_numericas(df, numericas):
    resumo = df[numericas].describe().T
    resumo["mediana"] = df[numericas].median()
    resumo["variancia"] = df[numericas].var()
    resumo["desvio_padrao"] = df[numericas].std()
    resumo["coef_variacao_%"] = (resumo["desvio_padrao"] / resumo["mean"]) * 100
    resumo["valores_ausentes"] = df[numericas].isna().sum()
    resumo["percentual_ausentes_%"] = df[numericas].isna().mean() * 100
    resumo = resumo.reset_index().rename(columns={"index": "variavel"})
    return resumo


def analise_descritiva_categoricas(df, categoricas):
    lista = []

    for col in categoricas:
        serie = df[col]

        total = len(serie)
        ausentes = serie.isna().sum()
        validos = total - ausentes
        categorias_unicas = serie.nunique(dropna=True)

        moda = serie.mode(dropna=True)
        moda_valor = moda.iloc[0] if len(moda) > 0 else np.nan

        freq_moda = serie.value_counts(dropna=True).iloc[0] if validos > 0 else 0
        perc_moda = (freq_moda / validos) * 100 if validos > 0 else 0

        lista.append({
            "variavel": col,
            "total_observacoes": total,
            "valores_validos": validos,
            "valores_ausentes": ausentes,
            "percentual_ausentes_%": (ausentes / total) * 100 if total > 0 else 0,
            "categorias_unicas": categorias_unicas,
            "categoria_mais_frequente": moda_valor,
            "frequencia_categoria_mais_frequente": freq_moda,
            "percentual_categoria_mais_frequente_%": perc_moda
        })

    return pd.DataFrame(lista)


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

        aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
            "Histograma",
            "Tabela de contingência",
            "Análise Descritiva",
            "Frequências categóricas",
            "Histogramas por nível",
            "Contingência com 3 fatores"
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
                        title=f"Histograma de {var_num_hist} para {var_cat_hist} = {nivel_escolhido}"
                    )

                    fig_hist.update_traces(
                        marker_color="red",
                        marker_line_color="black",
                        marker_line_width=1
                    )

                    fig_hist.update_layout(
                        template="plotly_white",
                        height=600,
                        xaxis_title=var_num_hist,
                        yaxis_title="Frequência",
                        xaxis=dict(
                            title_font=dict(size=22),
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict(
                            title_font=dict(size=22),
                            tickfont=dict(size=16)
                        ),
                        title_font=dict(size=22)
                    )

                    fig_hist = padronizar_fontes_plotly(fig_hist)

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

                    tabela_grafico = tabela_frequencia_dupla(
                        dados_cont,
                        var_cat1,
                        var_cat2
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

                    fig_bar.update_traces(
                        texttemplate="%{text:.0f}",
                        textposition="outside",
                        textfont=dict(size=20),#cambio 14/07/2026 (original era size=14)
                        cliponaxis=False
                    )
                    
                    fig_bar.update_layout(
                        template="plotly_white",
                        height=600,
                        xaxis_title=var_cat1,
                        yaxis_title="Frequência",
                        xaxis=dict(
                            title_font=dict(size=22),
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict(
                            title_font=dict(size=22),
                            tickfont=dict(size=16)
                        ),
                        title_font=dict(size=22),
                        legend=dict(
                            font=dict(size=16),
                            title_font=dict(size=16)
                        )
                    )

                    fig_bar = padronizar_fontes_plotly(fig_bar)

                    st.plotly_chart(fig_bar, use_container_width=True)

                    csv_cont = tabela_contingencia.to_csv().encode("utf-8")

                    st.download_button(
                        label="Baixar tabela de contingência em CSV",
                        data=csv_cont,
                        file_name="tabela_contingencia.csv",
                        mime="text/csv"
                    )

        # ===================================================
        # ABA 3 — ANÁLISE DESCRITIVA
        # ===================================================
        with aba3:

            st.header("Análise descritiva das variáveis")

            st.subheader("Resumo geral do banco de dados")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Número de linhas", df.shape[0])
            col2.metric("Número de colunas", df.shape[1])
            col3.metric("Variáveis numéricas", len(numericas))
            col4.metric("Variáveis categóricas", len(categoricas))

            st.divider()

            st.subheader("Análise descritiva das variáveis numéricas")

            if len(numericas) == 0:
                st.warning("Não há variáveis numéricas para descrever.")
            else:
                resumo_num = analise_descritiva_numericas(df, numericas)

                st.dataframe(
                    resumo_num,
                    use_container_width=True
                )

                csv_num = resumo_num.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Baixar análise descritiva das variáveis numéricas em CSV",
                    data=csv_num,
                    file_name="analise_descritiva_numericas.csv",
                    mime="text/csv"
                )

            st.divider()

            st.subheader("Análise descritiva das variáveis categóricas")

            if len(categoricas) == 0:
                st.warning("Não há variáveis categóricas para descrever.")
            else:
                resumo_cat = analise_descritiva_categoricas(df, categoricas)

                st.dataframe(
                    resumo_cat,
                    use_container_width=True
                )

                csv_cat = resumo_cat.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Baixar análise descritiva das variáveis categóricas em CSV",
                    data=csv_cat,
                    file_name="analise_descritiva_categoricas.csv",
                    mime="text/csv"
                )

                st.subheader("Tabela de frequência por variável categórica")

                var_cat_desc = st.selectbox(
                    "Escolha uma variável categórica para ver a frequência",
                    categoricas,
                    key="var_cat_desc"
                )

                tabela_freq = (
                    df[var_cat_desc]
                    .astype("object")
                    .fillna("Ausente")
                    .value_counts()
                    .reset_index()
                )

                tabela_freq.columns = [var_cat_desc, "frequencia"]
                tabela_freq["percentual_%"] = (
                    tabela_freq["frequencia"] / tabela_freq["frequencia"].sum()
                ) * 100

                st.dataframe(
                    tabela_freq,
                    use_container_width=True
                )

                fig_freq = px.bar(
                    tabela_freq,
                    x=var_cat_desc,
                    y="frequencia",
                    text="frequencia",
                    title=f"Frequência das categorias de {var_cat_desc}"
                )

                fig_freq.update_layout(
                    template="plotly_white",
                    height=600,
                    xaxis_title=var_cat_desc,
                    yaxis_title="Frequência",
                    xaxis=dict(
                        title_font=dict(size=22),
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        title_font=dict(size=22),
                        tickfont=dict(size=16)
                    ),
                    title_font=dict(size=22)
                )

                fig_freq = padronizar_fontes_plotly(fig_freq)

                st.plotly_chart(fig_freq, use_container_width=True)


        # ===================================================
        # ABA 4 — FREQUÊNCIAS ABSOLUTAS E RELATIVAS
        # ===================================================
        with aba4:

            st.header("Frequências de uma variável categórica")

            if len(categoricas) == 0:
                st.warning(
                    "Não há variáveis categóricas disponíveis para esta análise."
                )

            else:
                col_selecao, col_ordenacao = st.columns(2)

                with col_selecao:
                    var_cat_freq = st.selectbox(
                        "Escolha a variável categórica",
                        categoricas,
                        key="var_cat_freq_aba4"
                    )

                with col_ordenacao:
                    ordem_freq = st.selectbox(
                        "Ordenar as categorias por",
                        [
                            "Frequência decrescente",
                            "Frequência crescente",
                            "Ordem alfabética"
                        ],
                        key="ordem_freq_aba4"
                    )

                incluir_ausentes_freq = st.checkbox(
                    "Incluir valores ausentes como categoria 'Ausente'",
                    value=True,
                    key="incluir_ausentes_freq_aba4"
                )

                tabela_freq_simples = tabela_frequencia_simples(
                    df=df,
                    variavel=var_cat_freq,
                    incluir_ausentes=incluir_ausentes_freq
                )

                if ordem_freq == "Frequência crescente":
                    tabela_freq_simples = tabela_freq_simples.sort_values(
                        "frequencia_absoluta",
                        ascending=True
                    )
                elif ordem_freq == "Ordem alfabética":
                    tabela_freq_simples = tabela_freq_simples.sort_values(
                        "categoria",
                        ascending=True,
                        key=lambda s: s.str.lower()
                    )
                else:
                    tabela_freq_simples = tabela_freq_simples.sort_values(
                        "frequencia_absoluta",
                        ascending=False
                    )

                tabela_freq_simples = tabela_freq_simples.reset_index(drop=True)

                if tabela_freq_simples.empty:
                    st.error(
                        "Não há observações disponíveis para a variável selecionada."
                    )

                else:
                    total_observacoes_freq = int(
                        tabela_freq_simples["frequencia_absoluta"].sum()
                    )

                    st.write(
                        f"Total de observações consideradas: "
                        f"**{total_observacoes_freq}**"
                    )

                    ordem_categorias = tabela_freq_simples["categoria"].tolist()

                    # ---------------------------------------------------
                    # Gráfico horizontal de frequências absolutas
                    # ---------------------------------------------------
                    fig_abs = px.bar(
                        tabela_freq_simples,
                        x="frequencia_absoluta",
                        y="categoria",
                        text="frequencia_absoluta",
                        orientation="h",
                        category_orders={"categoria": ordem_categorias},
                        labels={
                            "categoria": var_cat_freq,
                            "frequencia_absoluta": "Frequência absoluta"
                        },
                        title="Frequências absolutas"
                    )

                    fig_abs.update_traces(
                        texttemplate="%{text:.0f}",
                        textposition="outside",
                        textfont=dict(size=20),
                        cliponaxis=False,
                        marker_line_color="black",
                        marker_line_width=1,
                        hovertemplate=(
                            f"{var_cat_freq}: %{{y}}<br>"
                            "Frequência absoluta: %{x:.0f}"
                            "<extra></extra>"
                        )
                    )

                    fig_abs.update_layout(
                        template="plotly_white",
                        height=max(550, 45 * len(ordem_categorias) + 180),
                        margin=dict(t=80, r=90, b=70, l=120),
                        uniformtext_minsize=12,
                        uniformtext_mode="show",
                        font=dict(size=14),
                        title_font=dict(size=14),
                        xaxis=dict(
                            title="Frequência absoluta",
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            rangemode="tozero",
                            automargin=True
                        ),
                        yaxis=dict(
                            title=var_cat_freq,
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            categoryorder="array",
                            categoryarray=ordem_categorias,
                            autorange="reversed",
                            automargin=True
                        ),
                        hoverlabel=dict(font_size=14)
                    )

                    fig_abs.update_xaxes(
                        range=[
                            0,
                            max(
                                tabela_freq_simples[
                                    "frequencia_absoluta"
                                ].max() * 1.20,
                                1
                            )
                        ]
                    )

                    # ---------------------------------------------------
                    # Gráfico horizontal de frequências relativas
                    # ---------------------------------------------------
                    fig_rel = px.bar(
                        tabela_freq_simples,
                        x="frequencia_relativa_%",
                        y="categoria",
                        text="frequencia_relativa_%",
                        orientation="h",
                        category_orders={"categoria": ordem_categorias},
                        labels={
                            "categoria": var_cat_freq,
                            "frequencia_relativa_%": "Frequência relativa (%)"
                        },
                        title="Frequências relativas"
                    )

                    fig_rel.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside",
                        textfont=dict(size=20),
                        cliponaxis=False,
                        marker_line_color="black",
                        marker_line_width=1,
                        hovertemplate=(
                            f"{var_cat_freq}: %{{y}}<br>"
                            "Frequência relativa: %{x:.2f}%"
                            "<extra></extra>"
                        )
                    )

                    fig_rel.update_layout(
                        template="plotly_white",
                        height=max(550, 45 * len(ordem_categorias) + 180),
                        margin=dict(t=80, r=90, b=70, l=120),
                        uniformtext_minsize=12,
                        uniformtext_mode="show",
                        font=dict(size=14),
                        title_font=dict(size=14),
                        xaxis=dict(
                            title="Frequência relativa (%)",
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            ticksuffix="%",
                            rangemode="tozero",
                            automargin=True
                        ),
                        yaxis=dict(
                            title=var_cat_freq,
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            categoryorder="array",
                            categoryarray=ordem_categorias,
                            autorange="reversed",
                            automargin=True
                        ),
                        hoverlabel=dict(font_size=14)
                    )

                    fig_rel.update_xaxes(
                        range=[
                            0,
                            max(
                                tabela_freq_simples[
                                    "frequencia_relativa_%"
                                ].max() * 1.20,
                                1
                            )
                        ]
                    )

                    fig_abs = padronizar_fontes_plotly(fig_abs)
                    fig_rel = padronizar_fontes_plotly(fig_rel)

                    col_graf_abs, col_graf_rel = st.columns(2)

                    with col_graf_abs:
                        st.plotly_chart(
                            fig_abs,
                            use_container_width=True,
                            key="grafico_freq_absoluta_aba4"
                        )

                    with col_graf_rel:
                        st.plotly_chart(
                            fig_rel,
                            use_container_width=True,
                            key="grafico_freq_relativa_aba4"
                        )

                    st.subheader("Tabela de frequências")

                    tabela_exibicao = tabela_freq_simples.rename(
                        columns={
                            "categoria": var_cat_freq,
                            "frequencia_absoluta": "Frequência absoluta",
                            "frequencia_relativa_%": "Frequência relativa (%)"
                        }
                    )

                    st.dataframe(
                        tabela_exibicao.style.format(
                            {"Frequência relativa (%)": "{:.2f}%"}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_freq = tabela_exibicao.to_csv(
                        index=False,
                        decimal=",",
                        sep=";"
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="Baixar tabela de frequências em CSV",
                        data=csv_freq,
                        file_name=f"frequencias_{var_cat_freq}.csv",
                        mime="text/csv",
                        key="download_freq_aba4"
                    )



        # ===================================================
        # ABA 5 — HISTOGRAMAS PARA TODOS OS NÍVEIS DO FATOR
        # ===================================================
        with aba5:

            st.header("Histogramas da variável numérica por nível do fator")

            st.markdown(
                "Selecione uma variável numérica e uma variável categórica. "
                "Cada nível da variável categórica será apresentado em um "
                "painel separado, utilizando as mesmas classes e a mesma "
                "escala no eixo horizontal."
            )

            if len(numericas) == 0 or len(categoricas) == 0:
                st.error(
                    "Esta análise exige pelo menos uma variável numérica e "
                    "uma variável categórica."
                )

            else:
                col_var_num, col_var_cat = st.columns(2)

                with col_var_num:
                    var_num_multihist = st.selectbox(
                        "Escolha a variável numérica",
                        numericas,
                        key="var_num_multihist_aba5"
                    )

                with col_var_cat:
                    var_cat_multihist = st.selectbox(
                        "Escolha a variável categórica",
                        categoricas,
                        key="var_cat_multihist_aba5"
                    )

                col_bins, col_colunas, col_ordem = st.columns(3)

                with col_bins:
                    nbins_multihist = st.slider(
                        "Número de classes",
                        min_value=5,
                        max_value=20,
                        value=10,
                        key="nbins_multihist_aba5"
                    )

                with col_colunas:
                    numero_colunas_multihist = st.slider(
                        "Painéis por linha",
                        min_value=1,
                        max_value=4,
                        value=3,
                        key="numero_colunas_multihist_aba5"
                    )

                with col_ordem:
                    ordem_niveis_multihist = st.selectbox(
                        "Ordenar os níveis por",
                        [
                            "Ordem alfabética",
                            "Frequência decrescente",
                            "Frequência crescente"
                        ],
                        key="ordem_niveis_multihist_aba5"
                    )

                incluir_ausentes_multihist = st.checkbox(
                    "Incluir valores ausentes da variável categórica como "
                    "nível 'Ausente'",
                    value=False,
                    key="incluir_ausentes_multihist_aba5"
                )

                dados_multihist = df[
                    [var_num_multihist, var_cat_multihist]
                ].copy()

                # Valores ausentes na variável numérica não podem compor o
                # histograma e, portanto, são sempre removidos.
                dados_multihist = dados_multihist.dropna(
                    subset=[var_num_multihist]
                )

                if incluir_ausentes_multihist:
                    dados_multihist[var_cat_multihist] = (
                        dados_multihist[var_cat_multihist]
                        .astype("object")
                        .where(
                            dados_multihist[var_cat_multihist].notna(),
                            "Ausente"
                        )
                    )
                else:
                    dados_multihist = dados_multihist.dropna(
                        subset=[var_cat_multihist]
                    )

                dados_multihist[var_cat_multihist] = (
                    dados_multihist[var_cat_multihist].astype(str)
                )

                contagem_niveis = (
                    dados_multihist[var_cat_multihist]
                    .value_counts()
                )

                if ordem_niveis_multihist == "Frequência decrescente":
                    niveis_ordenados = contagem_niveis.index.tolist()
                elif ordem_niveis_multihist == "Frequência crescente":
                    niveis_ordenados = (
                        contagem_niveis
                        .sort_values(ascending=True)
                        .index
                        .tolist()
                    )
                else:
                    niveis_ordenados = sorted(
                        contagem_niveis.index.tolist(),
                        key=lambda valor: valor.lower()
                    )

                niveis_selecionados = st.multiselect(
                    "Níveis que serão exibidos",
                    options=niveis_ordenados,
                    default=niveis_ordenados,
                    key="niveis_multihist_aba5"
                )

                dados_multihist = dados_multihist[
                    dados_multihist[var_cat_multihist].isin(
                        niveis_selecionados
                    )
                ].copy()

                if dados_multihist.empty or len(niveis_selecionados) == 0:
                    st.warning(
                        "Não há observações disponíveis para a combinação "
                        "selecionada. Escolha ao menos um nível do fator."
                    )

                else:
                    # Mantém a ordem escolhida pelo usuário nos painéis.
                    niveis_exibidos = [
                        nivel for nivel in niveis_ordenados
                        if nivel in niveis_selecionados
                    ]

                    dados_multihist[var_cat_multihist] = pd.Categorical(
                        dados_multihist[var_cat_multihist],
                        categories=niveis_exibidos,
                        ordered=True
                    )

                    minimo_global = float(
                        dados_multihist[var_num_multihist].min()
                    )
                    maximo_global = float(
                        dados_multihist[var_num_multihist].max()
                    )

                    if minimo_global == maximo_global:
                        amplitude_auxiliar = max(
                            abs(minimo_global) * 0.10,
                            1.0
                        )
                        inicio_classes = minimo_global - amplitude_auxiliar / 2
                        fim_classes = maximo_global + amplitude_auxiliar / 2
                        largura_classe = amplitude_auxiliar
                    else:
                        inicio_classes = minimo_global
                        fim_classes = maximo_global
                        largura_classe = (
                            maximo_global - minimo_global
                        ) / nbins_multihist

                    numero_niveis = len(niveis_exibidos)
                    numero_linhas = int(
                        np.ceil(numero_niveis / numero_colunas_multihist)
                    )
                    altura_figura = max(500, 330 * numero_linhas)

                    fig_multihist = px.histogram(
                        dados_multihist,
                        x=var_num_multihist,
                        color=var_cat_multihist,
                        facet_col=var_cat_multihist,
                        facet_col_wrap=numero_colunas_multihist,
                        category_orders={
                            var_cat_multihist: niveis_exibidos
                        },
                        labels={
                            var_num_multihist: var_num_multihist,
                            var_cat_multihist: var_cat_multihist
                        },
                        title=(
                            f"Distribuição de {var_num_multihist} por níveis "
                            f"de {var_cat_multihist}"
                        )
                    )

                    # As mesmas classes são aplicadas a todos os painéis,
                    # permitindo comparar diretamente as distribuições.
                    fig_multihist.update_traces(
                        xbins=dict(
                            start=inicio_classes,
                            end=fim_classes,
                            size=largura_classe
                        ),
                        marker_line_color="black",
                        marker_line_width=0.8,
                        opacity=0.85,
                        hovertemplate=(
                            "Centro da classe: %{x}<br>"
                            "Frequência: %{y}<extra></extra>"
                        )
                    )

                    # Remove o nome da variável antes do título de cada painel,
                    # deixando somente o nível correspondente.
                    fig_multihist.for_each_annotation(
                        lambda anotacao: anotacao.update(
                            text=anotacao.text.split("=")[-1]
                        )
                    )

                    fig_multihist.update_annotations(
                        font=dict(size=14)
                    )

                    fig_multihist.update_xaxes(
                        range=[inicio_classes, fim_classes],
                        title_font=dict(size=14),
                        tickfont=dict(size=12),
                        showgrid=True,
                        gridcolor="rgba(0, 0, 0, 0.10)",
                        automargin=True
                    )

                    fig_multihist.update_yaxes(
                        title_text="Frequência",
                        title_font=dict(size=14),
                        tickfont=dict(size=12),
                        rangemode="tozero",
                        showgrid=True,
                        gridcolor="rgba(0, 0, 0, 0.10)",
                        automargin=True
                    )

                    fig_multihist.update_layout(
                        template="plotly_white",
                        height=altura_figura,
                        showlegend=False,
                        bargap=0.05,
                        font=dict(size=12),
                        title_font=dict(size=14),
                        hoverlabel=dict(font_size=12),
                        margin=dict(t=90, r=30, b=70, l=70)
                    )

                    fig_multihist = padronizar_fontes_plotly(
                        fig_multihist
                    )

                    st.plotly_chart(
                        fig_multihist,
                        use_container_width=True,
                        key="grafico_multihist_aba5"
                    )

                    st.caption(
                        "Todos os painéis utilizam os mesmos limites e as "
                        "mesmas classes no eixo horizontal. A altura das "
                        "barras representa a frequência absoluta em cada "
                        "intervalo."
                    )

                    st.subheader("Resumo da variável numérica por nível")

                    resumo_multihist = (
                        dados_multihist
                        .groupby(
                            var_cat_multihist,
                            observed=True
                        )[var_num_multihist]
                        .agg(
                            numero_observacoes="count",
                            media="mean",
                            mediana="median",
                            desvio_padrao="std",
                            minimo="min",
                            maximo="max"
                        )
                        .reindex(niveis_exibidos)
                        .reset_index()
                    )

                    st.dataframe(
                        resumo_multihist.style.format({
                            "media": "{:.3f}",
                            "mediana": "{:.3f}",
                            "desvio_padrao": "{:.3f}",
                            "minimo": "{:.3f}",
                            "maximo": "{:.3f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_multihist = resumo_multihist.to_csv(
                        index=False,
                        decimal=",",
                        sep=";"
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="Baixar resumo dos histogramas em CSV",
                        data=csv_multihist,
                        file_name=(
                            f"resumo_{var_num_multihist}_por_"
                            f"{var_cat_multihist}.csv"
                        ),
                        mime="text/csv",
                        key="download_resumo_multihist_aba5"
                    )



        # ===================================================
        # ABA 6 — CONTINGÊNCIA COM TRÊS VARIÁVEIS CATEGÓRICAS
        # ===================================================
        with aba6:

            st.header(
                "Gráfico de barras e tabela de contingência com três fatores"
            )

            st.markdown(
                "Selecione três variáveis categóricas diferentes. A primeira "
                "será apresentada no eixo horizontal, a segunda definirá os "
                "grupos de barras e a terceira organizará os resultados em "
                "painéis. Também é possível filtrar um nível específico de "
                "cada fator."
            )

            if len(categoricas) < 3:
                st.error(
                    "Esta análise exige pelo menos três variáveis categóricas."
                )

            else:
                col_var1, col_var2, col_var3 = st.columns(3)

                with col_var1:
                    var_cat_3f_1 = st.selectbox(
                        "Primeira variável categórica (eixo x)",
                        categoricas,
                        key="var_cat_3f_1_aba6"
                    )

                opcoes_var2_3f = [
                    variavel for variavel in categoricas
                    if variavel != var_cat_3f_1
                ]

                with col_var2:
                    var_cat_3f_2 = st.selectbox(
                        "Segunda variável categórica (grupos de barras)",
                        opcoes_var2_3f,
                        key="var_cat_3f_2_aba6"
                    )

                opcoes_var3_3f = [
                    variavel for variavel in categoricas
                    if variavel not in [var_cat_3f_1, var_cat_3f_2]
                ]

                with col_var3:
                    var_cat_3f_3 = st.selectbox(
                        "Terceira variável categórica (painéis)",
                        opcoes_var3_3f,
                        key="var_cat_3f_3_aba6"
                    )

                dados_3f = df[
                    [var_cat_3f_1, var_cat_3f_2, var_cat_3f_3]
                ].copy()

                incluir_ausentes_3f = st.checkbox(
                    "Incluir valores ausentes como nível 'Ausente'",
                    value=False,
                    key="incluir_ausentes_3f_aba6"
                )

                if incluir_ausentes_3f:
                    for variavel in [
                        var_cat_3f_1,
                        var_cat_3f_2,
                        var_cat_3f_3
                    ]:
                        dados_3f[variavel] = (
                            dados_3f[variavel]
                            .astype("object")
                            .where(dados_3f[variavel].notna(), "Ausente")
                        )
                else:
                    dados_3f = dados_3f.dropna(
                        subset=[
                            var_cat_3f_1,
                            var_cat_3f_2,
                            var_cat_3f_3
                        ]
                    )

                for variavel in [
                    var_cat_3f_1,
                    var_cat_3f_2,
                    var_cat_3f_3
                ]:
                    dados_3f[variavel] = dados_3f[variavel].astype(str)

                if dados_3f.empty:
                    st.warning(
                        "Não há observações disponíveis para as variáveis "
                        "selecionadas."
                    )

                else:
                    niveis_3f_1 = sorted(
                        dados_3f[var_cat_3f_1].unique().tolist(),
                        key=lambda valor: valor.lower()
                    )
                    niveis_3f_2 = sorted(
                        dados_3f[var_cat_3f_2].unique().tolist(),
                        key=lambda valor: valor.lower()
                    )
                    niveis_3f_3 = sorted(
                        dados_3f[var_cat_3f_3].unique().tolist(),
                        key=lambda valor: valor.lower()
                    )

                    col_nivel1, col_nivel2, col_nivel3 = st.columns(3)

                    with col_nivel1:
                        nivel_3f_1 = st.selectbox(
                            f"Nível de {var_cat_3f_1}",
                            ["Todos os níveis"] + niveis_3f_1,
                            key="nivel_3f_1_aba6"
                        )

                    with col_nivel2:
                        nivel_3f_2 = st.selectbox(
                            f"Nível de {var_cat_3f_2}",
                            ["Todos os níveis"] + niveis_3f_2,
                            key="nivel_3f_2_aba6"
                        )

                    with col_nivel3:
                        nivel_3f_3 = st.selectbox(
                            f"Nível de {var_cat_3f_3}",
                            ["Todos os níveis"] + niveis_3f_3,
                            key="nivel_3f_3_aba6"
                        )

                    dados_3f_filtrados = dados_3f.copy()

                    filtros_3f = {
                        var_cat_3f_1: nivel_3f_1,
                        var_cat_3f_2: nivel_3f_2,
                        var_cat_3f_3: nivel_3f_3
                    }

                    for variavel, nivel in filtros_3f.items():
                        if nivel != "Todos os níveis":
                            dados_3f_filtrados = dados_3f_filtrados[
                                dados_3f_filtrados[variavel] == nivel
                            ]

                    if dados_3f_filtrados.empty:
                        st.warning(
                            "Não existem observações para a combinação de "
                            "níveis selecionada."
                        )

                    else:
                        tabela_3f = (
                            dados_3f_filtrados
                            .groupby(
                                [
                                    var_cat_3f_1,
                                    var_cat_3f_2,
                                    var_cat_3f_3
                                ],
                                dropna=False
                            )
                            .size()
                            .reset_index(name="Frequência")
                        )

                        total_3f = int(tabela_3f["Frequência"].sum())
                        tabela_3f["Frequência relativa (%)"] = (
                            100 * tabela_3f["Frequência"] / total_3f
                        )

                        ordem_x_3f = [
                            nivel for nivel in niveis_3f_1
                            if nivel in tabela_3f[var_cat_3f_1].unique()
                        ]
                        ordem_cor_3f = [
                            nivel for nivel in niveis_3f_2
                            if nivel in tabela_3f[var_cat_3f_2].unique()
                        ]
                        ordem_painel_3f = [
                            nivel for nivel in niveis_3f_3
                            if nivel in tabela_3f[var_cat_3f_3].unique()
                        ]

                        st.write(
                            f"Número total de observações consideradas: "
                            f"**{total_3f}**"
                        )

                        tipo_barra_3f = st.radio(
                            "Organização das barras",
                            ["Barras agrupadas", "Barras empilhadas"],
                            horizontal=True,
                            key="tipo_barra_3f_aba6"
                        )

                        barmode_3f = (
                            "group"
                            if tipo_barra_3f == "Barras agrupadas"
                            else "stack"
                        )

                        numero_paineis_3f = len(ordem_painel_3f)
                        paineis_por_linha_3f = min(
                            max(numero_paineis_3f, 1),
                            3
                        )
                        numero_linhas_3f = int(
                            np.ceil(
                                numero_paineis_3f / paineis_por_linha_3f
                            )
                        )
                        altura_grafico_3f = max(
                            550,
                            390 * numero_linhas_3f
                        )

                        fig_3f = px.bar(
                            tabela_3f,
                            x=var_cat_3f_1,
                            y="Frequência",
                            color=var_cat_3f_2,
                            facet_col=var_cat_3f_3,
                            facet_col_wrap=paineis_por_linha_3f,
                            text="Frequência",
                            barmode=barmode_3f,
                            category_orders={
                                var_cat_3f_1: ordem_x_3f,
                                var_cat_3f_2: ordem_cor_3f,
                                var_cat_3f_3: ordem_painel_3f
                            },
                            labels={
                                var_cat_3f_1: var_cat_3f_1,
                                var_cat_3f_2: var_cat_3f_2,
                                var_cat_3f_3: var_cat_3f_3,
                                "Frequência": "Frequência absoluta"
                            },
                            title=(
                                f"Frequências de {var_cat_3f_1} e "
                                f"{var_cat_3f_2}, por {var_cat_3f_3}"
                            )
                        )

                        fig_3f.update_traces(
                            texttemplate="%{text:.0f}",
                            textposition="outside",
                            textfont=dict(size=20),#cambio 14/07/2026 (original era size=14)
                            cliponaxis=False,
                            marker_line_color="black",
                            marker_line_width=0.8,
                            hovertemplate=(
                                f"{var_cat_3f_1}: %{{x}}<br>"
                                f"{var_cat_3f_2}: %{{fullData.name}}<br>"
                                "Frequência: %{y:.0f}<extra></extra>"
                            )
                        )

                        fig_3f.for_each_annotation(
                            lambda anotacao: anotacao.update(
                                text=anotacao.text.split("=")[-1]
                            )
                        )

                        fig_3f.update_annotations(
                            font=dict(size=14)
                        )

                        fig_3f.update_xaxes(
                            title_font=dict(size=14),
                            tickfont=dict(size=14),
                            automargin=True,
                            categoryorder="array",
                            categoryarray=ordem_x_3f
                        )

                        fig_3f.update_yaxes(
                            title_text="Frequência absoluta",
                            title_font=dict(size=14),
                            tickfont=dict(size=14),
                            rangemode="tozero",
                            automargin=True
                        )

                        fig_3f.update_layout(
                            template="plotly_white",
                            height=altura_grafico_3f,
                            font=dict(size=14),
                            title_font=dict(size=14),
                            legend=dict(
                                title_text=var_cat_3f_2,
                                font=dict(size=14),
                                title_font=dict(size=14),
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="left",
                                x=0
                            ),
                            hoverlabel=dict(font_size=14),
                            margin=dict(t=120, r=30, b=100, l=80),
                            uniformtext_minsize=14,
                            uniformtext_mode="show"
                        )

                        fig_3f = padronizar_fontes_plotly(fig_3f)

                        st.plotly_chart(
                            fig_3f,
                            use_container_width=True,
                            key="grafico_contingencia_3f_aba6"
                        )

                        st.subheader(
                            "Tabela de contingência com três variáveis"
                        )

                        tabela_3f_exibicao = tabela_3f.copy()

                        st.dataframe(
                            tabela_3f_exibicao.style.format({
                                "Frequência": "{:.0f}",
                                "Frequência relativa (%)": "{:.2f}%"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.subheader(
                            "Tabelas cruzadas entre as duas primeiras "
                            "variáveis"
                        )

                        for nivel_painel_3f in ordem_painel_3f:
                            dados_nivel_painel_3f = (
                                dados_3f_filtrados[
                                    dados_3f_filtrados[var_cat_3f_3]
                                    == nivel_painel_3f
                                ]
                            )

                            tabela_cruzada_3f = pd.crosstab(
                                dados_nivel_painel_3f[var_cat_3f_1],
                                dados_nivel_painel_3f[var_cat_3f_2],
                                margins=True,
                                margins_name="Total"
                            )

                            with st.expander(
                                f"{var_cat_3f_3} = {nivel_painel_3f}",
                                expanded=(len(ordem_painel_3f) == 1)
                            ):
                                st.dataframe(
                                    tabela_cruzada_3f,
                                    use_container_width=True
                                )

                        csv_tabela_3f = tabela_3f_exibicao.to_csv(
                            index=False,
                            decimal=",",
                            sep=";"
                        ).encode("utf-8-sig")

                        st.download_button(
                            label=(
                                "Baixar tabela de contingência com três "
                                "fatores em CSV"
                            ),
                            data=csv_tabela_3f,
                            file_name="tabela_contingencia_3_fatores.csv",
                            mime="text/csv",
                            key="download_tabela_3f_aba6"
                        )

    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo: {e}")

else:
    st.info("Envie um arquivo .xlsx para começar.")
