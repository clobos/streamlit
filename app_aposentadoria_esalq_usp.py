# ------------------------------------------------------------
# APLICATIVO STREAMLIT — APOSENTADORIA ESALQ/USP
# ------------------------------------------------------------
# Instalação no Windows:
# py -m venv .venv
# .venv\Scripts\activate.bat
# python -m pip install --upgrade pip
# python -m pip install streamlit pandas numpy openpyxl plotly
#
# Execução:
# python -m streamlit run app_aposentadoria_esalq_usp_corrigido.py
# ------------------------------------------------------------

from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Configuração geral
# ------------------------------------------------------------
st.set_page_config(
    page_title="App de Gráficos com XLSX",
    layout="wide"
)

st.title("Aposentadoria ESALQ USP")

st.markdown(
    """
    Faça upload de um arquivo **.xlsx** e escolha uma análise no menu lateral.

    O aplicativo possui seis análises:

    1. **Histograma filtrado por nível de uma variável categórica**
    2. **Tabela de contingência entre duas variáveis categóricas**
    3. **Análise descritiva das variáveis numéricas e categóricas**
    4. **Frequências absolutas e relativas de uma variável categórica**
    5. **Histogramas comparativos por nível de uma variável categórica**
    6. **Gráfico de barras e contingência com três variáveis categóricas**

    O menu lateral substitui as abas para que apenas a análise selecionada
    seja executada. Isso reduz o uso de memória no Streamlit Community Cloud.
    """
)


# ------------------------------------------------------------
# Constantes de formatação
# ------------------------------------------------------------
FONTE_TITULO_FIGURA = 18
FONTE_EIXOS = 20
FONTE_NIVEIS = 20
FONTE_LEGENDA = 20
FONTE_ROTULOS_BARRAS = 20


# ------------------------------------------------------------
# Leitura e preparação dos dados
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def listar_abas_excel(conteudo_arquivo: bytes) -> list[str]:
    """Retorna os nomes das planilhas do arquivo Excel."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Failed to load a conditional formatting rule.*",
            category=UserWarning
        )
        xls = pd.ExcelFile(
            BytesIO(conteudo_arquivo),
            engine="openpyxl"
        )
        return xls.sheet_names


@st.cache_data(show_spinner=False)
def carregar_excel(
    conteudo_arquivo: bytes,
    sheet_name: str
) -> pd.DataFrame:
    """Lê uma planilha específica do arquivo Excel."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Failed to load a conditional formatting rule.*",
            category=UserWarning
        )
        return pd.read_excel(
            BytesIO(conteudo_arquivo),
            sheet_name=sheet_name,
            engine="openpyxl"
        )


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa espaços vazios e converte para número somente as colunas
    em que todos os valores não ausentes podem ser convertidos.

    Esta implementação não usa errors="ignore", que foi descontinuado
    nas versões recentes do pandas.
    """
    df2 = df.copy()

    colunas_texto = df2.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for coluna in colunas_texto:
        df2[coluna] = df2[coluna].replace(
            r"^\s*$",
            np.nan,
            regex=True
        )

    for coluna in colunas_texto:
        serie_original = df2[coluna]
        serie_convertida = pd.to_numeric(
            serie_original,
            errors="coerce"
        )

        n_validos_original = int(serie_original.notna().sum())
        n_validos_convertido = int(serie_convertida.notna().sum())

        if (
            n_validos_original > 0
            and n_validos_original == n_validos_convertido
        ):
            df2[coluna] = serie_convertida

    return df2


def classificar_variaveis(
    df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Classifica as colunas em numéricas e categóricas."""
    numericas = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categoricas = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    datas = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    categoricas.extend(
        coluna for coluna in datas
        if coluna not in categoricas
    )

    return numericas, categoricas


# ------------------------------------------------------------
# Funções estatísticas e tabelas
# ------------------------------------------------------------
def resumo_variavel(
    df: pd.DataFrame,
    var_num: str
) -> pd.DataFrame:
    """Retorna o resumo descritivo de uma variável numérica."""
    return df[var_num].describe().to_frame().T


def tabela_frequencia_dupla(
    df: pd.DataFrame,
    var_cat1: str,
    var_cat2: str
) -> pd.DataFrame:
    """Calcula frequências conjuntas de duas variáveis categóricas."""
    frequencias = (
        df.groupby(
            [var_cat1, var_cat2],
            dropna=False
        )
        .size()
        .reset_index(name="frequencia")
    )

    total = frequencias["frequencia"].sum()
    frequencias["percentual"] = np.where(
        total > 0,
        100 * frequencias["frequencia"] / total,
        0.0
    )

    return frequencias


def tabela_frequencia_simples(
    df: pd.DataFrame,
    variavel: str,
    incluir_ausentes: bool = True
) -> pd.DataFrame:
    """Calcula frequências absoluta e relativa."""
    serie = df[variavel].copy()

    if incluir_ausentes:
        serie = (
            serie.astype("object")
            .where(serie.notna(), "Ausente")
        )
    else:
        serie = serie.dropna()

    serie = serie.astype(str)

    tabela = (
        serie.value_counts(dropna=False)
        .rename_axis("categoria")
        .reset_index(name="frequencia_absoluta")
    )

    total = tabela["frequencia_absoluta"].sum()

    tabela["frequencia_relativa_%"] = np.where(
        total > 0,
        100 * tabela["frequencia_absoluta"] / total,
        0.0
    )

    return tabela


def analise_descritiva_numericas(
    df: pd.DataFrame,
    numericas: list[str]
) -> pd.DataFrame:
    """Cria uma tabela descritiva para as variáveis numéricas."""
    resumo = df[numericas].describe().T
    resumo["mediana"] = df[numericas].median()
    resumo["variancia"] = df[numericas].var()
    resumo["desvio_padrao"] = df[numericas].std()

    resumo["coef_variacao_%"] = np.where(
        resumo["mean"].ne(0),
        100 * resumo["desvio_padrao"] / resumo["mean"],
        np.nan
    )

    resumo["valores_ausentes"] = df[numericas].isna().sum()
    resumo["percentual_ausentes_%"] = (
        100 * df[numericas].isna().mean()
    )

    return (
        resumo
        .reset_index()
        .rename(columns={"index": "variavel"})
    )


def analise_descritiva_categoricas(
    df: pd.DataFrame,
    categoricas: list[str]
) -> pd.DataFrame:
    """Cria uma tabela descritiva para as variáveis categóricas."""
    resultados = []

    for coluna in categoricas:
        serie = df[coluna]

        total = len(serie)
        ausentes = int(serie.isna().sum())
        validos = total - ausentes
        categorias_unicas = int(serie.nunique(dropna=True))

        moda = serie.mode(dropna=True)
        moda_valor = moda.iloc[0] if not moda.empty else np.nan

        if validos > 0:
            frequencias = serie.value_counts(dropna=True)
            frequencia_moda = int(frequencias.iloc[0])
            percentual_moda = 100 * frequencia_moda / validos
        else:
            frequencia_moda = 0
            percentual_moda = 0.0

        resultados.append({
            "variavel": coluna,
            "total_observacoes": total,
            "valores_validos": validos,
            "valores_ausentes": ausentes,
            "percentual_ausentes_%": (
                100 * ausentes / total if total > 0 else 0.0
            ),
            "categorias_unicas": categorias_unicas,
            "categoria_mais_frequente": moda_valor,
            "frequencia_categoria_mais_frequente": frequencia_moda,
            "percentual_categoria_mais_frequente_%": percentual_moda
        })

    return pd.DataFrame(resultados)


# ------------------------------------------------------------
# Formatação dos gráficos Plotly
# ------------------------------------------------------------
def padronizar_fontes_plotly(fig):
    """
    Padroniza:
    - título da figura: 18;
    - títulos dos eixos: 20;
    - níveis e valores dos eixos: 20;
    - legendas: 20;
    - títulos dos painéis: 20.
    """
    fig.update_layout(
        font=dict(size=FONTE_NIVEIS),
        title_font=dict(size=FONTE_TITULO_FIGURA),
        legend=dict(
            font=dict(size=FONTE_LEGENDA),
            title_font=dict(size=FONTE_LEGENDA)
        ),
        hoverlabel=dict(font_size=FONTE_NIVEIS)
    )

    fig.update_xaxes(
        title_font=dict(size=FONTE_EIXOS),
        tickfont=dict(size=FONTE_NIVEIS),
        automargin=True
    )

    fig.update_yaxes(
        title_font=dict(size=FONTE_EIXOS),
        tickfont=dict(size=FONTE_NIVEIS),
        automargin=True
    )

    fig.update_annotations(
        font=dict(size=FONTE_NIVEIS)
    )

    return fig


# ------------------------------------------------------------
# ANÁLISE 1 — HISTOGRAMA POR UM NÍVEL
# ------------------------------------------------------------
def exibir_histograma(
    df: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str]
) -> None:
    st.header("Histograma por nível de variável categórica")

    if not numericas or not categoricas:
        st.error(
            "Esta análise exige pelo menos uma variável numérica "
            "e uma variável categórica."
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        var_cat = st.selectbox(
            "Escolha a variável categórica",
            categoricas,
            key="hist_var_cat"
        )

    niveis = (
        df[var_cat]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if not niveis:
        st.warning("A variável categórica selecionada não possui níveis válidos.")
        return

    with col2:
        nivel = st.selectbox(
            "Escolha o nível da categoria",
            niveis,
            key="hist_nivel"
        )

    with col3:
        var_num = st.selectbox(
            "Escolha a variável numérica",
            numericas,
            key="hist_var_num"
        )

    nbins = st.slider(
        "Número de classes do histograma",
        min_value=5,
        max_value=80,
        value=20,
        key="hist_nbins"
    )

    dados = df[[var_cat, var_num]].copy()
    dados = dados.dropna(subset=[var_cat, var_num])
    dados[var_cat] = dados[var_cat].astype(str)
    dados = dados[dados[var_cat] == nivel]

    if dados.empty:
        st.error("Não há observações para a seleção realizada.")
        return

    st.write(f"Categoria selecionada: **{var_cat}**")
    st.write(f"Nível selecionado: **{nivel}**")
    st.write(f"Variável numérica: **{var_num}**")
    st.write(f"Número de observações: **{len(dados)}**")

    fig = px.histogram(
        dados,
        x=var_num,
        nbins=nbins,
        histnorm=None,
        title=f"Histograma de {var_num}: {var_cat} = {nivel}"
    )

    fig.update_traces(
        marker_color="red",
        marker_line_color="black",
        marker_line_width=1,
        hovertemplate=(
            "Centro da classe: %{x}<br>"
            "Frequência absoluta: %{y}<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        xaxis_title=var_num,
        yaxis_title="Frequência absoluta",
        margin=dict(t=90, r=40, b=90, l=90)
    )

    fig = padronizar_fontes_plotly(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="hist_grafico"
    )

    st.subheader("Resumo estatístico")
    tabela_resumo = resumo_variavel(dados, var_num)

    st.dataframe(
        tabela_resumo,
        use_container_width=True
    )

    csv = tabela_resumo.to_csv(
        index=False,
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar resumo do histograma em CSV",
        data=csv,
        file_name="resumo_histograma.csv",
        mime="text/csv",
        key="hist_download"
    )


# ------------------------------------------------------------
# ANÁLISE 2 — CONTINGÊNCIA COM DUAS VARIÁVEIS
# ------------------------------------------------------------
def exibir_contingencia_dupla(
    df: pd.DataFrame,
    categoricas: list[str]
) -> None:
    st.header("Tabela de contingência entre duas variáveis categóricas")

    if len(categoricas) < 2:
        st.error(
            "Esta análise exige pelo menos duas variáveis categóricas."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        var_cat1 = st.selectbox(
            "Escolha a primeira variável categórica",
            categoricas,
            key="cont2_var1"
        )

    opcoes_var2 = [
        coluna for coluna in categoricas
        if coluna != var_cat1
    ]

    with col2:
        var_cat2 = st.selectbox(
            "Escolha a segunda variável categórica",
            opcoes_var2,
            key="cont2_var2"
        )

    incluir_ausentes = st.checkbox(
        "Incluir valores ausentes como categoria 'Ausente'",
        value=False,
        key="cont2_ausentes"
    )

    dados = df[[var_cat1, var_cat2]].copy()

    if incluir_ausentes:
        for coluna in [var_cat1, var_cat2]:
            dados[coluna] = (
                dados[coluna]
                .astype("object")
                .where(dados[coluna].notna(), "Ausente")
            )
    else:
        dados = dados.dropna(subset=[var_cat1, var_cat2])

    if dados.empty:
        st.error("Não há dados disponíveis para as variáveis selecionadas.")
        return

    dados[var_cat1] = dados[var_cat1].astype(str)
    dados[var_cat2] = dados[var_cat2].astype(str)

    tabela_contingencia = pd.crosstab(
        dados[var_cat1],
        dados[var_cat2],
        margins=True,
        margins_name="Total"
    )

    st.subheader("Tabela de contingência")
    st.dataframe(
        tabela_contingencia,
        use_container_width=True
    )

    tabela_grafico = tabela_frequencia_dupla(
        dados,
        var_cat1,
        var_cat2
    )

    st.subheader("Gráfico de barras")

    tipo_barra = st.radio(
        "Tipo de gráfico",
        ["Barras agrupadas", "Barras empilhadas"],
        horizontal=True,
        key="cont2_tipo"
    )

    barmode = (
        "group"
        if tipo_barra == "Barras agrupadas"
        else "stack"
    )

    fig = px.bar(
        tabela_grafico,
        x=var_cat1,
        y="frequencia",
        color=var_cat2,
        text="frequencia",
        barmode=barmode,
        title=f"Frequência conjunta de {var_cat1} e {var_cat2}",
        labels={"frequencia": "Frequência absoluta"}
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        textfont=dict(size=FONTE_ROTULOS_BARRAS),
        cliponaxis=False,
        marker_line_color="black",
        marker_line_width=0.8
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        xaxis_title=var_cat1,
        yaxis_title="Frequência absoluta",
        margin=dict(t=100, r=50, b=110, l=90)
    )

    fig.update_yaxes(
        rangemode="tozero"
    )

    fig = padronizar_fontes_plotly(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cont2_grafico"
    )

    csv = tabela_contingencia.to_csv(
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar tabela de contingência em CSV",
        data=csv,
        file_name="tabela_contingencia.csv",
        mime="text/csv",
        key="cont2_download"
    )


# ------------------------------------------------------------
# ANÁLISE 3 — ANÁLISE DESCRITIVA
# ------------------------------------------------------------
def exibir_analise_descritiva(
    df: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str]
) -> None:
    st.header("Análise descritiva das variáveis")

    st.subheader("Resumo geral do banco de dados")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Número de linhas", df.shape[0])
    col2.metric("Número de colunas", df.shape[1])
    col3.metric("Variáveis numéricas", len(numericas))
    col4.metric("Variáveis categóricas", len(categoricas))

    st.divider()
    st.subheader("Variáveis numéricas")

    if not numericas:
        st.warning("Não há variáveis numéricas para descrever.")
    else:
        resumo_num = analise_descritiva_numericas(
            df,
            numericas
        )

        st.dataframe(
            resumo_num,
            use_container_width=True,
            hide_index=True
        )

        csv_num = resumo_num.to_csv(
            index=False,
            decimal=",",
            sep=";"
        ).encode("utf-8-sig")

        st.download_button(
            "Baixar análise numérica em CSV",
            data=csv_num,
            file_name="analise_descritiva_numericas.csv",
            mime="text/csv",
            key="desc_download_num"
        )

    st.divider()
    st.subheader("Variáveis categóricas")

    if not categoricas:
        st.warning("Não há variáveis categóricas para descrever.")
        return

    resumo_cat = analise_descritiva_categoricas(
        df,
        categoricas
    )

    st.dataframe(
        resumo_cat,
        use_container_width=True,
        hide_index=True
    )

    csv_cat = resumo_cat.to_csv(
        index=False,
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar análise categórica em CSV",
        data=csv_cat,
        file_name="analise_descritiva_categoricas.csv",
        mime="text/csv",
        key="desc_download_cat"
    )

    st.subheader("Tabela e gráfico de frequência")

    var_cat = st.selectbox(
        "Escolha uma variável categórica",
        categoricas,
        key="desc_var_cat"
    )

    tabela = tabela_frequencia_simples(
        df,
        var_cat,
        incluir_ausentes=True
    )

    st.dataframe(
        tabela.rename(columns={
            "categoria": var_cat,
            "frequencia_absoluta": "Frequência absoluta",
            "frequencia_relativa_%": "Frequência relativa (%)"
        }).style.format({
            "Frequência relativa (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    ordem = tabela["categoria"].tolist()

    fig = px.bar(
        tabela,
        x="categoria",
        y="frequencia_absoluta",
        text="frequencia_absoluta",
        category_orders={"categoria": ordem},
        labels={
            "categoria": var_cat,
            "frequencia_absoluta": "Frequência absoluta"
        },
        title=f"Frequências das categorias de {var_cat}"
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        textfont=dict(size=FONTE_ROTULOS_BARRAS),
        cliponaxis=False,
        marker_line_color="black",
        marker_line_width=0.8
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        xaxis_title=var_cat,
        yaxis_title="Frequência absoluta",
        margin=dict(t=100, r=50, b=120, l=90)
    )

    fig.update_yaxes(
        rangemode="tozero",
        range=[
            0,
            max(
                float(tabela["frequencia_absoluta"].max()) * 1.20,
                1
            )
        ]
    )

    fig = padronizar_fontes_plotly(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="desc_grafico"
    )


# ------------------------------------------------------------
# ANÁLISE 4 — FREQUÊNCIAS CATEGÓRICAS
# ------------------------------------------------------------
def exibir_frequencias_categoricas(
    df: pd.DataFrame,
    categoricas: list[str]
) -> None:
    st.header("Frequências de uma variável categórica")

    if not categoricas:
        st.warning("Não há variáveis categóricas disponíveis.")
        return

    col1, col2 = st.columns(2)

    with col1:
        var_cat = st.selectbox(
            "Escolha a variável categórica",
            categoricas,
            key="freq_var_cat"
        )

    with col2:
        ordenacao = st.selectbox(
            "Ordenar as categorias por",
            [
                "Frequência decrescente",
                "Frequência crescente",
                "Ordem alfabética"
            ],
            key="freq_ordenacao"
        )

    incluir_ausentes = st.checkbox(
        "Incluir valores ausentes como categoria 'Ausente'",
        value=True,
        key="freq_ausentes"
    )

    tabela = tabela_frequencia_simples(
        df,
        var_cat,
        incluir_ausentes
    )

    if ordenacao == "Frequência crescente":
        tabela = tabela.sort_values(
            "frequencia_absoluta",
            ascending=True
        )
    elif ordenacao == "Ordem alfabética":
        tabela = tabela.sort_values(
            "categoria",
            key=lambda serie: serie.str.lower()
        )
    else:
        tabela = tabela.sort_values(
            "frequencia_absoluta",
            ascending=False
        )

    tabela = tabela.reset_index(drop=True)

    if tabela.empty:
        st.error("Não há observações para a variável selecionada.")
        return

    total = int(tabela["frequencia_absoluta"].sum())
    st.write(f"Total de observações consideradas: **{total}**")

    ordem = tabela["categoria"].tolist()
    altura = max(550, 50 * len(ordem) + 180)

    fig_abs = px.bar(
        tabela,
        x="frequencia_absoluta",
        y="categoria",
        text="frequencia_absoluta",
        orientation="h",
        category_orders={"categoria": ordem},
        labels={
            "categoria": var_cat,
            "frequencia_absoluta": "Frequência absoluta"
        },
        title="Frequências absolutas"
    )

    fig_abs.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        textfont=dict(size=FONTE_ROTULOS_BARRAS),
        cliponaxis=False,
        marker_line_color="black",
        marker_line_width=1,
        hovertemplate=(
            f"{var_cat}: %{{y}}<br>"
            "Frequência absoluta: %{x:.0f}<extra></extra>"
        )
    )

    fig_abs.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(t=90, r=110, b=90, l=140),
        xaxis_title="Frequência absoluta",
        yaxis_title=var_cat
    )

    fig_abs.update_xaxes(
        rangemode="tozero",
        range=[
            0,
            max(
                float(tabela["frequencia_absoluta"].max()) * 1.25,
                1
            )
        ]
    )

    fig_abs.update_yaxes(
        categoryorder="array",
        categoryarray=ordem,
        autorange="reversed"
    )

    fig_rel = px.bar(
        tabela,
        x="frequencia_relativa_%",
        y="categoria",
        text="frequencia_relativa_%",
        orientation="h",
        category_orders={"categoria": ordem},
        labels={
            "categoria": var_cat,
            "frequencia_relativa_%": "Frequência relativa (%)"
        },
        title="Frequências relativas"
    )

    fig_rel.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(size=FONTE_ROTULOS_BARRAS),
        cliponaxis=False,
        marker_line_color="black",
        marker_line_width=1,
        hovertemplate=(
            f"{var_cat}: %{{y}}<br>"
            "Frequência relativa: %{x:.2f}%<extra></extra>"
        )
    )

    fig_rel.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(t=90, r=110, b=90, l=140),
        xaxis_title="Frequência relativa (%)",
        yaxis_title=var_cat
    )

    fig_rel.update_xaxes(
        ticksuffix="%",
        rangemode="tozero",
        range=[
            0,
            max(
                float(tabela["frequencia_relativa_%"].max()) * 1.25,
                1
            )
        ]
    )

    fig_rel.update_yaxes(
        categoryorder="array",
        categoryarray=ordem,
        autorange="reversed"
    )

    fig_abs = padronizar_fontes_plotly(fig_abs)
    fig_rel = padronizar_fontes_plotly(fig_rel)

    col_abs, col_rel = st.columns(2)

    with col_abs:
        st.plotly_chart(
            fig_abs,
            use_container_width=True,
            key="freq_grafico_abs"
        )

    with col_rel:
        st.plotly_chart(
            fig_rel,
            use_container_width=True,
            key="freq_grafico_rel"
        )

    st.subheader("Tabela de frequências")

    tabela_exibicao = tabela.rename(columns={
        "categoria": var_cat,
        "frequencia_absoluta": "Frequência absoluta",
        "frequencia_relativa_%": "Frequência relativa (%)"
    })

    st.dataframe(
        tabela_exibicao.style.format({
            "Frequência relativa (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    csv = tabela_exibicao.to_csv(
        index=False,
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar tabela de frequências em CSV",
        data=csv,
        file_name=f"frequencias_{var_cat}.csv",
        mime="text/csv",
        key="freq_download"
    )


# ------------------------------------------------------------
# ANÁLISE 5 — HISTOGRAMAS POR NÍVEL
# ------------------------------------------------------------
def exibir_histogramas_por_nivel(
    df: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str]
) -> None:
    st.header("Histogramas da variável numérica por nível do fator")

    st.markdown(
        "Todos os painéis utilizam os mesmos limites e os mesmos "
        "intervalos de classe. A altura de cada barra representa a "
        "frequência absoluta dentro daquele intervalo."
    )

    if not numericas or not categoricas:
        st.error(
            "Esta análise exige pelo menos uma variável numérica "
            "e uma variável categórica."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        var_num = st.selectbox(
            "Escolha a variável numérica",
            numericas,
            key="multi_var_num"
        )

    with col2:
        var_cat = st.selectbox(
            "Escolha a variável categórica",
            categoricas,
            key="multi_var_cat"
        )

    col_bins, col_paineis, col_ordem = st.columns(3)

    with col_bins:
        nbins = st.slider(
            "Número de classes",
            min_value=3,
            max_value=30,
            value=8,
            key="multi_nbins"
        )

    with col_paineis:
        paineis_por_linha = st.slider(
            "Painéis por linha",
            min_value=1,
            max_value=4,
            value=3,
            key="multi_paineis"
        )

    with col_ordem:
        ordenacao = st.selectbox(
            "Ordenar os níveis por",
            [
                "Ordem alfabética",
                "Frequência decrescente",
                "Frequência crescente"
            ],
            key="multi_ordenacao"
        )

    incluir_ausentes = st.checkbox(
        "Incluir valores ausentes da variável categórica "
        "como nível 'Ausente'",
        value=False,
        key="multi_ausentes"
    )

    dados = df[[var_num, var_cat]].copy()
    dados = dados.dropna(subset=[var_num])

    if incluir_ausentes:
        dados[var_cat] = (
            dados[var_cat]
            .astype("object")
            .where(dados[var_cat].notna(), "Ausente")
        )
    else:
        dados = dados.dropna(subset=[var_cat])

    dados[var_cat] = dados[var_cat].astype(str)

    contagens = dados[var_cat].value_counts()

    if ordenacao == "Frequência decrescente":
        niveis_ordenados = contagens.index.tolist()
    elif ordenacao == "Frequência crescente":
        niveis_ordenados = (
            contagens
            .sort_values(ascending=True)
            .index
            .tolist()
        )
    else:
        niveis_ordenados = sorted(
            contagens.index.tolist(),
            key=str.lower
        )

    niveis_selecionados = st.multiselect(
        "Níveis que serão exibidos",
        options=niveis_ordenados,
        default=niveis_ordenados,
        key="multi_niveis"
    )

    if not niveis_selecionados:
        st.warning("Selecione pelo menos um nível.")
        return

    dados = dados[
        dados[var_cat].isin(niveis_selecionados)
    ].copy()

    if dados.empty:
        st.warning("Não há observações para a seleção realizada.")
        return

    niveis_exibidos = [
        nivel for nivel in niveis_ordenados
        if nivel in niveis_selecionados
    ]

    dados[var_cat] = pd.Categorical(
        dados[var_cat],
        categories=niveis_exibidos,
        ordered=True
    )

    minimo = float(dados[var_num].min())
    maximo = float(dados[var_num].max())

    if minimo == maximo:
        amplitude = max(abs(minimo) * 0.10, 1.0)
        inicio_classes = minimo - amplitude / 2
        fim_classes = maximo + amplitude / 2
        largura_classe = amplitude
    else:
        inicio_classes = minimo
        fim_classes = maximo
        largura_classe = (maximo - minimo) / nbins

    numero_linhas = int(
        np.ceil(len(niveis_exibidos) / paineis_por_linha)
    )
    altura = max(550, 380 * numero_linhas)

    fig = px.histogram(
        dados,
        x=var_num,
        facet_col=var_cat,
        facet_col_wrap=paineis_por_linha,
        histnorm=None,
        category_orders={var_cat: niveis_exibidos},
        labels={
            var_num: var_num,
            var_cat: var_cat
        },
        title=f"Distribuição de {var_num} por níveis de {var_cat}"
    )

    fig.update_traces(
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
            "Frequência absoluta: %{y}<extra></extra>"
        )
    )

    contagens_paineis = (
        dados[var_cat]
        .value_counts()
        .to_dict()
    )

    def atualizar_titulo_painel(anotacao) -> None:
        nivel = anotacao.text.split("=")[-1]
        n = int(contagens_paineis.get(nivel, 0))
        anotacao.update(text=f"{nivel} (n = {n})")

    fig.for_each_annotation(atualizar_titulo_painel)

    fig.update_xaxes(
        range=[inicio_classes, fim_classes],
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.10)"
    )

    # Todos os painéis usam a mesma escala vertical.
    fig.update_yaxes(
        matches="y",
        title_text="Frequência absoluta",
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.10)"
    )

    fig.update_layout(
        template="plotly_white",
        height=altura,
        showlegend=False,
        bargap=0.05,
        margin=dict(t=110, r=40, b=100, l=90)
    )

    fig = padronizar_fontes_plotly(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="multi_grafico"
    )

    st.caption(
        "O valor n apresentado em cada painel é o total de observações "
        "do nível. A soma das barras de cada painel corresponde a esse n."
    )

    st.subheader("Resumo da variável numérica por nível")

    resumo = (
        dados
        .groupby(
            var_cat,
            observed=True
        )[var_num]
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
        resumo.style.format({
            "media": "{:.3f}",
            "mediana": "{:.3f}",
            "desvio_padrao": "{:.3f}",
            "minimo": "{:.3f}",
            "maximo": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    csv = resumo.to_csv(
        index=False,
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar resumo dos histogramas em CSV",
        data=csv,
        file_name=f"resumo_{var_num}_por_{var_cat}.csv",
        mime="text/csv",
        key="multi_download"
    )


# ------------------------------------------------------------
# ANÁLISE 6 — CONTINGÊNCIA COM TRÊS FATORES
# ------------------------------------------------------------
def exibir_contingencia_tres_fatores(
    df: pd.DataFrame,
    categoricas: list[str]
) -> None:
    st.header(
        "Gráfico de barras e tabela de contingência com três fatores"
    )

    if len(categoricas) < 3:
        st.error(
            "Esta análise exige pelo menos três variáveis categóricas."
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        var1 = st.selectbox(
            "Primeira variável categórica (eixo x)",
            categoricas,
            key="cont3_var1"
        )

    opcoes_var2 = [
        coluna for coluna in categoricas
        if coluna != var1
    ]

    with col2:
        var2 = st.selectbox(
            "Segunda variável categórica (grupos)",
            opcoes_var2,
            key="cont3_var2"
        )

    opcoes_var3 = [
        coluna for coluna in categoricas
        if coluna not in [var1, var2]
    ]

    with col3:
        var3 = st.selectbox(
            "Terceira variável categórica (painéis)",
            opcoes_var3,
            key="cont3_var3"
        )

    incluir_ausentes = st.checkbox(
        "Incluir valores ausentes como nível 'Ausente'",
        value=False,
        key="cont3_ausentes"
    )

    dados = df[[var1, var2, var3]].copy()

    if incluir_ausentes:
        for coluna in [var1, var2, var3]:
            dados[coluna] = (
                dados[coluna]
                .astype("object")
                .where(dados[coluna].notna(), "Ausente")
            )
    else:
        dados = dados.dropna(subset=[var1, var2, var3])

    for coluna in [var1, var2, var3]:
        dados[coluna] = dados[coluna].astype(str)

    if dados.empty:
        st.warning("Não há observações para as variáveis selecionadas.")
        return

    niveis1 = sorted(dados[var1].unique().tolist(), key=str.lower)
    niveis2 = sorted(dados[var2].unique().tolist(), key=str.lower)
    niveis3 = sorted(dados[var3].unique().tolist(), key=str.lower)

    col_n1, col_n2, col_n3 = st.columns(3)

    with col_n1:
        nivel1 = st.selectbox(
            f"Nível de {var1}",
            ["Todos os níveis"] + niveis1,
            key="cont3_nivel1"
        )

    with col_n2:
        nivel2 = st.selectbox(
            f"Nível de {var2}",
            ["Todos os níveis"] + niveis2,
            key="cont3_nivel2"
        )

    with col_n3:
        nivel3 = st.selectbox(
            f"Nível de {var3}",
            ["Todos os níveis"] + niveis3,
            key="cont3_nivel3"
        )

    filtros = {
        var1: nivel1,
        var2: nivel2,
        var3: nivel3
    }

    dados_filtrados = dados.copy()

    for coluna, nivel in filtros.items():
        if nivel != "Todos os níveis":
            dados_filtrados = dados_filtrados[
                dados_filtrados[coluna] == nivel
            ]

    if dados_filtrados.empty:
        st.warning(
            "Não existem observações para a combinação selecionada."
        )
        return

    tabela = (
        dados_filtrados
        .groupby(
            [var1, var2, var3],
            dropna=False
        )
        .size()
        .reset_index(name="Frequência")
    )

    total = int(tabela["Frequência"].sum())
    tabela["Frequência relativa (%)"] = (
        100 * tabela["Frequência"] / total
    )

    ordem_x = [
        nivel for nivel in niveis1
        if nivel in tabela[var1].unique()
    ]
    ordem_cor = [
        nivel for nivel in niveis2
        if nivel in tabela[var2].unique()
    ]
    ordem_painel = [
        nivel for nivel in niveis3
        if nivel in tabela[var3].unique()
    ]

    st.write(
        f"Número total de observações consideradas: **{total}**"
    )

    tipo = st.radio(
        "Organização das barras",
        ["Barras agrupadas", "Barras empilhadas"],
        horizontal=True,
        key="cont3_tipo"
    )

    barmode = "group" if tipo == "Barras agrupadas" else "stack"

    paineis_por_linha = min(max(len(ordem_painel), 1), 3)
    numero_linhas = int(
        np.ceil(len(ordem_painel) / paineis_por_linha)
    )
    altura = max(600, 430 * numero_linhas)

    fig = px.bar(
        tabela,
        x=var1,
        y="Frequência",
        color=var2,
        facet_col=var3,
        facet_col_wrap=paineis_por_linha,
        text="Frequência",
        barmode=barmode,
        category_orders={
            var1: ordem_x,
            var2: ordem_cor,
            var3: ordem_painel
        },
        labels={"Frequência": "Frequência absoluta"},
        title=f"Frequências de {var1} e {var2}, por {var3}"
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        textfont=dict(size=FONTE_ROTULOS_BARRAS),
        cliponaxis=False,
        marker_line_color="black",
        marker_line_width=0.8
    )

    fig.for_each_annotation(
        lambda anotacao: anotacao.update(
            text=anotacao.text.split("=")[-1]
        )
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=ordem_x
    )

    fig.update_yaxes(
        title_text="Frequência absoluta",
        rangemode="tozero"
    )

    fig.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(t=140, r=50, b=120, l=100),
        legend=dict(
            title_text=var2,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig = padronizar_fontes_plotly(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cont3_grafico"
    )

    st.subheader("Tabela de contingência com três variáveis")

    st.dataframe(
        tabela.style.format({
            "Frequência": "{:.0f}",
            "Frequência relativa (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Tabelas cruzadas por painel")

    for nivel_painel in ordem_painel:
        dados_painel = dados_filtrados[
            dados_filtrados[var3] == nivel_painel
        ]

        tabela_cruzada = pd.crosstab(
            dados_painel[var1],
            dados_painel[var2],
            margins=True,
            margins_name="Total"
        )

        with st.expander(
            f"{var3} = {nivel_painel}",
            expanded=(len(ordem_painel) == 1)
        ):
            st.dataframe(
                tabela_cruzada,
                use_container_width=True
            )

    csv = tabela.to_csv(
        index=False,
        decimal=",",
        sep=";"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar tabela de contingência com três fatores em CSV",
        data=csv,
        file_name="tabela_contingencia_3_fatores.csv",
        mime="text/csv",
        key="cont3_download"
    )


# ------------------------------------------------------------
# Upload e navegação
# ------------------------------------------------------------
arquivo = st.file_uploader(
    "Envie o arquivo XLSX",
    type=["xlsx"]
)

if arquivo is None:
    st.info("Envie um arquivo .xlsx para começar.")
    st.stop()

try:
    conteudo_arquivo = arquivo.getvalue()

    abas_planilha = listar_abas_excel(conteudo_arquivo)

    aba_planilha = st.selectbox(
        "Selecione a aba da planilha",
        abas_planilha,
        index=0,
        key="planilha_selecionada"
    )

    with st.spinner("Lendo e preparando os dados..."):
        df = carregar_excel(
            conteudo_arquivo,
            aba_planilha
        )
        df = preparar_dados(df)

    numericas, categoricas = classificar_variaveis(df)

    with st.expander("Pré-visualização dos dados", expanded=False):
        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    st.sidebar.header("Navegação")

    analise_selecionada = st.sidebar.radio(
        "Selecione a análise",
        [
            "Histograma",
            "Tabela de contingência",
            "Análise descritiva",
            "Frequências categóricas",
            "Histogramas por nível",
            "Contingência com 3 fatores"
        ],
        key="analise_selecionada"
    )

    st.sidebar.divider()
    st.sidebar.metric("Linhas", df.shape[0])
    st.sidebar.metric("Colunas", df.shape[1])
    st.sidebar.metric("Variáveis numéricas", len(numericas))
    st.sidebar.metric("Variáveis categóricas", len(categoricas))

    if st.sidebar.button(
        "Limpar cache dos dados",
        key="limpar_cache"
    ):
        st.cache_data.clear()
        st.rerun()

    # Apenas uma função de análise é executada por vez.
    if analise_selecionada == "Histograma":
        exibir_histograma(
            df,
            numericas,
            categoricas
        )

    elif analise_selecionada == "Tabela de contingência":
        exibir_contingencia_dupla(
            df,
            categoricas
        )

    elif analise_selecionada == "Análise descritiva":
        exibir_analise_descritiva(
            df,
            numericas,
            categoricas
        )

    elif analise_selecionada == "Frequências categóricas":
        exibir_frequencias_categoricas(
            df,
            categoricas
        )

    elif analise_selecionada == "Histogramas por nível":
        exibir_histogramas_por_nivel(
            df,
            numericas,
            categoricas
        )

    elif analise_selecionada == "Contingência com 3 fatores":
        exibir_contingencia_tres_fatores(
            df,
            categoricas
        )

except MemoryError:
    st.error(
        "O aplicativo ficou sem memória ao processar os dados. "
        "Tente uma planilha menor ou reduza o número de níveis exibidos."
    )

except Exception as erro:
    st.error(
        "Não foi possível ler ou processar o arquivo."
    )
    st.exception(erro)
