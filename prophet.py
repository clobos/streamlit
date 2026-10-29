#py -m venv .venv
#.venv\Scripts\activate.bat 
#py -m pip install --upgrade pip
#py -m pip install streamlit openpyxl
#py -m streamlit run opiniaoCampus.py

"""
App Streamlit: análise e previsão da série temporal anual de emissões de metano (CH4).

Fluxo:
1. Upload de planilha .xlsx (formato longo ou largo) e limpeza dos dados.
2. Filtros em cascata: Estado -> Cidade.
3. Gráfico da série histórica (Plotly).
4. Previsão com Prophet, com hiperparâmetros ajustáveis na barra lateral.
5. Gráfico final com observado, previsão e intervalo de confiança de 95%.

Para executar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import logging
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# Reduz o ruído de log do cmdstanpy (backend do Prophet) no terminal
logging.getLogger("cmdstanpy").disabled = True

# Número mínimo de observações anuais para ajustar o Prophet
MIN_OBSERVACOES = 3
# Horizonte de previsão: 2 anos à frente, frequência anual (início de ano)
HORIZONTE = 2
FREQ = "YS"


# ---------------------------------------------------------------------------
# 1. IMPORTAÇÃO E LIMPEZA DE DADOS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lendo a planilha...")
def carregar_excel(conteudo: bytes) -> pd.DataFrame:
    """Lê a primeira aba do arquivo .xlsx (engine openpyxl).

    Recebe os bytes do arquivo (e não o objeto do uploader) para que o
    cache do Streamlit funcione corretamente: só relê se o arquivo mudar.
    """
    return pd.read_excel(io.BytesIO(conteudo), sheet_name=0, engine="openpyxl")


def encontrar_coluna(df: pd.DataFrame, nome: str):
    """Retorna o nome real da coluna cujo texto (sem espaços, minúsculo) é `nome`."""
    for c in df.columns:
        if str(c).strip().lower() == nome:
            return c
    return None


def para_formato_longo(df: pd.DataFrame) -> pd.DataFrame:
    """Garante o formato longo com colunas Estado, Cidade, ano e ch4.

    - Se já houver colunas 'ano' e 'ch4', apenas seleciona e renomeia.
    - Se o arquivo estiver em formato largo (um ano por coluna), aplica
      pd.melt (equivalente ao pivot_longer do R).
    """
    # Remove espaços nos nomes das colunas
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_estado = encontrar_coluna(df, "estado")
    col_cidade = encontrar_coluna(df, "cidade")
    if col_estado is None or col_cidade is None:
        raise ValueError("A planilha precisa ter as colunas 'Estado' e 'Cidade'.")

    col_ano = encontrar_coluna(df, "ano")
    col_ch4 = encontrar_coluna(df, "ch4")

    # Caso 1: formato longo
    if col_ano is not None and col_ch4 is not None:
        longo = df[[col_estado, col_cidade, col_ano, col_ch4]].copy()
        longo.columns = ["Estado", "Cidade", "ano", "ch4"]
        return longo

    # Caso 2: formato largo -> colunas cujo nome é um ano (ex.: 2000, "2000", "2000.0")
    anos = [c for c in df.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))]
    if not anos:
        raise ValueError(
            "Formato não reconhecido. Use colunas 'Estado', 'Cidade', 'ano' e 'ch4' "
            "(formato longo) ou uma coluna por ano (formato largo)."
        )

    largo = df.melt(
        id_vars=[col_estado, col_cidade],
        value_vars=anos,
        var_name="ano",
        value_name="ch4",
    )
    largo.columns = ["Estado", "Cidade", "ano", "ch4"]
    return largo


def limpar_dados(df: pd.DataFrame):
    """Converte tipos e remove linhas com valores ausentes.

    Retorna o DataFrame limpo e o número de linhas descartadas, para
    informar o usuário.
    """
    total_antes = len(df)

    # Converte para numérico; valores não numéricos viram NaN
    df = df.copy()
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["ch4"] = pd.to_numeric(df["ch4"], errors="coerce")

    # Remove linhas sem Estado, Cidade, ano ou ch4
    df = df.dropna(subset=["Estado", "Cidade", "ano", "ch4"])

    # Ajusta tipos e textos
    df["ano"] = df["ano"].astype(int)
    df["Estado"] = df["Estado"].astype(str).str.strip()
    df["Cidade"] = df["Cidade"].astype(str).str.strip()

    descartados = total_antes - len(df)
    return df.reset_index(drop=True), descartados


# ---------------------------------------------------------------------------
# 2. FILTRAGEM
# ---------------------------------------------------------------------------
def filtrar_serie(df: pd.DataFrame, estado: str, cidade: str) -> pd.DataFrame:
    """Retorna a série anual de uma cidade, ordenada por ano."""
    serie = df[(df["Estado"] == estado) & (df["Cidade"] == cidade)]
    return serie.sort_values("ano").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. GRÁFICO HISTÓRICO
# ---------------------------------------------------------------------------
def grafico_historico(serie: pd.DataFrame, estado: str, cidade: str):
    """Linha da série histórica de CH4 com título dinâmico."""
    fig = px.line(
        serie,
        x="ano",
        y="ch4",
        markers=True,
        title=f"Emissões de CH4 em {cidade} ({estado})",
        labels={"ano": "Ano", "ch4": "CH4"},
    )
    fig.update_xaxes(tickformat="d")  # ano sem separador de milhar
    return fig


# ---------------------------------------------------------------------------
# 4. MODELAGEM COM PROPHET
# ---------------------------------------------------------------------------
def preparar_prophet(serie: pd.DataFrame) -> pd.DataFrame:
    """Converte a série para o formato do Prophet: ds (data) e y (valor).

    A data é construída como 1º de janeiro de cada ano. Se houver mais de
    uma linha por ano para a mesma cidade, usa-se a média.
    """
    base = serie.groupby("ano", as_index=False)["ch4"].mean()
    base["ds"] = pd.to_datetime(base["ano"].astype(str) + "-01-01")
    base = base.rename(columns={"ch4": "y"})
    return base[["ds", "y"]].sort_values("ds").reset_index(drop=True)


@st.cache_resource(show_spinner="Ajustando o modelo Prophet...")
def ajustar_prophet(dados: pd.DataFrame, params: dict) -> Prophet:
    """Cria e ajusta o modelo Prophet.

    Cacheado com st.cache_resource: o modelo só é reajustado quando os
    dados ou os hiperparâmetros mudarem.
    """
    modelo = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
        interval_width=0.95,  # fixo, conforme especificação
    )
    modelo.fit(dados)
    return modelo


def prever(modelo: Prophet) -> pd.DataFrame:
    """Gera o histórico ajustado mais 2 anos de previsão (frequência anual)."""
    futuro = modelo.make_future_dataframe(periods=HORIZONTE, freq=FREQ)
    return modelo.predict(futuro)


# ---------------------------------------------------------------------------
# 5. GRÁFICO FINAL: OBSERVADO + PREVISÃO + IC 95%
# ---------------------------------------------------------------------------
def grafico_previsao(hist: pd.DataFrame, previsao: pd.DataFrame, estado: str, cidade: str):
    """Um único gráfico com série original, previsão e faixa de confiança."""
    # Converte a coluna ds de volta para ano, para o eixo x
    previsao = previsao.copy()
    previsao["ano"] = previsao["ds"].dt.year

    fig = go.Figure()

    # Faixa do IC 95%: primeiro o limite superior, depois o inferior com
    # preenchimento até o traço anterior (tonexty). Adicionados antes das
    # outras linhas para ficarem ao fundo.
    fig.add_trace(go.Scatter(
        x=previsao["ano"], y=previsao["yhat_upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=previsao["ano"], y=previsao["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31, 119, 180, 0.2)",
        name="IC 95%",
    ))

    # Série original, com marcadores
    fig.add_trace(go.Scatter(
        x=hist["ano"], y=hist["ch4"],
        mode="lines+markers", name="Série original",
    ))

    # Previsão tracejada
    fig.add_trace(go.Scatter(
        x=previsao["ano"], y=previsao["yhat"],
        mode="lines", name="Previsão",
        line=dict(dash="dash"),
    ))

    fig.update_layout(
        title=f"Previsão de CH4 para {cidade} ({estado})",
        xaxis_title="Ano",
        yaxis_title="CH4",
        legend_title="Legenda",
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="d")
    return fig


# ---------------------------------------------------------------------------
# PAINEL LATERAL: HIPERPARÂMETROS
# ---------------------------------------------------------------------------
def painel_hiperparametros() -> dict:
    """Desenha os controles na barra lateral e devolve os valores escolhidos."""
    st.sidebar.header("Hiperparâmetros do Prophet")

    changepoint = st.sidebar.slider(
        "changepoint_prior_scale", min_value=0.001, max_value=0.5,
        value=0.05, step=0.001, format="%.3f",
        help="Flexibilidade da tendência. Valores maiores permitem mais mudanças de direção.",
    )
    seasonality = st.sidebar.slider(
        "seasonality_prior_scale", min_value=0.01, max_value=10.0,
        value=10.0, step=0.01,
        help="Intensidade da sazonalidade. Valores menores a suavizam.",
    )
    modo = st.sidebar.selectbox(
        "seasonality_mode", ["additive", "multiplicative"],
        help="Aditiva: efeito constante. Multiplicativa: efeito proporcional ao nível.",
    )

    st.sidebar.markdown("**Sazonalidades**")
    # Com dados anuais, a sazonalidade anual não é identificável; por padrão fica desligada
    yearly = st.sidebar.checkbox(
        "yearly_seasonality", value=False,
        help="Normalmente deve ficar desligada com dados anuais.",
    )
    weekly = st.sidebar.checkbox("weekly_seasonality", value=False)
    daily = st.sidebar.checkbox("daily_seasonality", value=False)

    st.sidebar.caption("interval_width fixo em 0.95.")

    return {
        "changepoint_prior_scale": changepoint,
        "seasonality_prior_scale": seasonality,
        "seasonality_mode": modo,
        "yearly_seasonality": yearly,
        "weekly_seasonality": weekly,
        "daily_seasonality": daily,
    }


# ---------------------------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Previsão de CH4", layout="wide")
    st.title("Análise e previsão de emissões de metano (CH4)")

    # --- 1. Importação -----------------------------------------------------
    arquivo = st.file_uploader("Envie a planilha de emissões (.xlsx)", type=["xlsx"])
    if arquivo is None:
        st.info("Envie um arquivo .xlsx para começar.")
        st.stop()

    try:
        bruto = carregar_excel(arquivo.getvalue())
        st.subheader("Pré-visualização dos dados enviados")
        st.dataframe(bruto.head(10), use_container_width=True)

        longo = para_formato_longo(bruto)
        df, descartados = limpar_dados(longo)
    except ValueError as erro:
        st.error(f"Erro na estrutura do arquivo: {erro}")
        st.stop()
    except Exception as erro:  # arquivo ilegível, aba vazia etc.
        st.error(f"Não foi possível ler o arquivo: {erro}")
        st.stop()

    if descartados > 0:
        st.warning(f"{descartados} linha(s) com valores ausentes ou inválidos em 'ch4' "
                   f"(ou 'ano') foram descartadas.")
    else:
        st.success("Nenhum valor ausente encontrado.")

    if df.empty:
        st.error("Não restaram dados válidos após a limpeza.")
        st.stop()

    # --- 2. Filtros em cascata --------------------------------------------
    st.subheader("Seleção de localidade")
    col1, col2 = st.columns(2)

    with col1:
        estados = sorted(df["Estado"].unique())
        estado = st.selectbox("Estado", estados)

    # Cidades restritas ao estado escolhido. A chave do widget depende do
    # estado, então a seleção de cidade é reiniciada quando o estado muda.
    with col2:
        cidades = sorted(df.loc[df["Estado"] == estado, "Cidade"].unique())
        if not cidades:
            st.info("Não há cidades para este estado.")
            st.stop()
        cidade = st.selectbox("Cidade", cidades, key=f"cidade_{estado}")

    serie = filtrar_serie(df, estado, cidade)
    if serie.empty:
        st.info(f"Não há dados de CH4 para {cidade} ({estado}).")
        st.stop()

    # --- 3. Série histórica -----------------------------------------------
    st.plotly_chart(grafico_historico(serie, estado, cidade), use_container_width=True)

    # --- 4 e 5. Previsão com Prophet --------------------------------------
    st.subheader("Previsão com Prophet")
    params = painel_hiperparametros()
    executar = st.sidebar.button("Executar previsão", type="primary")

    base = preparar_prophet(serie)
    n_obs = len(base)

    if n_obs < MIN_OBSERVACOES:
        st.warning(
            f"{cidade} tem apenas {n_obs} observação(ões). O Prophet exige pelo menos "
            f"{MIN_OBSERVACOES} anos de dados para ajustar o modelo."
        )
    elif executar:
        # Ajuste e previsão só são feitos ao clicar no botão
        modelo = ajustar_prophet(base, params)
        previsao = prever(modelo)
        st.session_state["resultado"] = {
            "chave": (estado, cidade),
            "previsao": previsao,
            "hist": serie[["ano", "ch4"]],
            "params": params,
        }

    resultado = st.session_state.get("resultado")
    if n_obs >= MIN_OBSERVACOES:
        if resultado is None:
            st.info("Ajuste os hiperparâmetros na barra lateral e clique em "
                    "**Executar previsão**.")
        elif resultado["chave"] != (estado, cidade):
            st.info("Estado ou cidade mudou. Clique em **Executar previsão** "
                    "para gerar a previsão desta localidade.")
        else:
            st.plotly_chart(
                grafico_previsao(resultado["hist"], resultado["previsao"], estado, cidade),
                use_container_width=True,
            )
            p = resultado["params"]
            st.caption(
                f"Parâmetros usados: changepoint_prior_scale={p['changepoint_prior_scale']}, "
                f"seasonality_prior_scale={p['seasonality_prior_scale']}, "
                f"seasonality_mode={p['seasonality_mode']}, "
                f"yearly={p['yearly_seasonality']}, weekly={p['weekly_seasonality']}, "
                f"daily={p['daily_seasonality']}."
            )


if __name__ == "__main__":
    main()
