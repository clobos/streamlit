"""
App Streamlit: comparação de previsão de CH4 entre Prophet e LSTM.

Fluxo:
1. Upload de planilha .xlsx (formato longo ou largo) e limpeza dos dados.
2. Filtros em cascata: Estado -> Cidade.
3. Gráfico da série histórica.
4. Avaliação em holdout: os últimos `h` anos são reservados como teste.
   Os dois modelos são treinados só com o restante e comparados por RMSE e MAE,
   junto com um benchmark ingênuo (repetir o último valor observado).
5. Previsão de 2 anos com os dois modelos, treinados com todos os dados.

Para executar:
    pip install -r requirements_comparacao.txt
    streamlit run app_comparacao.py
"""

import io
import logging
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Silencia logs do cmdstanpy (Prophet) e do TensorFlow no terminal
logging.getLogger("cmdstanpy").disabled = True
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402  (importado após configurar o ambiente)

# Constantes
HORIZONTE = 2        # anos à frente na previsão final
FREQ = "YS"          # frequência anual (início de ano)
MIN_OBS_PROPHET = 3  # mínimo de observações para o Prophet
SEED = 42            # semente para reprodutibilidade


# ---------------------------------------------------------------------------
# 1. IMPORTAÇÃO E LIMPEZA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lendo a planilha...")
def carregar_excel(conteudo: bytes) -> pd.DataFrame:
    """Lê a primeira aba do .xlsx com engine openpyxl (cacheado pelos bytes do arquivo)."""
    return pd.read_excel(io.BytesIO(conteudo), sheet_name=0, engine="openpyxl")


def encontrar_coluna(df: pd.DataFrame, nome: str):
    """Retorna o nome real da coluna cujo texto (sem espaços, minúsculo) é `nome`."""
    for c in df.columns:
        if str(c).strip().lower() == nome:
            return c
    return None


def para_formato_longo(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas Estado, Cidade, ano e ch4. Faz melt se o arquivo for largo."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_estado = encontrar_coluna(df, "estado")
    col_cidade = encontrar_coluna(df, "cidade")
    if col_estado is None or col_cidade is None:
        raise ValueError("A planilha precisa ter as colunas 'Estado' e 'Cidade'.")

    col_ano = encontrar_coluna(df, "ano")
    col_ch4 = encontrar_coluna(df, "ch4")

    # Formato longo: já tem 'ano' e 'ch4'
    if col_ano is not None and col_ch4 is not None:
        longo = df[[col_estado, col_cidade, col_ano, col_ch4]].copy()
        longo.columns = ["Estado", "Cidade", "ano", "ch4"]
        return longo

    # Formato largo: colunas cujo nome é um ano (2000, "2000", "2000.0")
    anos = [c for c in df.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))]
    if not anos:
        raise ValueError(
            "Formato não reconhecido. Use colunas 'Estado', 'Cidade', 'ano' e 'ch4', "
            "ou uma coluna por ano."
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
    """Converte tipos e remove linhas com valores ausentes. Retorna (df, n_descartados)."""
    total_antes = len(df)
    df = df.copy()
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["ch4"] = pd.to_numeric(df["ch4"], errors="coerce")
    df = df.dropna(subset=["Estado", "Cidade", "ano", "ch4"])
    df["ano"] = df["ano"].astype(int)
    df["Estado"] = df["Estado"].astype(str).str.strip()
    df["Cidade"] = df["Cidade"].astype(str).str.strip()
    return df.reset_index(drop=True), total_antes - len(df)


def filtrar_serie(df: pd.DataFrame, estado: str, cidade: str) -> pd.DataFrame:
    """Série anual de uma cidade, ordenada por ano."""
    serie = df[(df["Estado"] == estado) & (df["Cidade"] == cidade)]
    return serie.sort_values("ano").reset_index(drop=True)


def preparar_prophet(serie: pd.DataFrame) -> pd.DataFrame:
    """Formato do Prophet: ds = 1º de janeiro de cada ano, y = ch4 (média se houver duplicatas)."""
    base = serie.groupby("ano", as_index=False)["ch4"].mean()
    base["ds"] = pd.to_datetime(base["ano"].astype(str) + "-01-01")
    base = base.rename(columns={"ch4": "y"})
    return base[["ds", "y"]].sort_values("ds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. PROPHET
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ajustar_prophet(dados: pd.DataFrame, params: dict) -> Prophet:
    """Ajusta o Prophet. Cacheado: só reajusta se dados ou hiperparâmetros mudarem."""
    modelo = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
        interval_width=0.95,
    )
    modelo.fit(dados)
    return modelo


def previsao_prophet(modelo: Prophet, periodos: int) -> pd.DataFrame:
    """Retorna apenas as linhas futuras (ds, yhat, yhat_lower, yhat_upper)."""
    futuro = modelo.make_future_dataframe(periods=periodos, freq=FREQ)
    return modelo.predict(futuro).tail(periodos).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. LSTM
# ---------------------------------------------------------------------------
def criar_janelas(valores: np.ndarray, lags: int):
    """Transforma a série em pares (janela de `lags` valores -> próximo valor).

    Com n pequeno, o número de janelas é n - lags: é isso que limita o treino.
    """
    X = np.array([valores[i:i + lags] for i in range(len(valores) - lags)])
    y = valores[lags:]
    return X[..., np.newaxis], y  # formato (amostras, passos, variáveis)


@st.cache_resource(show_spinner=False)
def treinar_lstm(valores: np.ndarray, lags: int, unidades: int, epocas: int) -> dict:
    """Treina uma LSTM univariada com normalização Min-Max ajustada só no treino.

    Retorna um dicionário com modelo, escalador e número de lags.
    """
    tf.keras.utils.set_random_seed(SEED)

    scaler = MinMaxScaler(feature_range=(0, 1))
    escalonado = scaler.fit_transform(valores.reshape(-1, 1)).ravel()
    X, y = criar_janelas(escalonado, lags)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lags, 1)),
        tf.keras.layers.LSTM(unidades),
        tf.keras.layers.Dense(1),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    modelo.fit(X, y, epochs=epocas, batch_size=4, shuffle=False, verbose=0)

    return {"modelo": modelo, "scaler": scaler, "lags": lags}


def previsao_lstm(bundle: dict, valores: np.ndarray, passos: int) -> np.ndarray:
    """Previsão recursiva: cada passo usa a própria previsão anterior como entrada."""
    lags = bundle["lags"]
    escalonado = bundle["scaler"].transform(valores.reshape(-1, 1)).ravel()
    janela = list(escalonado[-lags:])

    previsoes = []
    for _ in range(passos):
        x = np.array(janela[-lags:]).reshape(1, lags, 1)
        p = float(bundle["modelo"].predict(x, verbose=0)[0, 0])
        previsoes.append(p)
        janela.append(p)

    return bundle["scaler"].inverse_transform(np.array(previsoes).reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# 4. AVALIAÇÃO EM HOLDOUT E PREVISÃO FINAL
# ---------------------------------------------------------------------------
def metricas(real: np.ndarray, previsto: np.ndarray) -> dict:
    """RMSE e MAE na escala original."""
    erro = real - previsto
    return {
        "RMSE": float(np.sqrt(np.mean(erro ** 2))),
        "MAE": float(np.mean(np.abs(erro))),
    }


def avaliar_holdout(base: pd.DataFrame, h: int, params_p: dict, params_l: dict):
    """Treina com todos os anos exceto os últimos `h` e compara com o observado.

    Inclui benchmark ingênuo (repete o último valor do treino), que é referência
    mínima: um modelo que não o supera não agrega informação.
    """
    treino = base.iloc[:-h]
    teste = base.iloc[-h:]
    valores_treino = treino["y"].to_numpy()
    real = teste["y"].to_numpy()

    # Prophet
    m_p = ajustar_prophet(treino, params_p)
    prev_p = previsao_prophet(m_p, h)["yhat"].to_numpy()

    # LSTM
    bundle = treinar_lstm(valores_treino, params_l["lags"],
                          params_l["unidades"], params_l["epocas"])
    prev_l = previsao_lstm(bundle, valores_treino, h)

    # Ingênuo
    prev_ing = np.repeat(valores_treino[-1], h)

    tabela = pd.DataFrame([
        {"Modelo": "Prophet", **metricas(real, prev_p)},
        {"Modelo": "LSTM", **metricas(real, prev_l)},
        {"Modelo": "Ingênuo (último valor)", **metricas(real, prev_ing)},
    ])

    backtest = pd.DataFrame({
        "ano": teste["ds"].dt.year.to_numpy(),
        "Observado": real,
        "Prophet": prev_p,
        "LSTM": prev_l,
        "Ingênuo": prev_ing,
    })
    return tabela, backtest


def previsao_final(base: pd.DataFrame, params_p: dict, params_l: dict):
    """Treina com todos os dados e prevê HORIZONTE anos à frente."""
    m_p = ajustar_prophet(base, params_p)
    fc_p = previsao_prophet(m_p, HORIZONTE)

    valores = base["y"].to_numpy()
    bundle = treinar_lstm(valores, params_l["lags"],
                          params_l["unidades"], params_l["epocas"])
    prev_l = previsao_lstm(bundle, valores, HORIZONTE)

    ultimo_ano = int(base["ds"].dt.year.max())
    anos_l = np.arange(ultimo_ano + 1, ultimo_ano + 1 + HORIZONTE)
    return fc_p, anos_l, prev_l


# ---------------------------------------------------------------------------
# 5. GRÁFICOS
# ---------------------------------------------------------------------------
def grafico_historico(serie: pd.DataFrame, estado: str, cidade: str):
    fig = px.line(serie, x="ano", y="ch4", markers=True,
                  title=f"Emissões de CH4 em {cidade} ({estado})",
                  labels={"ano": "Ano", "ch4": "CH4"})
    fig.update_xaxes(tickformat="d")
    return fig


def grafico_comparacao(base: pd.DataFrame, fc_p: pd.DataFrame, anos_l, prev_l,
                       estado: str, cidade: str):
    """Série original, previsão do Prophet com IC 95% e previsão da LSTM."""
    anos_p = fc_p["ds"].dt.year

    fig = go.Figure()
    # Banda do IC 95% do Prophet (limite superior, depois inferior com preenchimento)
    fig.add_trace(go.Scatter(x=anos_p, y=fc_p["yhat_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anos_p, y=fc_p["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31, 119, 180, 0.2)", name="IC 95% (Prophet)"))
    # Série original
    fig.add_trace(go.Scatter(x=base["ds"].dt.year, y=base["y"], mode="lines+markers",
                             name="Série original"))
    # Previsão Prophet
    fig.add_trace(go.Scatter(x=anos_p, y=fc_p["yhat"], mode="lines", name="Prophet",
                             line=dict(dash="dash")))
    # Previsão LSTM (sem intervalo de confiança)
    fig.add_trace(go.Scatter(x=anos_l, y=prev_l, mode="lines+markers", name="LSTM",
                             line=dict(dash="dot")))

    fig.update_layout(title=f"Previsão de CH4 para {cidade} ({estado})",
                      xaxis_title="Ano", yaxis_title="CH4",
                      hovermode="x unified", legend_title="Legenda")
    fig.update_xaxes(tickformat="d")
    return fig


# ---------------------------------------------------------------------------
# PAINEL LATERAL
# ---------------------------------------------------------------------------
def painel_prophet() -> dict:
    st.sidebar.header("Prophet")
    cps = st.sidebar.slider("changepoint_prior_scale", 0.001, 0.5, 0.05, 0.001, format="%.3f")
    sps = st.sidebar.slider("seasonality_prior_scale", 0.01, 10.0, 10.0, 0.01)
    modo = st.sidebar.selectbox("seasonality_mode", ["additive", "multiplicative"])
    yearly = st.sidebar.checkbox("yearly_seasonality", value=False,
                                 help="Normalmente desligada com dados anuais.")
    weekly = st.sidebar.checkbox("weekly_seasonality", value=False)
    daily = st.sidebar.checkbox("daily_seasonality", value=False)
    return {
        "changepoint_prior_scale": cps,
        "seasonality_prior_scale": sps,
        "seasonality_mode": modo,
        "yearly_seasonality": yearly,
        "weekly_seasonality": weekly,
        "daily_seasonality": daily,
    }


def painel_lstm() -> dict:
    st.sidebar.header("LSTM")
    lags = st.sidebar.slider("Lags (janela de entrada)", 1, 5, 3,
                             help="Quantos anos anteriores são usados para prever o próximo.")
    unidades = st.sidebar.slider("Unidades da LSTM", 8, 64, 16, step=8)
    epocas = st.sidebar.slider("Épocas", 50, 500, 200, step=50)
    return {"lags": lags, "unidades": unidades, "epocas": epocas}


# ---------------------------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Prophet x LSTM - CH4", layout="wide")
    st.title("Comparação de previsão de emissões de metano (CH4): Prophet x LSTM")

    # --- Importação --------------------------------------------------------
    arquivo = st.file_uploader("Envie a planilha de emissões (.xlsx)", type=["xlsx"])
    if arquivo is None:
        st.info("Envie um arquivo .xlsx para começar.")
        st.stop()

    try:
        bruto = carregar_excel(arquivo.getvalue())
        st.subheader("Pré-visualização dos dados enviados")
        st.dataframe(bruto.head(10), use_container_width=True)
        df, descartados = limpar_dados(para_formato_longo(bruto))
    except ValueError as erro:
        st.error(f"Erro na estrutura do arquivo: {erro}")
        st.stop()
    except Exception as erro:
        st.error(f"Não foi possível ler o arquivo: {erro}")
        st.stop()

    if descartados > 0:
        st.warning(f"{descartados} linha(s) com valores ausentes ou inválidos foram descartadas.")
    else:
        st.success("Nenhum valor ausente encontrado.")
    if df.empty:
        st.error("Não restaram dados válidos após a limpeza.")
        st.stop()

    # --- Filtros em cascata ------------------------------------------------
    st.subheader("Seleção de localidade")
    c1, c2 = st.columns(2)
    with c1:
        estado = st.selectbox("Estado", sorted(df["Estado"].unique()))
    with c2:
        cidades = sorted(df.loc[df["Estado"] == estado, "Cidade"].unique())
        if not cidades:
            st.info("Não há cidades para este estado.")
            st.stop()
        cidade = st.selectbox("Cidade", cidades, key=f"cidade_{estado}")

    serie = filtrar_serie(df, estado, cidade)
    if serie.empty:
        st.info(f"Não há dados de CH4 para {cidade} ({estado}).")
        st.stop()

    st.plotly_chart(grafico_historico(serie, estado, cidade), use_container_width=True)

    # --- Controles ---------------------------------------------------------
    params_p = painel_prophet()
    params_l = painel_lstm()
    st.sidebar.header("Avaliação")
    h = st.sidebar.slider("Anos reservados para teste (holdout)", 2, 6, 4)
    executar = st.sidebar.button("Executar comparação", type="primary")

    base = preparar_prophet(serie)
    n = len(base)
    n_treino = n - h
    minimo_lstm = params_l["lags"] + 5  # janelas mínimas para o treino da LSTM

    st.subheader("Comparação dos modelos")
    st.caption(f"Observações anuais disponíveis: {n}. Treino na avaliação: {n_treino} anos. "
               f"Teste: últimos {h} anos.")

    if n < MIN_OBS_PROPHET + 1 or n_treino < max(MIN_OBS_PROPHET, minimo_lstm):
        st.warning(
            "Dados insuficientes para esta combinação de teste e lags. Reduza o holdout "
            "ou o número de lags. O Prophet exige pelo menos 3 anos de treino e a LSTM "
            f"cerca de {minimo_lstm} anos (lags + 5)."
        )
        st.stop()

    if executar:
        with st.spinner("Treinando e avaliando os modelos. A LSTM pode levar alguns segundos..."):
            tabela, backtest = avaliar_holdout(base, h, params_p, params_l)
            fc_p, anos_l, prev_l = previsao_final(base, params_p, params_l)
        st.session_state["resultado"] = {
            "chave": (estado, cidade),
            "tabela": tabela,
            "backtest": backtest,
            "fc_p": fc_p,
            "anos_l": anos_l,
            "prev_l": prev_l,
        }

    resultado = st.session_state.get("resultado")
    if resultado is None:
        st.info("Ajuste os controles na barra lateral e clique em **Executar comparação**.")
        return
    if resultado["chave"] != (estado, cidade):
        st.info("Estado ou cidade mudou. Clique em **Executar comparação** para atualizar.")
        return

    # --- Resultados --------------------------------------------------------
    st.markdown("**Desempenho no holdout** (menor é melhor)")
    st.dataframe(resultado["tabela"].round(3), use_container_width=True, hide_index=True)

    with st.expander("Valores observados x previstos no holdout"):
        st.dataframe(resultado["backtest"].round(3), use_container_width=True, hide_index=True)

    st.plotly_chart(
        grafico_comparacao(base, resultado["fc_p"], resultado["anos_l"],
                           resultado["prev_l"], estado, cidade),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
