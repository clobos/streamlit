"""
App Streamlit: previsão de CH4 com LSTM (somente LSTM).

Fluxo:
1. Upload de planilha .xlsx (formato longo ou largo) e limpeza dos dados.
2. Filtros em cascata: Estado -> Cidade.
3. Gráfico da série histórica.
4. Avaliação em holdout: os últimos `h` anos são reservados como teste.
   O modelo é treinado só com o restante e comparado ao observado (RMSE e MAE).
   O benchmark ingênuo (repetir o último valor do treino) serve de referência.
5. Previsão de 2 anos, com o modelo treinado em todos os dados.

Para executar localmente:
    pip install -r requirements_lstm.txt
    streamlit run app_lstm.py
"""

import io
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Reduz logs do TensorFlow no terminal
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402  (importado após configurar o ambiente)

# Constantes
HORIZONTE = 2   # anos à frente na previsão final
SEED = 42       # semente para reprodutibilidade


# ---------------------------------------------------------------------------
# 1. IMPORTAÇÃO E LIMPEZA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lendo a planilha...")
def carregar_excel(conteudo: bytes) -> pd.DataFrame:
    """Lê a primeira aba do .xlsx (cacheado pelos bytes do arquivo)."""
    return pd.read_excel(io.BytesIO(conteudo), sheet_name=0, engine="openpyxl")


def encontrar_coluna(df: pd.DataFrame, nome: str):
    """Retorna a coluna cujo texto (sem espaços, minúsculo) é `nome`."""
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

    # Formato longo
    if col_ano is not None and col_ch4 is not None:
        longo = df[[col_estado, col_cidade, col_ano, col_ch4]].copy()
        longo.columns = ["Estado", "Cidade", "ano", "ch4"]
        return longo

    # Formato largo: colunas cujo nome é um ano
    anos = [c for c in df.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))]
    if not anos:
        raise ValueError(
            "Formato não reconhecido. Use colunas 'Estado', 'Cidade', 'ano' e 'ch4', "
            "ou uma coluna por ano."
        )
    largo = df.melt(id_vars=[col_estado, col_cidade], value_vars=anos,
                    var_name="ano", value_name="ch4")
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


# ---------------------------------------------------------------------------
# 2. LSTM
# ---------------------------------------------------------------------------
def criar_janelas(valores: np.ndarray, lags: int):
    """Pares (janela de `lags` valores -> próximo valor), no formato (amostras, passos, 1)."""
    X = np.array([valores[i:i + lags] for i in range(len(valores) - lags)])
    y = valores[lags:]
    return X[..., np.newaxis], y


@st.cache_resource(show_spinner=False)
def treinar_lstm(valores: np.ndarray, lags: int, unidades: int, epocas: int) -> dict:
    """Treina uma LSTM univariada. A normalização Min-Max é ajustada só nos dados de treino."""
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
    """Previsão recursiva: cada passo usa a previsão anterior como entrada."""
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


def metricas(real: np.ndarray, previsto: np.ndarray) -> dict:
    """RMSE e MAE na escala original."""
    erro = real - previsto
    return {"RMSE": float(np.sqrt(np.mean(erro ** 2))),
            "MAE": float(np.mean(np.abs(erro)))}


# ---------------------------------------------------------------------------
# 3. AVALIAÇÃO E PREVISÃO
# ---------------------------------------------------------------------------
def avaliar_holdout(valores: np.ndarray, h: int, params: dict):
    """Treina com todos os anos exceto os últimos `h` e avalia no trecho reservado."""
    treino, real = valores[:-h], valores[-h:]

    bundle = treinar_lstm(treino, params["lags"], params["unidades"], params["epocas"])
    prev_lstm = previsao_lstm(bundle, treino, h)
    prev_ing = np.repeat(treino[-1], h)  # referência mínima

    tabela = pd.DataFrame([
        {"Modelo": "LSTM", **metricas(real, prev_lstm)},
        {"Modelo": "Referência: último valor", **metricas(real, prev_ing)},
    ])
    return tabela, prev_lstm, prev_ing, real


def previsao_final(valores: np.ndarray, ultimo_ano: int, params: dict):
    """Treina com todos os dados e prevê HORIZONTE anos à frente."""
    bundle = treinar_lstm(valores, params["lags"], params["unidades"], params["epocas"])
    prev = previsao_lstm(bundle, valores, HORIZONTE)
    anos_futuros = np.arange(ultimo_ano + 1, ultimo_ano + 1 + HORIZONTE)
    return anos_futuros, prev


# ---------------------------------------------------------------------------
# 4. GRÁFICOS
# ---------------------------------------------------------------------------
def grafico_historico(serie: pd.DataFrame, estado: str, cidade: str):
    fig = px.line(serie, x="ano", y="ch4", markers=True,
                  title=f"Emissões de CH4 em {cidade} ({estado})",
                  labels={"ano": "Ano", "ch4": "CH4"})
    fig.update_xaxes(tickformat="d")
    return fig


def grafico_previsao(serie: pd.DataFrame, anos_futuros, prev, estado: str, cidade: str):
    """Série original e previsão da LSTM, com a previsão ligada ao último ponto observado."""
    ultimo_ano = serie["ano"].iloc[-1]
    ultimo_valor = serie["ch4"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie["ano"], y=serie["ch4"], mode="lines+markers",
                             name="Série original"))
    # Conecta a previsão ao último valor observado para facilitar a leitura
    fig.add_trace(go.Scatter(
        x=np.concatenate([[ultimo_ano], anos_futuros]),
        y=np.concatenate([[ultimo_valor], prev]),
        mode="lines+markers", name="Previsão LSTM",
        line=dict(dash="dot"),
    ))
    fig.update_layout(title=f"Previsão de CH4 com LSTM para {cidade} ({estado})",
                      xaxis_title="Ano", yaxis_title="CH4",
                      hovermode="x unified", legend_title="Legenda")
    fig.update_xaxes(tickformat="d")
    return fig


# ---------------------------------------------------------------------------
# 5. PAINEL LATERAL
# ---------------------------------------------------------------------------
def painel_lstm() -> dict:
    st.sidebar.header("Hiperparâmetros da LSTM")
    lags = st.sidebar.slider("Lags (janela de entrada)", 1, 5, 3,
                             help="Quantos anos anteriores são usados para prever o próximo.")
    unidades = st.sidebar.slider("Unidades da LSTM", 8, 64, 16, step=8)
    epocas = st.sidebar.slider("Épocas", 50, 500, 200, step=50)
    return {"lags": lags, "unidades": unidades, "epocas": epocas}


# ---------------------------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="LSTM - CH4", layout="wide")
    st.title("Previsão de emissões de metano (CH4) com LSTM")

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
    params = painel_lstm()
    st.sidebar.header("Avaliação")
    h = st.sidebar.slider("Anos reservados para teste (holdout)", 2, 6, 4)
    executar = st.sidebar.button("Executar LSTM", type="primary")

    valores = serie["ch4"].to_numpy(dtype=float)
    n = len(valores)
    n_treino = n - h
    minimo = params["lags"] + 5  # janelas mínimas para treinar

    st.subheader("Modelo LSTM")
    st.caption(f"Observações anuais: {n}. Treino na avaliação: {n_treino} anos. "
               f"Teste: últimos {h} anos.")

    if n_treino < minimo:
        st.warning(
            f"Dados insuficientes para esta combinação. A LSTM precisa de pelo menos "
            f"{minimo} anos de treino (lags + 5). Reduza o holdout ou os lags."
        )
        st.stop()

    if executar:
        with st.spinner("Treinando a LSTM. Isso pode levar alguns segundos..."):
            tabela, prev_lstm, prev_ing, real = avaliar_holdout(valores, h, params)
            anos_futuros, prev_futuro = previsao_final(valores, int(serie["ano"].iloc[-1]), params)
        st.session_state["resultado"] = {
            "chave": (estado, cidade),
            "tabela": tabela,
            "backtest": pd.DataFrame({
                "ano": serie["ano"].iloc[-h:].to_numpy(),
                "Observado": real,
                "LSTM": prev_lstm,
                "Referência": prev_ing,
            }),
            "anos_futuros": anos_futuros,
            "prev_futuro": prev_futuro,
        }

    resultado = st.session_state.get("resultado")
    if resultado is None:
        st.info("Ajuste os hiperparâmetros na barra lateral e clique em **Executar LSTM**.")
        return
    if resultado["chave"] != (estado, cidade):
        st.info("Estado ou cidade mudou. Clique em **Executar LSTM** para atualizar.")
        return

    # --- Resultados --------------------------------------------------------
    st.markdown("**Desempenho no holdout** (menor é melhor)")
    st.dataframe(resultado["tabela"].round(3), use_container_width=True, hide_index=True)

    with st.expander("Valores observados x previstos no holdout"):
        st.dataframe(resultado["backtest"].round(3), use_container_width=True, hide_index=True)

    st.plotly_chart(
        grafico_previsao(serie, resultado["anos_futuros"], resultado["prev_futuro"],
                         estado, cidade),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
