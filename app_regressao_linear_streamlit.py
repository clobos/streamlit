import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Regressão Linear Simples",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILO VISUAL
# ============================================================
st.markdown(
    '''
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            font-size: 1.05rem;
            opacity: 0.78;
            margin-bottom: 1rem;
        }
        .info-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 12px;
            padding: 0.70rem;
        }
        code {
            font-size: 0.90rem !important;
        }
    </style>
    ''',
    unsafe_allow_html=True,
)

# ============================================================
# FUNÇÕES
# ============================================================
def gerar_dados(n, intercepto, inclinacao, ruido, semente):
    rng = np.random.default_rng(semente)
    x = np.sort(rng.uniform(0, 10, n))
    erro = rng.normal(0, ruido, n)
    y = intercepto + inclinacao * x + erro
    return pd.DataFrame({"x": x, "y": y})


def ajustar_reta(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    beta1_hat = (
        np.sum((x - x.mean()) * (y - y.mean()))
        / np.sum((x - x.mean()) ** 2)
    )
    beta0_hat = y.mean() - beta1_hat * x.mean()

    y_hat = beta0_hat + beta1_hat * x

    sq_res = np.sum((y - y_hat) ** 2)
    sq_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sq_res / sq_tot

    return beta0_hat, beta1_hat, y_hat, r2


def correlacao_pearson(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    numerador = np.sum((x - x.mean()) * (y - y.mean()))
    denominador = np.sqrt(
        np.sum((x - x.mean()) ** 2)
        * np.sum((y - y.mean()) ** 2)
    )
    return numerador / denominador


# ============================================================
# CABEÇALHO
# ============================================================
st.markdown(
    '<div class="main-title">📈 Laboratório de Regressão Linear Simples</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Explore intercepto, coeficiente angular, mínimos quadrados, '
    'R² e correlação linear de Pearson.'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# CONTROLES
# ============================================================
with st.sidebar:
    st.header("⚙️ Parâmetros da simulação")

    intercepto = st.slider(
        "Intercepto β₀",
        -20.0, 20.0, 5.0, 0.5,
        help="Valor esperado de Y quando X = 0."
    )

    inclinacao = st.slider(
        "Coeficiente angular β₁",
        -10.0, 10.0, 2.0, 0.1,
        help="Mudança média em Y para cada aumento de uma unidade em X."
    )

    ruido = st.slider(
        "Desvio-padrão do erro σ",
        0.0, 15.0, 3.0, 0.5,
        help="Controla a dispersão dos pontos ao redor da relação linear."
    )

    n = st.slider(
        "Número de observações",
        10, 300, 60, 5
    )

    semente = st.number_input(
        "Semente aleatória",
        min_value=0,
        max_value=100000,
        value=42,
        step=1,
    )

    st.divider()

    mostrar_reta_teorica = st.checkbox(
        "Mostrar reta definida por β₀ e β₁",
        value=True
    )
    mostrar_reta_ajustada = st.checkbox(
        "Mostrar reta ajustada aos dados",
        value=True
    )
    mostrar_residuos = st.checkbox(
        "Mostrar resíduos",
        value=False
    )

# ============================================================
# DADOS E ESTIMATIVAS
# ============================================================
df = gerar_dados(
    n=n,
    intercepto=intercepto,
    inclinacao=inclinacao,
    ruido=ruido,
    semente=int(semente),
)

beta0_hat, beta1_hat, y_hat, r2 = ajustar_reta(df)
r = correlacao_pearson(df)

df["y_ajustado"] = y_hat
df["residuo"] = df["y"] - df["y_ajustado"]

# ============================================================
# ABAS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Simulador",
    "📘 Fórmulas e conceitos",
    "🔗 Correlação de Pearson",
    "💻 Código Python",
])

# ============================================================
# ABA 1: SIMULADOR
# ============================================================
with tab1:
    st.subheader("Gráfico de dispersão dinâmico")

    col_grafico, col_metricas = st.columns([3.2, 1.2], gap="large")

    with col_grafico:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                name="Dados simulados",
                marker=dict(size=9, opacity=0.78),
                hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>",
            )
        )

        x_linha = np.linspace(df["x"].min(), df["x"].max(), 250)

        if mostrar_reta_teorica:
            y_teorico = intercepto + inclinacao * x_linha
            fig.add_trace(
                go.Scatter(
                    x=x_linha,
                    y=y_teorico,
                    mode="lines",
                    name="Reta escolhida pelo usuário",
                    line=dict(width=3, dash="dash"),
                )
            )

        if mostrar_reta_ajustada:
            y_fit = beta0_hat + beta1_hat * x_linha
            fig.add_trace(
                go.Scatter(
                    x=x_linha,
                    y=y_fit,
                    mode="lines",
                    name="Reta ajustada por MQO",
                    line=dict(width=4),
                )
            )

        if mostrar_residuos and mostrar_reta_ajustada:
            for xi, yi, yhi in zip(df["x"], df["y"], df["y_ajustado"]):
                fig.add_trace(
                    go.Scatter(
                        x=[xi, xi],
                        y=[yhi, yi],
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(
            height=590,
            margin=dict(l=20, r=20, t=25, b=20),
            xaxis_title="Variável explicativa X",
            yaxis_title="Variável resposta Y",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_metricas:
        st.markdown("#### Resultados")

        st.metric("β₀ escolhido", f"{intercepto:.2f}")
        st.metric("β₁ escolhido", f"{inclinacao:.2f}")
        st.metric("β̂₀ estimado", f"{beta0_hat:.3f}")
        st.metric("β̂₁ estimado", f"{beta1_hat:.3f}")
        st.metric("R²", f"{r2:.3f}")
        st.metric("Pearson r", f"{r:.3f}")

        if r > 0:
            sentido = "positiva"
        elif r < 0:
            sentido = "negativa"
        else:
            sentido = "nula"

        intensidade = (
            "muito forte" if abs(r) >= 0.90 else
            "forte" if abs(r) >= 0.70 else
            "moderada" if abs(r) >= 0.50 else
            "fraca" if abs(r) >= 0.30 else
            "muito fraca"
        )

        st.info(
            f"Associação linear **{sentido}** e **{intensidade}** "
            f"nesta amostra simulada."
        )

    st.divider()
    st.markdown("### Equações atuais")

    eq1, eq2 = st.columns(2)

    with eq1:
        st.markdown("**Modelo usado para gerar os dados**")
        st.latex(
            rf"Y = {intercepto:.2f} + ({inclinacao:.2f})X + \varepsilon"
        )

    with eq2:
        st.markdown("**Reta estimada pelos dados**")
        st.latex(
            rf"\hat{{Y}} = {beta0_hat:.3f} + ({beta1_hat:.3f})X"
        )

    with st.expander("Ver dados simulados, valores ajustados e resíduos"):
        st.dataframe(
            df.round(4),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# ABA 2: FÓRMULAS
# ============================================================
with tab2:
    st.header("Regressão linear simples")

    st.markdown(
        "A regressão linear simples representa a relação entre uma "
        "variável explicativa **X** e uma variável resposta **Y** por meio "
        "de uma reta."
    )

    st.latex(r"Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            '''
            <div class="info-card">
            <b>β₀ — intercepto</b><br>
            Valor esperado de Y quando X = 0.
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            '''
            <div class="info-card">
            <b>β₁ — coeficiente angular</b><br>
            Mudança média esperada em Y quando X aumenta uma unidade.
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            '''
            <div class="info-card">
            <b>εᵢ — erro aleatório</b><br>
            Parte da observação Y não explicada pela relação linear com X.
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown("#### Estimativas de mínimos quadrados")

        st.markdown("Coeficiente angular:")
        st.latex(
            r"\hat{\beta}_1 = "
            r"\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}"
            r"{\sum_{i=1}^{n}(x_i-\bar{x})^2}"
        )

        st.markdown("Intercepto:")
        st.latex(
            r"\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}"
        )

        st.markdown("Valor ajustado:")
        st.latex(
            r"\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i"
        )

        st.markdown("Resíduo:")
        st.latex(
            r"e_i = y_i - \hat{y}_i"
        )

    st.markdown("### Critério dos mínimos quadrados")

    st.latex(
        r"\operatorname{SQRes} = "
        r"\sum_{i=1}^{n}(y_i-\hat{y}_i)^2"
    )

    st.markdown(
        "A reta de mínimos quadrados é aquela que minimiza a soma dos "
        "quadrados das diferenças verticais entre os pontos observados "
        "e os valores previstos pela reta."
    )

    st.markdown("### Coeficiente de determinação")

    st.latex(
        r"R^2 = 1 - "
        r"\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
        r"{\sum_{i=1}^{n}(y_i-\bar{y})^2}"
    )

    st.markdown(
        "Em regressão linear simples com intercepto, **R²** representa a "
        "proporção da variabilidade observada de Y explicada pela relação "
        "linear com X."
    )

    st.markdown("### Código Python das estimativas")

    st.code(
        '''x = df["x"].to_numpy()
y = df["y"].to_numpy()

beta1_hat = (
    np.sum((x - x.mean()) * (y - y.mean()))
    / np.sum((x - x.mean()) ** 2)
)

beta0_hat = y.mean() - beta1_hat * x.mean()

y_hat = beta0_hat + beta1_hat * x
residuos = y - y_hat
''',
        language="python"
    )

# ============================================================
# ABA 3: PEARSON
# ============================================================
with tab3:
    st.header("Correlação linear de Pearson")

    st.markdown(
        "O coeficiente de correlação linear de Pearson mede a **direção** "
        "e a **intensidade da associação linear** entre duas variáveis "
        "quantitativas. Seus valores variam de -1 a +1."
    )

    st.latex(
        r"r = "
        r"\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}"
        r"{\sqrt{\left[\sum_{i=1}^{n}(x_i-\bar{x})^2\right]"
        r"\left[\sum_{i=1}^{n}(y_i-\bar{y})^2\right]}}"
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("r = -1", "Negativa perfeita")
    p2.metric("r = 0", "Sem associação linear")
    p3.metric("r = +1", "Positiva perfeita")

    st.markdown("### Resultado nos dados simulados")
    st.metric("Coeficiente de Pearson", f"{r:.4f}")

    st.progress(
        min(abs(r), 1.0),
        text=f"Intensidade |r| = {abs(r):.4f}"
    )

    st.warning(
        "Correlação não implica causalidade. Além disso, r próximo de zero "
        "não prova ausência de relação: pode existir uma relação não linear."
    )

    st.markdown("### Código Python — usando a própria fórmula")

    st.code(
        '''x = df["x"].to_numpy()
y = df["y"].to_numpy()

numerador = np.sum(
    (x - x.mean()) * (y - y.mean())
)

denominador = np.sqrt(
    np.sum((x - x.mean()) ** 2)
    * np.sum((y - y.mean()) ** 2)
)

r = numerador / denominador
print(r)
''',
        language="python"
    )

    st.markdown("### Código Python — usando NumPy")

    st.code(
        '''r_numpy = np.corrcoef(df["x"], df["y"])[0, 1]
print(r_numpy)
''',
        language="python"
    )

    st.markdown("### Relação entre Pearson e R²")

    st.latex(r"R^2 = r^2")

    st.write(
        f"Na simulação atual: **r = {r:.4f}**, "
        f"**r² = {r**2:.4f}** e **R² = {r2:.4f}**."
    )

# ============================================================
# ABA 4: CÓDIGO
# ============================================================
with tab4:
    st.header("Código Python do núcleo do aplicativo")

    st.markdown(
        "Este é o núcleo computacional usado para gerar os dados, "
        "ajustar a regressão e calcular a correlação."
    )

    st.code(
        '''import numpy as np
import pandas as pd

# 1. Simulação
rng = np.random.default_rng(42)

n = 60
beta0 = 5
beta1 = 2
sigma = 3

x = np.sort(rng.uniform(0, 10, n))
erro = rng.normal(0, sigma, n)
y = beta0 + beta1 * x + erro

df = pd.DataFrame({"x": x, "y": y})

# 2. Regressão linear por mínimos quadrados
beta1_hat = (
    np.sum((x - x.mean()) * (y - y.mean()))
    / np.sum((x - x.mean()) ** 2)
)

beta0_hat = y.mean() - beta1_hat * x.mean()

y_hat = beta0_hat + beta1_hat * x

# 3. Coeficiente de determinação
sq_res = np.sum((y - y_hat) ** 2)
sq_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - sq_res / sq_tot

# 4. Correlação de Pearson
r = (
    np.sum((x - x.mean()) * (y - y.mean()))
    / np.sqrt(
        np.sum((x - x.mean()) ** 2)
        * np.sum((y - y.mean()) ** 2)
    )
)

print("Intercepto estimado:", beta0_hat)
print("Coeficiente angular estimado:", beta1_hat)
print("R²:", r2)
print("Pearson r:", r)
''',
        language="python"
    )

    st.markdown("### Dependências")

    st.code(
        '''streamlit
numpy
pandas
plotly
''',
        language="text"
    )

    st.markdown(
        "Salve o arquivo como `app.py` e execute no terminal: "
        "`streamlit run app.py`."
    )

st.caption(
    "Aplicativo didático para ensino de regressão linear simples "
    "e correlação de Pearson."
)
