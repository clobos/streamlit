import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, t, f, chi2

st.set_page_config(
    page_title="Explorador de Distribuições de Probabilidade",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Explorador de Distribuições de Probabilidade")
st.markdown(
    "Use os controles para observar, em tempo real, como os parâmetros alteram "
    "a **função densidade de probabilidade (PDF)**, a **função distribuição acumulada (CDF)** "
    "e o **percentil** escolhido."
)

# -----------------------------
# Helpers
# -----------------------------
def safe_interval(rv, q_low=0.001, q_high=0.999, fallback=(-10.0, 10.0)):
    """Obtém uma faixa de x robusta a partir de quantis."""
    try:
        lo = float(rv.ppf(q_low))
        hi = float(rv.ppf(q_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            return fallback
        return lo, hi
    except Exception:
        return fallback


def summary_stats(rv):
    mean, var, skew, kurt = rv.stats(moments="mvsk")
    return {
        "Média": float(mean) if np.isfinite(mean) else np.nan,
        "Variância": float(var) if np.isfinite(var) else np.nan,
        "Desvio-padrão": float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan,
        "Assimetria": float(skew) if np.isfinite(skew) else np.nan,
        "Curtose (excesso)": float(kurt) if np.isfinite(kurt) else np.nan,
    }


def fmt(v, digits=4):
    if v is None or not np.isfinite(v):
        return "não definida"
    return f"{v:.{digits}f}"


def plot_pdf(x, y, xp, yp, percentile, title, reference=None):
    fig = go.Figure()

    if reference is not None:
        xr, yr, label = reference
        fig.add_trace(
            go.Scatter(
                x=xr,
                y=yr,
                mode="lines",
                name=label,
                line=dict(dash="dash", width=2),
                opacity=0.65,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="PDF atual",
            line=dict(width=3),
        )
    )

    # Área acumulada até o percentil
    mask = x <= xp
    if np.any(mask):
        x_fill = np.concatenate(([x[0]], x[mask], [xp]))
        y_fill = np.concatenate(([0.0], y[mask], [0.0]))
        fig.add_trace(
            go.Scatter(
                x=x_fill,
                y=y_fill,
                fill="toself",
                mode="lines",
                line=dict(width=0),
                name=f"Área = {percentile/100:.3f}",
                opacity=0.22,
            )
        )

    fig.add_vline(
        x=xp,
        line_dash="dot",
        annotation_text=f"P{percentile:g} = {xp:.4f}",
        annotation_position="top",
    )
    fig.add_trace(
        go.Scatter(
            x=[xp],
            y=[yp],
            mode="markers",
            name="Percentil",
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        hovermode="x unified",
        legend_title="Curvas",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def plot_cdf(x, y, xp, percentile, title, reference=None):
    fig = go.Figure()

    if reference is not None:
        xr, yr, label = reference
        fig.add_trace(
            go.Scatter(
                x=xr,
                y=yr,
                mode="lines",
                name=label,
                line=dict(dash="dash", width=2),
                opacity=0.65,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="CDF atual",
            line=dict(width=3),
        )
    )
    fig.add_vline(x=xp, line_dash="dot")
    fig.add_hline(y=percentile / 100, line_dash="dot")
    fig.add_trace(
        go.Scatter(
            x=[xp],
            y=[percentile / 100],
            mode="markers",
            name="Percentil",
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="F(x) = P(X ≤ x)",
        yaxis=dict(range=[0, 1.02], tickformat=".0%"),
        hovermode="x unified",
        legend_title="Curvas",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Configurações")
distribution = st.sidebar.selectbox(
    "Distribuição",
    ["Normal", "t de Student", "F de Fisher-Snedecor", "Qui-quadrado"],
)

percentile = st.sidebar.slider(
    "Percentil desejado (%)",
    min_value=0.1,
    max_value=99.9,
    value=95.0,
    step=0.1,
    help="Ex.: 95 significa encontrar x tal que P(X ≤ x) = 0,95.",
)
q = percentile / 100.0

show_reference = st.sidebar.checkbox(
    "Comparar com uma curva de referência",
    value=True,
    help="Ajuda a visualizar o efeito das mudanças de parâmetros.",
)

# -----------------------------
# Distribution definitions
# -----------------------------
reference_rv = None
reference_label = None
formula = ""
parameter_explanation = ""
concept_note = ""

if distribution == "Normal":
    st.sidebar.subheader("Parâmetros da Normal")
    mu = st.sidebar.slider("Média μ", -10.0, 10.0, 0.0, 0.1)
    sigma = st.sidebar.slider("Desvio-padrão σ", 0.1, 10.0, 1.0, 0.1)

    rv = norm(loc=mu, scale=sigma)
    reference_rv = norm(loc=0, scale=1)
    reference_label = "Referência: N(0, 1)"
    formula = r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]"
    parameter_explanation = (
        f"**μ = {mu:.2f}** controla a posição da curva: aumentar μ desloca toda a distribuição para a direita, "
        f"sem alterar sua forma. **σ = {sigma:.2f}** controla a dispersão: aumentar σ deixa a curva mais larga e baixa; "
        "diminuir σ a torna mais estreita e alta."
    )
    concept_note = "A distribuição Normal é simétrica. Média, mediana e moda coincidem em μ."
    fallback = (mu - 5 * sigma, mu + 5 * sigma)

elif distribution == "t de Student":
    st.sidebar.subheader("Parâmetros da t de Student")
    df = st.sidebar.slider("Graus de liberdade ν", 1, 100, 10, 1)
    loc_t = st.sidebar.slider("Localização μ", -10.0, 10.0, 0.0, 0.1)
    scale_t = st.sidebar.slider("Escala s", 0.1, 10.0, 1.0, 0.1)

    rv = t(df=df, loc=loc_t, scale=scale_t)
    reference_rv = t(df=10, loc=0, scale=1)
    reference_label = "Referência: t(ν=10, μ=0, s=1)"
    formula = r"f(x)=\frac{1}{s}\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{1}{\nu}\left(\frac{x-\mu}{s}\right)^2\right)^{-(\nu+1)/2}"
    parameter_explanation = (
        f"**ν = {df}** controla principalmente o peso das caudas. Com poucos graus de liberdade, a t apresenta caudas mais pesadas. "
        "À medida que ν cresce, ela se aproxima da Normal. "
        f"**μ = {loc_t:.2f}** desloca o centro e **s = {scale_t:.2f}** altera a escala horizontal da distribuição."
    )
    concept_note = (
        "Na forma padronizada, a t é simétrica em zero. A média só existe para ν > 1 e a variância só é finita para ν > 2."
    )
    fallback = (loc_t - 10 * scale_t, loc_t + 10 * scale_t)

elif distribution == "F de Fisher-Snedecor":
    st.sidebar.subheader("Parâmetros da F")
    dfn = st.sidebar.slider("Graus de liberdade do numerador d₁", 1, 100, 5, 1)
    dfd = st.sidebar.slider("Graus de liberdade do denominador d₂", 1, 200, 20, 1)

    rv = f(dfn=dfn, dfd=dfd)
    reference_rv = f(dfn=5, dfd=20)
    reference_label = "Referência: F(d₁=5, d₂=20)"
    formula = r"F=\frac{(U_1/d_1)}{(U_2/d_2)},\quad U_1\sim\chi^2_{d_1},\;U_2\sim\chi^2_{d_2}"
    parameter_explanation = (
        f"**d₁ = {dfn}** e **d₂ = {dfd}** determinam a forma e a assimetria da distribuição. "
        "Com graus de liberdade pequenos, a curva tende a ser mais assimétrica e apresentar cauda direita mais longa. "
        "À medida que ambos aumentam, a massa se concentra mais perto de 1."
    )
    concept_note = "A distribuição F assume apenas valores positivos e aparece, por exemplo, em comparações de variâncias e ANOVA."
    fallback = (0.0, 10.0)

else:
    st.sidebar.subheader("Parâmetros do Qui-quadrado")
    df_chi = st.sidebar.slider("Graus de liberdade k", 1, 100, 5, 1)

    rv = chi2(df=df_chi)
    reference_rv = chi2(df=5)
    reference_label = "Referência: χ²(k=5)"
    formula = r"f(x;k)=\frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)},\quad x>0"
    parameter_explanation = (
        f"**k = {df_chi}** controla a posição, dispersão e assimetria. Como E(X)=k e Var(X)=2k, aumentar k "
        "desloca a distribuição para a direita e aumenta sua variância absoluta; ao mesmo tempo, a assimetria relativa diminui."
    )
    concept_note = "O Qui-quadrado assume apenas valores não negativos e é a soma de quadrados de variáveis Normais padrão independentes."
    fallback = (0.0, max(10.0, df_chi + 5 * np.sqrt(2 * df_chi)))

# -----------------------------
# Numerical calculations
# -----------------------------
lo, hi = safe_interval(rv, fallback=fallback)
# For positive-support distributions, keep x >= 0
if distribution in ["F de Fisher-Snedecor", "Qui-quadrado"]:
    lo = max(0.0, lo)

# Ensure percentile is visible even for extreme parameter settings
xp = float(rv.ppf(q))
if np.isfinite(xp):
    lo = min(lo, xp)
    hi = max(hi, xp)

x = np.linspace(lo, hi, 1400)
pdf_y = rv.pdf(x)
cdf_y = rv.cdf(x)
yp = float(rv.pdf(xp)) if np.isfinite(xp) else np.nan

ref_pdf = None
ref_cdf = None
if show_reference and reference_rv is not None:
    ref_pdf = (x, reference_rv.pdf(x), reference_label)
    ref_cdf = (x, reference_rv.cdf(x), reference_label)

stats = summary_stats(rv)

# -----------------------------
# Main area
# -----------------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader(f"Distribuição {distribution}")
    st.latex(formula)
    st.info(parameter_explanation)
    st.caption(concept_note)

with right:
    st.subheader("Percentil selecionado")
    c1, c2 = st.columns(2)
    c1.metric(f"P{percentile:g}", fmt(xp))
    c2.metric("Probabilidade acumulada", f"{q:.1%}")
    st.markdown(
        f"Interpretação: **{percentile:g}%** da distribuição está à esquerda de "
        f"**x = {fmt(xp)}**; isto é, $P(X \\leq {fmt(xp)}) = {q:.3f}$."
    )

st.divider()

tab_pdf, tab_cdf, tab_summary, tab_table = st.tabs(
    ["📈 Densidade (PDF)", "📉 Acumulada (CDF)", "🧠 Efeito dos parâmetros", "🔢 Tabela de percentis"]
)

with tab_pdf:
    fig_pdf = plot_pdf(
        x,
        pdf_y,
        xp,
        yp,
        percentile,
        f"Função densidade — {distribution}",
        reference=ref_pdf,
    )
    st.plotly_chart(fig_pdf, use_container_width=True, key="pdf_chart")
    st.markdown(
        "A **PDF** mostra onde os valores são relativamente mais concentrados. "
        "A área total sob a curva é 1. A região destacada corresponde à probabilidade acumulada até o percentil escolhido."
    )

with tab_cdf:
    fig_cdf = plot_cdf(
        x,
        cdf_y,
        xp,
        percentile,
        f"Função distribuição acumulada — {distribution}",
        reference=ref_cdf,
    )
    st.plotly_chart(fig_cdf, use_container_width=True, key="cdf_chart")
    st.markdown(
        "A **CDF** é $F(x)=P(X\\leq x)$. O ponto marcado é a interseção entre o percentil escolhido "
        "e o valor de x retornado pela função quantil (PPF)."
    )

with tab_summary:
    st.subheader("Resumo numérico da distribuição atual")
    cols = st.columns(5)
    for col, (name, value) in zip(cols, stats.items()):
        col.metric(name, fmt(value))

    st.markdown("### Como interpretar as mudanças")
    if distribution == "Normal":
        st.markdown(
            "- **Aumente μ:** a curva desloca-se para a direita e todos os percentis aumentam na mesma quantidade.\n"
            "- **Diminua μ:** a curva desloca-se para a esquerda.\n"
            "- **Aumente σ:** a curva fica mais dispersa, o pico diminui e percentis extremos se afastam de μ.\n"
            "- **Diminua σ:** a curva se concentra em torno de μ."
        )
    elif distribution == "t de Student":
        st.markdown(
            "- **Aumente ν:** as caudas ficam menos pesadas e a curva converge para a Normal.\n"
            "- **Diminua ν:** eventos extremos recebem mais probabilidade; percentis de cauda ficam mais distantes.\n"
            "- **Altere μ:** a curva inteira se desloca.\n"
            "- **Altere s:** a distribuição se contrai ou se expande horizontalmente."
        )
    elif distribution == "F de Fisher-Snedecor":
        st.markdown(
            "- **d₁ pequeno:** pode produzir maior concentração próxima de zero e forte assimetria.\n"
            "- **d₂ pequeno:** aumenta o peso da cauda direita e pode tornar momentos como média/variância indefinidos em casos extremos.\n"
            "- **d₁ e d₂ maiores:** a curva tende a se concentrar mais perto de 1.\n"
            "- Os percentis superiores são especialmente importantes em testes F e ANOVA."
        )
    else:
        st.markdown(
            "- **Aumente k:** a média aumenta porque $E(X)=k$.\n"
            "- A variância também aumenta, pois $Var(X)=2k$.\n"
            "- Apesar disso, a distribuição fica relativamente menos assimétrica.\n"
            "- Para k grande, a forma se torna progressivamente mais semelhante a uma curva aproximadamente Normal."
        )

with tab_table:
    st.subheader("Percentis usuais")
    probs = np.array([0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995])
    values = rv.ppf(probs)
    table = pd.DataFrame(
        {
            "Percentil (%)": probs * 100,
            "Probabilidade acumulada": probs,
            "Valor x": values,
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Percentil (%)": "{:.1f}",
                "Probabilidade acumulada": "{:.3f}",
                "Valor x": "{:.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
with st.expander("📚 Fórmulas e observações didáticas"):
    st.markdown(
        r"""
**PDF — função densidade:** descreve a densidade relativa de probabilidade. Para variáveis contínuas, a probabilidade em um único ponto é zero; probabilidades são áreas sob a curva.

**CDF — função distribuição acumulada:**
\[
F(x)=P(X\leq x)
\]

**Percentil / quantil:** para uma probabilidade \(p\), o quantil \(x_p\) satisfaz
\[
F(x_p)=p.
\]

No código, a PDF é calculada com `pdf`, a CDF com `cdf` e o percentil com `ppf` da biblioteca `scipy.stats`.
        """
    )

st.caption(
    "Aplicativo didático em Python + Streamlit + SciPy + Plotly. "
    "A curva de referência pode ser ativada/desativada no painel lateral."
)
