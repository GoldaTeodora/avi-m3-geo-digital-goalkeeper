# =====================================================
# VISUALIZAÇÕES — DIGITAL GOALKEEPER
# =====================================================

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from functools import lru_cache

@lru_cache(maxsize=8)
def compute_kde_cached(x_tuple, y_tuple):
    x = np.array(x_tuple)
    y = np.array(y_tuple)
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.25)
    xi, yi = np.mgrid[0:1:120j, 0:1:120j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    return xi, yi, zi


# =====================================================
# PI 1 — Distribuição Posicional do Guarda-Redes (INTERATIVO)
# Persona: Treinador de Guarda-Redes
# =====================================================

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from functools import lru_cache


# --------------------------------------------------
# KDE cacheado (performance)
# --------------------------------------------------
@lru_cache(maxsize=8)
def compute_kde_cached(x_tuple, y_tuple):
    x = np.array(x_tuple)
    y = np.array(y_tuple)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.25)
    xi, yi = np.mgrid[0:1:120j, 0:1:120j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    return xi, yi, zi


# --------------------------------------------------
# PI 1 — Visualização
# --------------------------------------------------
def plot_pi1_positional_distribution_plotly(
    positions: pd.DataFrame,
    mean_position: tuple | None,
    tactical_reading: str,
    view_mode: str = "densidade",      # "densidade" | "pontinhos"
    zone_color_mode: str = "Neutro"     # "Neutro" | "Semântico"
):
    """
    PI 1 — Distribuição Posicional do Guarda-Redes
    Visualização robusta, rápida e defensável (TRL-6)
    """

    # --------------------------------------------------
    # Robustez mínima
    # --------------------------------------------------
    if positions is None or positions.empty:
        fig = go.Figure()
        fig.update_layout(
            title="PI 1 — Distribuição Posicional do Guarda-Redes",
            plot_bgcolor="#0E0E0E",
            paper_bgcolor="#0E0E0E",
            annotations=[
                dict(
                    text="Dados insuficientes para visualização posicional",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="white", size=14)
                )
            ]
        )
        return fig

    fig = go.Figure()

    # --------------------------------------------------
    # Campo normalizado
    # --------------------------------------------------
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[0, 1],
        visible=False,
        scaleanchor="x",
        fixedrange=True
    )

    fig.update_layout(
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        margin=dict(l=180, r=40, t=60, b=40),
        title="PI 1 — Distribuição Posicional do Guarda-Redes",
        legend=dict(font=dict(color="white"))
    )

    # --------------------------------------------------
    # Zonas funcionais (FIXAS)
    # --------------------------------------------------
    Z_LOW = 0.18
    Z_MID = 0.35

    if zone_color_mode == "Semântico":
        colors = {
            "low": "rgba(31,119,180,0.25)",
            "mid": "rgba(44,160,44,0.25)",
            "high": "rgba(214,39,40,0.25)"
        }
    else:
        colors = {
            "low": "rgba(180,180,180,0.12)",
            "mid": "rgba(140,140,140,0.12)",
            "high": "rgba(100,100,100,0.12)"
        }

    # Zonas de fundo
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=Z_LOW,
                  fillcolor=colors["low"], line=dict(width=0), layer="below")

    fig.add_shape(type="rect", x0=0, x1=1, y0=Z_LOW, y1=Z_MID,
                  fillcolor=colors["mid"], line=dict(width=0), layer="below")

    fig.add_shape(type="rect", x0=0, x1=1, y0=Z_MID, y1=1,
                  fillcolor=colors["high"], line=dict(width=0), layer="below")

    # Linhas horizontais
    for y in [Z_LOW, Z_MID]:
        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=y, y1=y,
            line=dict(color="white", dash="dot", width=1)
        )

    # --------------------------------------------------
    # Labels alinhadas às linhas (fora do campo)
    # --------------------------------------------------
    labels = [
        ("Zona Alta — Comportamento sweeper", Z_MID),
        ("Zona Média — Zona de cobertura", Z_LOW),
        ("Zona Baixa — Linha da baliza", 0.0),
    ]

    for text, y in labels:
        fig.add_annotation(
            x=-0.06,
            y=y,
            xref="paper",
            yref="y",
            text=text,
            showarrow=False,
            font=dict(color="white", size=12),
            xanchor="right",
            yanchor="middle",
            align="right"
        )

    # --------------------------------------------------
    # MODO: DENSIDADE
    # --------------------------------------------------
    if view_mode == "densidade" and len(positions) > 30:
        try:
            MAX_KDE_POINTS = 3000
            positions_kde = (
                positions.sample(n=MAX_KDE_POINTS, random_state=42)
                if len(positions) > MAX_KDE_POINTS
                else positions
            )

            x = positions_kde["#x0"].values
            y = positions_kde["#y0"].values

            xi, yi, zi = compute_kde_cached(
                tuple(x.round(4)),
                tuple(y.round(4))
            )

            fig.add_trace(
                go.Contour(
                    x=xi[:, 0],
                    y=yi[0, :],
                    z=zi.T,
                    colorscale="Blues",
                    ncontours=10,
                    opacity=0.9,
                    showscale=True,
                    hoverinfo="skip",
                    name="Densidade"
                )
            )
        except Exception:
            pass

    # --------------------------------------------------
    # MODO: PONTINHOS
    # --------------------------------------------------
    if view_mode == "pontinhos":
        fig.add_trace(
            go.Scatter(
                x=positions["#x0"],
                y=positions["#y0"],
                mode="markers",
                marker=dict(size=4, color="rgba(255,255,255,0.35)"),
                name="Posições"
            )
        )

    # --------------------------------------------------
    # Posição média do guarda-redes
    # --------------------------------------------------
    if mean_position is not None:
        fig.add_trace(
            go.Scatter(
                x=[mean_position[0]],
                y=[mean_position[1]],
                mode="markers",
                marker=dict(
                    size=22,
                    color="white",
                    line=dict(color="black", width=2)
                ),
                name="Guarda-Redes"
            )
        )

        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=mean_position[1],
            y1=mean_position[1],
            line=dict(color="#FFD700", width=2, dash="dot")
        )

    # --------------------------------------------------
    # Perfil tático
    # --------------------------------------------------
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=f"Perfil: {tactical_reading}",
        showarrow=False,
        font=dict(size=13, color="#FFD700")
    )

    return fig






# =====================================================
# PI 2 — Distância Percorrida
# =====================================================
def plot_pi2_distance_travelled(distances):
    cumulative_distance = np.cumsum(distances)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative_distance)

    ax.set_title("PI 2 — Distância Percorrida pelo Guarda-Redes")
    ax.set_xlabel("Instante (frames)")
    ax.set_ylabel("Distância acumulada")

    return fig


# =====================================================
# PI 3 — Frequência de Ameaças (INTERATIVO)
# =====================================================
def plot_pi3_threat_frequency_interactive(heatmap):
    """
    PI 3 — Origem Espacial das Ameaças Ofensivas
    Versão consolidada (sem erros).
    """

    heatmap_safe = np.where(heatmap <= 0, 1, heatmap)
    heatmap_log = np.log10(heatmap_safe)

    df = pd.DataFrame(
        heatmap_log,
        columns=[f"X{i}" for i in range(heatmap.shape[1])],
        index=[f"Y{i}" for i in range(heatmap.shape[0])]
    )

    fig = px.imshow(
        df,
        color_continuous_scale="viridis",
        aspect="equal",
        labels=dict(color="log10(Frequência)")
    )

    fig.update_coloraxes(
        cmin=heatmap_log.min(),
        cmax=np.percentile(heatmap_log, 95),
        colorbar=dict(
            title="Frequência de Ameaças",
            tickmode="array",
            tickvals=[1.3, 1.6, 2.0],
            ticktext=[
                "Baixa",
                "Média",
                "Alta"
            ]
            )
    )

    fig.update_layout(
        title="PI 3 — Origem Espacial das Ameaças Ofensivas",
        template="plotly_dark",
        xaxis_title="Eixo X (zonas)",
        yaxis_title="Eixo Y (zonas)"
    )

    return fig



# =====================================================
# PI 4 — Intensidade de Reação
# =====================================================
def plot_pi4_reaction_intensity(
    speeds: np.ndarray,
    mean_speed: float,
    max_speed: float
):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(speeds, alpha=0.6, label="Velocidade instantânea")

    ax.axhline(
        mean_speed,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Média ({mean_speed:.2f})"
    )

    ax.axhline(
        max_speed,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Máxima ({max_speed:.2f})"
    )

    ax.set_title("PI 4 — Intensidade de Reação do Guarda-Redes")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Velocidade")
    ax.legend()

    return fig


# =====================================================
# PI 5 — Canal de Progressão das Ameaças (BARRAS)
# =====================================================
def plot_pi5_threat_progression_channels(pi5_data: dict):
    """
    Visualização do PI 5 — Canal de Progressão das Ameaças
    (barras, versão defensável para tese)
    """

    channels = ["Esquerdo", "Central", "Direito"]
    counts = [pi5_data["counts"][c] for c in channels]
    percentages = [pi5_data["percentages"][c] for c in channels]

    colors = ["#4C78A8", "#F2C300", "#4C78A8"]  # Central em destaque

    fig = go.Figure(
        data=[
            go.Bar(
                x=channels,
                y=counts,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                marker_color=colors
            )
        ]
    )

    fig.update_layout(
        title="PI 5 — Canal de Progressão das Ameaças Ofensivas",
        xaxis_title="Canal do Campo",
        yaxis_title="Número de Ameaças",
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=60, b=40, l=40, r=40)
    )

    return fig
