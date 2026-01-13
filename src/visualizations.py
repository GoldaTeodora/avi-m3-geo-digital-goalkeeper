# =====================================================
# VISUALIZAÇÕES — DIGITAL GOALKEEPER
# =====================================================

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


# =====================================================
# PI 1 — Distribuição Posicional do Guarda-Redes (INTERATIVO)
# Persona: Treinador de Guarda-Redes
# =====================================================
def plot_pi1_positional_distribution_plotly(
    positions: pd.DataFrame,
    mean_position: tuple,
    tactical_reading: str,
    view_mode: str = "Pontinhos (posições)",
    zone_color_mode: str = "Semântico (Azul/Verde/Vermelho)"
):

    if positions is None or positions.empty:
        fig = go.Figure()
        fig.update_layout(
            title="PI 1 — Distribuição Posicional do Guarda-Redes",
            plot_bgcolor="#0E0E0E",
            paper_bgcolor="#0E0E0E"
        )
        return fig

    # Subamostragem (performance)
    MAX_POINTS = 3000
    if len(positions) > MAX_POINTS:
        positions = positions.sample(MAX_POINTS, random_state=42)

    fig = go.Figure()

    # Campo normalizado
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[0, 1],
        visible=False,
        scaleanchor="x",
        fixedrange=True
    )

    fig.update_layout(
        title="PI 1 — Distribuição Posicional do Guarda-Redes",
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        margin=dict(l=40, r=220, t=90, b=40),  # espaço lateral p/ legendas
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
    )

    # Baliza
    fig.add_shape(
        type="line",
        x0=0.44, x1=0.56,
        y0=0.0, y1=0.0,
        line=dict(color="white", width=3)
    )

    # Área defensiva
    fig.add_shape(
        type="rect",
        x0=0.30, x1=0.70,
        y0=0.00, y1=0.18,
        line=dict(color="white", dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )

    # Zonas funcionais (APENAS CORES — SEM TEXTO)
    if zone_color_mode == "Semântico (Azul/Verde/Vermelho)":
        zones = [
            (0.00, 0.18, "rgba(31,119,180,0.25)"),
            (0.18, 0.35, "rgba(44,160,44,0.25)"),
            (0.35, 1.00, "rgba(214,39,40,0.25)")
        ]
    else:
        zones = [
            (0.00, 0.18, "rgba(200,200,200,0.15)"),
            (0.18, 0.35, "rgba(160,160,160,0.15)"),
            (0.35, 1.00, "rgba(120,120,120,0.15)")
        ]

    for y0, y1, color in zones:
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=y0, y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )

    # Scatter ou densidade
    x = positions["#x0"].values
    y = positions["#y0"].values

    if view_mode == "Pontinhos (posições)":
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, color="rgba(255,255,255,0.35)"),
                name="Posições",
                hovertemplate=(
                    "Largura: %{x:.2f}<br>"
                    "Profundidade: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )
    else:
        if len(x) > 30:
            values = np.vstack([x, y])
            kde = gaussian_kde(values, bw_method=0.25)

            xi, yi = np.mgrid[0:1:200j, 0:1:200j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            zi = zi.reshape(xi.shape)

            fig.add_trace(
                go.Contour(
                    x=xi[:, 0],
                    y=yi[0, :],
                    z=zi.T,
                    ncontours=8,
                    colorscale="Blues",
                    opacity=0.85,
                    contours=dict(coloring="fill"),
                    showscale=True,
                    hoverinfo="skip",
                    name="Densidade Posicional"
                )
            )

    # Posição média
    fig.add_trace(
        go.Scatter(
            x=[mean_position[0]],
            y=[mean_position[1]],
            mode="markers",
            marker=dict(
                size=14,
                color="#FFD700",
                line=dict(color="black", width=2)
            ),
            name="Posição Média"
        )
    )

    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=mean_position[1],
        y1=mean_position[1],
        line=dict(color="#FFD700", width=2, dash="dot")
    )

    # Guarda-redes
    fig.add_trace(
        go.Scatter(
            x=[mean_position[0]],
            y=[mean_position[1]],
            mode="markers",
            marker=dict(
                size=26,
                color="rgba(255,255,255,0.95)",
                line=dict(color="black", width=2)
            ),
            name="Guarda-Redes"
        )
    )

    # ---------- LEGENDAS EXTERNAS (FORA DO GRÁFICO) ----------
    fig.add_annotation(
        x=1.05, y=0.78,
        text="🔴 Zona Alta<br><span style='font-size:11px'>Comportamento sweeper</span>",
        showarrow=False,
        align="left",
        font=dict(color="white", size=13),
        xref="paper", yref="paper"
    )

    fig.add_annotation(
        x=1.05, y=0.52,
        text="🟢 Zona Média<br><span style='font-size:11px'>Zona de cobertura</span>",
        showarrow=False,
        align="left",
        font=dict(color="white", size=13),
        xref="paper", yref="paper"
    )

    fig.add_annotation(
        x=1.05, y=0.26,
        text="🔵 Zona Baixa<br><span style='font-size:11px'>Linha da baliza</span>",
        showarrow=False,
        align="left",
        font=dict(color="white", size=13),
        xref="paper", yref="paper"
    )

    # Texto explicativo superior
    fig.add_annotation(
        x=0.5, y=1.08,
        text="Eixo vertical = profundidade do guarda-redes (baliza → campo)",
        showarrow=False,
        font=dict(size=12, color="rgba(255,255,255,0.7)"),
        xanchor="center"
    )

    fig.add_annotation(
        x=0.5, y=1.02,
        text=f"Perfil: {tactical_reading}",
        showarrow=False,
        font=dict(size=13, color="#FFD700"),
        xanchor="center"
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
