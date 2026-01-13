# =====================================================
# VISUALIZAÇÕES — DIGITAL GOALKEEPER
# =====================================================

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# =====================================================
# PI 1 — Distribuição Posicional do Guarda-Redes (INTERATIVO)
# Persona: Treinador de Guarda-Redes
# =====================================================
def plot_pi1_positional_distribution_plotly(
    positions: pd.DataFrame,
    mean_position: tuple,
    tactical_reading: str
):
    # --------------------------------------------------
    # Robustez mínima
    # --------------------------------------------------
    if positions is None or positions.empty:
        fig = go.Figure()
        fig.update_layout(
            title="PI 1 — Distribuição Posicional do Guarda-Redes",
            plot_bgcolor="#0E0E0E",
            paper_bgcolor="#0E0E0E"
        )
        return fig

    # --------------------------------------------------
    # Subamostragem (CRÍTICO para performance)
    # --------------------------------------------------
    MAX_POINTS = 3000
    if len(positions) > MAX_POINTS:
        positions = positions.sample(MAX_POINTS, random_state=42)

    fig = go.Figure()

    # --------------------------------------------------
    # Campo normalizado (0–1)
    # --------------------------------------------------
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[0, 1],
        visible=False,
        scaleanchor="x",
        fixedrange=True
    )

    # --------------------------------------------------
    # Layout base
    # --------------------------------------------------
    fig.update_layout(
        title="PI 1 — Distribuição Posicional do Guarda-Redes",
        plot_bgcolor="#0E0E0E",
        paper_bgcolor="#0E0E0E",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
    )

    # --------------------------------------------------
    # Baliza
    # --------------------------------------------------
    fig.add_shape(
        type="line",
        x0=0.44, x1=0.56,
        y0=0.0, y1=0.0,
        line=dict(color="white", width=3)
    )

    # --------------------------------------------------
    # Área defensiva
    # --------------------------------------------------
    fig.add_shape(
        type="rect",
        x0=0.30, x1=0.70,
        y0=0.00, y1=0.18,
        line=dict(color="white", dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )

    # --------------------------------------------------
    # Zonas funcionais
    # --------------------------------------------------
    zones = [
    (0.00, 0.18, "Baixa (linha de baliza)", "rgba(31,119,180,0.12)"),
    (0.18, 0.35, "Média (zona de cobertura)", "rgba(44,160,44,0.12)"),
    (0.35, 1.00, "Alta (comportamento sweeper)", "rgba(214,39,40,0.10)")
]


    for y0, y1, label, color in zones:
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=y0, y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )

        fig.add_annotation(
            x=0.01,
            y=(y0 + y1) / 2,
            text=label,
            showarrow=False,
            font=dict(color="white", size=11),
            xanchor="left"
        )

    # --------------------------------------------------
    # Classificação de zona (hover)
    # --------------------------------------------------
    def classify_zone(y):
        if y <= 0.18:
            return "Zona Baixa"
        elif y <= 0.35:
            return "Zona Média"
        else:
            return "Zona Alta"

    zones_hover = positions["#y0"].apply(classify_zone)

    # --------------------------------------------------
    # Scatter posicional
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=positions["#x0"],
            y=positions["#y0"],
            mode="markers",
            marker=dict(
                size=4,
                color="rgba(255,255,255,0.25)"
            ),
            customdata=zones_hover,
            hovertemplate=(
                "x: %{x:.2f}<br>"
                "y: %{y:.2f}<br>"
                "<b>%{customdata}</b>"
                "<extra></extra>"
            ),
            name="Posições"
        )
    )

    # --------------------------------------------------
    # Posição média
    # --------------------------------------------------
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

    
    # --------------------------------------------------
    # 🧤 Guarda-redes (representação explícita)
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
             x=[mean_position[0]],
             y=[mean_position[1]],
             mode="markers",
             marker=dict(
                 size=26,                 # maior que a posição média
                 color="rgba(255,255,255,0.95)",
                 line=dict(
                    color="black",
                    width=2
                ),
              symbol="circle"
        ),
        name="Guarda-Redes",
        hovertemplate=(
            "<b>Guarda-Redes</b><br>"
            "x: %{x:.2f}<br>"
            "y: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


    # --------------------------------------------------
    # Perfil tático
    # --------------------------------------------------
    fig.add_annotation(
        x=0.5,
        y=0.97,
        text=f"Perfil: {tactical_reading}",
        showarrow=False,
        font=dict(size=13, color="#FFD700"),
        xanchor="center"
    )

    fig.add_annotation(
    x=0.5,
    y=1.05,
    text="Eixo vertical = profundidade do guarda-redes (baliza → campo)",
    showarrow=False,
    font=dict(size=12, color="rgba(255,255,255,0.7)"),
    xanchor="center",
    yanchor="bottom"
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
# PI 3 — Frequência de Ameaças por Zona (ESTÁTICO)
# =====================================================
def plot_pi3_threat_frequency(heatmap: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot"
    )

    ax.set_title("PI 3 — Frequência de Ameaças por Zona")
    ax.set_xlabel("Eixo X (zonas)")
    ax.set_ylabel("Eixo Y (zonas)")

    plt.colorbar(im, ax=ax)

    return fig


# =====================================================
# PI 3 — Origem Espacial das Ameaças (INTERATIVO)
# =====================================================
def plot_pi3_threat_frequency_interactive(heatmap: np.ndarray):
    heatmap_safe = np.where(heatmap <= 0, 1, heatmap)
    heatmap_log = np.log10(heatmap_safe)

    df = pd.DataFrame(heatmap_log)

    fig = px.imshow(
        df,
        color_continuous_scale="viridis",
        aspect="equal"
    )

    fig.update_layout(
        title="PI 3 — Origem Espacial das Ameaças Ofensivas",
        template="plotly_dark"
    )

    return fig


# =====================================================
# PI 4 — Intensidade de Reação
# =====================================================
def plot_pi4_reaction_intensity(speeds, mean_speed, max_speed):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(speeds, alpha=0.6)

    ax.axhline(mean_speed, linestyle="--", linewidth=2)
    ax.axhline(max_speed, linestyle=":", linewidth=2)

    ax.set_title("PI 4 — Intensidade de Reação do Guarda-Redes")
    ax.set_xlabel("Instante (frames)")
    ax.set_ylabel("Velocidade")

    return fig


# =====================================================
# PI 5 — Canal de Progressão das Ameaças
# =====================================================
def plot_pi5_threat_progression_channels(pi5_data: dict):
    channels = ["Esquerdo", "Central", "Direito"]
    counts = [pi5_data["counts"][c] for c in channels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=channels,
                y=counts
            )
        ]
    )

    fig.update_layout(
        title="PI 5 — Canal de Progressão das Ameaças Ofensivas",
        template="plotly_dark"
    )

    return fig
