# =====================================================
# KPIs — DIGITAL GOALKEEPER
# =====================================================

import pandas as pd
import numpy as np


# =====================================================
# PI 1 — Distribuição Posicional do Guarda-Redes
# Persona: Treinador de Guarda-Redes
# =====================================================
def pi1_positional_distribution(X: pd.DataFrame):
    """
    PI 1 — Distribuição Posicional do Guarda-Redes
    Persona: Treinador de Guarda-Redes
    """

    # -----------------------------
    # Validação mínima
    # -----------------------------
    required_cols = {"#x0", "#y0"}
    if not required_cols.issubset(X.columns):
        raise ValueError("PI1 requer colunas '#x0' e '#y0'.")

    positions = X[["#x0", "#y0"]].copy()
    positions = positions.replace([np.inf, -np.inf], np.nan).dropna()

    if len(positions) < 20:
        return {
            "positions": positions,
            "mean_position": (np.nan, np.nan),
            "zone_distribution": {"baixo": 0.0, "medio": 0.0, "alto": 0.0},
            "tactical_reading": "Dados insuficientes para leitura tática"
        }

    # -----------------------------
    # Posição média (visual)
    # -----------------------------
    mean_x = positions["#x0"].mean()
    mean_y = positions["#y0"].mean()

    # -----------------------------
    # Zonas funcionais (eixo Y)
    # -----------------------------
    ZONE_LOW_Y = 0.18
    ZONE_MID_Y = 0.35

    y = positions["#y0"].values
    total = len(y)

    zone_low = np.sum(y <= ZONE_LOW_Y) / total * 100
    zone_mid = np.sum((y > ZONE_LOW_Y) & (y <= ZONE_MID_Y)) / total * 100
    zone_high = np.sum(y > ZONE_MID_Y) / total * 100

    zone_distribution = {
        "baixo": zone_low,
        "medio": zone_mid,
        "alto": zone_high
    }

    # -----------------------------
    # 🔑 PROFUNDIDADE DOMINANTE
    # -----------------------------
    dominant_y = positions["#y0"].quantile(0.65)

    # -----------------------------
    # Leitura tática automática
    # -----------------------------
    if dominant_y <= ZONE_LOW_Y:
        tactical_reading = "Guarda-redes de baliza (baixo envolvimento)"
    elif dominant_y <= ZONE_MID_Y:
        tactical_reading = "Guarda-redes equilibrado (zona de cobertura)"
    else:
        tactical_reading = "Guarda-redes frequentemente adiantado (sweeper)"

    return {
        "positions": positions,
        "mean_position": (mean_x, mean_y),
        "zone_distribution": zone_distribution,
        "tactical_reading": tactical_reading
    }



# =====================================================
# PI 2 — Distância Percorrida
# Persona: Treinador de Guarda-Redes
# =====================================================
def pi2_distance_travelled(X: pd.DataFrame):
    """
    PI 2 — Distância Percorrida pelo Guarda-Redes
    """

    if "#x0" not in X.columns or "#y0" not in X.columns:
        raise ValueError("PI2 requer colunas '#x0' e '#y0'.")

    x = X["#x0"].values
    y = X["#y0"].values

    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(dx**2 + dy**2)

    return {
        "total_distance": float(np.nansum(distances)),
        "instant_distances": distances
    }


# =====================================================
# PI 3 — Frequência de Ameaças por Zona
# Persona: Treinador Principal
# =====================================================
def pi3_threat_frequency_by_zone(
    X: pd.DataFrame,
    bins_x: int = 10,
    bins_y: int = 10
):
    """
    PI 3 — Frequência de Ameaças por Zona
    """

    if "#ball_x" not in X.columns or "#ball_y" not in X.columns:
        raise ValueError("PI3 requer colunas '#ball_x' e '#ball_y'.")

    heatmap, x_edges, y_edges = np.histogram2d(
        X["#ball_x"].values,
        X["#ball_y"].values,
        bins=[bins_x, bins_y],
        range=[[0, 1], [0, 1]]
    )

    return {
        "heatmap": heatmap,
        "x_edges": x_edges,
        "y_edges": y_edges
    }


# =====================================================
# PI 4 — Intensidade de Reação
# Persona: Treinador de Guarda-Redes
# =====================================================
def pi4_reaction_intensity(X: pd.DataFrame):
    """
    PI 4 — Intensidade de Reação do Guarda-Redes
    """

    if "#vx0" not in X.columns or "#vy0" not in X.columns:
        raise ValueError("PI4 requer colunas '#vx0' e '#vy0'.")

    vx = X["#vx0"].values
    vy = X["#vy0"].values

    speed_series = np.sqrt(vx**2 + vy**2)

    return {
        "speed_series": speed_series,
        "mean_speed": float(np.nanmean(speed_series)),
        "max_speed": float(np.nanmax(speed_series))
    }


# =====================================================
# PI 5 — Canal de Progressão das Ameaças
# Persona: Treinador Principal
# =====================================================
def pi5_threat_progression_sankey(X: pd.DataFrame):
    """
    PI 5 — Canal de Progressão das Ameaças (Sankey)
    Unidade: sequência ofensiva
    """

    if "#ball_x" not in X.columns or "#ball_y" not in X.columns:
        raise ValueError("PI5 requer colunas '#ball_x' e '#ball_y'.")

    def channel(x):
        if x < 0.33:
            return "Esquerdo"
        elif x > 0.66:
            return "Direito"
        return "Central"

    flows = []

    current_sequence = []

    for _, row in X.iterrows():
        current_sequence.append(row)

        # ameaça termina quando entra em zona crítica
        if row["#ball_y"] > 0.75:
            df_seq = pd.DataFrame(current_sequence)

            origin = channel(df_seq.iloc[0]["#ball_x"])
            progression = df_seq["#ball_x"].apply(channel).mode()[0]
            final = channel(df_seq.iloc[-1]["#ball_x"])

            flows.append((origin, progression, final))
            current_sequence = []

    counts = {}
    for f in flows:
        counts[f] = counts.get(f, 0) + 1

    # -----------------------------
    # Construção dos nós Sankey
    # -----------------------------
    labels = ["Esquerdo", "Central", "Direito"]

    label_index = {label: i for i, label in enumerate(labels)}

    sources = []
    targets = []
    values = []

    for (origin, progression, final), count in counts.items():
        # origem → progressão
        sources.append(label_index[origin])
        targets.append(label_index[progression])
        values.append(count)

        # progressão → final
        sources.append(label_index[progression])
        targets.append(label_index[final])
        values.append(count)

    return {
        "labels": labels,
        "sources": sources,
        "targets": targets,
        "values": values,
        "raw_flows": flows  # mantido para debug/análise
}

