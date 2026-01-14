import pandas as pd
from src.visualizations import plot_pi1_positional_distribution_plotly
from src.kpis import pi1_positional_distribution
from src.data_loading import load_datasets

# carregar dados
data = load_datasets()
X = data["X_train"]

# KPI
pi1 = pi1_positional_distribution(X)

# TESTE DIRETO (fora do Streamlit)
fig = plot_pi1_positional_distribution_plotly(
    positions=pi1["positions"],
    mean_position=pi1["mean_position"],
    tactical_reading=pi1["tactical_reading"],
    view_mode="pontinhos"
)

fig.show()
