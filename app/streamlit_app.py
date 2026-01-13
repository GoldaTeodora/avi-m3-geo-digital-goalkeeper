import sys
from pathlib import Path

positions = None


# --------------------------------------------------
# GARANTIR IMPORTS DO PROJETO
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

@st.cache_data
def infer_game_context_cached(X):
    return infer_game_context(X)

from src.data_loading import load_datasets
from src.preprocessing import infer_game_context

from src.kpis import (
    pi1_positional_distribution,
    pi2_distance_travelled,
    pi3_threat_frequency_by_zone,
    pi4_reaction_intensity,
    pi5_threat_progression_channels
)

from src.visualizations import (
    plot_pi1_positional_distribution_plotly,
    plot_pi2_distance_travelled,
    plot_pi3_threat_frequency_interactive,
    plot_pi4_reaction_intensity,
    plot_pi5_threat_progression_channels
)
def apply_abc_braga_theme():
    st.markdown(
        """
        <style>
        /* Fundo geral */
        .stApp {
            background-color: #0E0E0E;
            color: #FFFFFF;
        }

        /* Títulos */
        h1, h2, h3 {
            color: #F2C300;
        }

        /* Subtítulos e texto */
        p, span, label {
            color: #FFFFFF;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }

        /* Botões */
        button[kind="primary"] {
            background-color: #F2C300;
            color: #0E0E0E;
            border-radius: 8px;
            font-weight: 600;
        }

        button[kind="primary"]:hover {
            background-color: #FFD84D;
            color: #0E0E0E;
        }

        /* Cards / info boxes */
        div[data-testid="stAlert"] {
            background-color: #2A2A2A;
            border-left: 4px solid #F2C300;
        }

        /* Radio / select */
        div[role="radiogroup"] label {
            color: #FFFFFF;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# --------------------------------------------------
st.set_page_config(
    page_title="Digital Goalkeeper — Gémeo Digital",
    layout="wide"
)

apply_abc_braga_theme()


# --------------------------------------------------
# ESTADO GLOBAL DA APP
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "persona" not in st.session_state:
    st.session_state.persona = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ==================================================
# PÁGINA 1 — BOAS-VINDAS
# ==================================================
if st.session_state.page == "welcome":

    st.markdown("##  Bem-vindo ao Painel Analítico do Gémeo Digital")

    st.markdown(
        """
        Esta plataforma permite analisar o **comportamento defensivo**
        e o **desempenho do guarda-redes** através de indicadores avançados
        baseados em **dados espaciais e temporais**.

        > **Transformar dados em decisões defensivas.**  
        > Um sistema analítico orientado por **contexto, persona e evidência**.
        """
    )

    st.markdown("---")

    st.markdown("### O que pode analisar")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(" **Tempo**")
        st.caption("Padrões defensivos ao longo do jogo.")

    with col2:
        st.markdown(" **Espaço**")
        st.caption("Origem e progressão das ameaças ofensivas.")

    with col3:
        st.markdown(" **Guarda-Redes**")
        st.caption("Posicionamento, reação e carga física.")

    st.markdown("---")

    if st.button(" Continuar"):
        st.session_state.page = "persona"
    st.stop()


# ==================================================
# PÁGINA 2 — SELEÇÃO DE PERSONA
# ==================================================
elif st.session_state.page == "persona":

    st.markdown("## Selecionar persona")

    persona = st.radio(
        "Perfil de utilizador",
        ["Treinador Principal", "Treinador de Guarda-Redes"]
    )

    if persona == "Treinador Principal":
        st.info(
            " **Treinador Principal**\n\n"
            "Análise estratégica defensiva pós-jogo, focada na equipa."
        )
    else:
        st.info(
            " **Treinador de Guarda-Redes**\n\n"
            "Análise individual do desempenho do guarda-redes."
        )

    st.markdown("---")

    if st.button(" Continuar para Login"):
        st.session_state.persona = persona
        st.session_state.page = "login"
    st.stop()


# ==================================================
# PÁGINA 3 — LOGIN
# ==================================================
elif st.session_state.page == "login" and not st.session_state.authenticated:

    st.markdown("## Autenticação")

    user = st.text_input("Utilizador")
    password = st.text_input("Password", type="password")

    if st.button(" Entrar"):
        if user and password:
            st.session_state.authenticated = True
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Credenciais inválidas.")
    st.stop()




# ==================================================
# PÁGINA 4 — DASHBOARD
# ==================================================
elif st.session_state.page == "dashboard" and st.session_state.authenticated:

    if st.session_state.persona is None:
        st.warning("Persona não definida. Volte à seleção de persona.")
        st.stop()

    # --------------------------------------------------
    # CARREGAMENTO DE DADOS
    # --------------------------------------------------
    @st.cache_data
    def load_data():
        data = load_datasets()
        return data["X_train"]

    X_train = load_data()

    # --------------------------------------------------
    # SIDEBAR — CONTROLOS
    # --------------------------------------------------
    st.sidebar.title("Configurações")

    st.sidebar.markdown(f"**Persona:** {st.session_state.persona}")

    # Escala visual
    fig_scale = st.sidebar.slider(
        "Escala das visualizações",
        0.5, 1.5, 1.0, 0.1
    )

    # Intervalo temporal
    frame_start, frame_end = st.sidebar.slider(
        "Selecionar frames",
        0, len(X_train), (0, len(X_train))
    )

    step = st.sidebar.selectbox(
        "Subamostragem (frames)",
        [1, 5, 10, 20, 50],
        index=1
    )

    context = st.sidebar.selectbox(
        "Contexto de análise",
        ["Pós-Jogo", "Treino"]
    )

    data_context = "Jogo" if context == "Pós-Jogo" else context

    X_filtered = X_train.iloc[frame_start:frame_end:step]
    X_contextual = infer_game_context_cached(X_filtered)
    X_persona = (
    X_contextual[X_contextual["contexto"] == data_context]
    if "contexto" in X_contextual.columns
    else X_contextual
)


    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------
    @st.cache_data
    def compute_kpis(X):
        return {
            "pi1": pi1_positional_distribution(X),
            "pi2": pi2_distance_travelled(X),
            "pi3": pi3_threat_frequency_by_zone(X),
            "pi4": pi4_reaction_intensity(X),
        }

    kpis = compute_kpis(X_persona)

  
# ==================================================
# DASHBOARD — TREINADOR PRINCIPAL
# ==================================================
if st.session_state.persona == "Treinador Principal":

    st.header("📊 Análise Estratégica Defensiva")
    st.caption("Contexto: Pós-Jogo")

    selected_pi = st.sidebar.radio(
        "Selecionar indicador",
        [
            "PI 3 — Origem Espacial das Ameaças",
            "PI 5 — Canal de Progressão das Ameaças"
        ],
        key="pi_tr_principal"
    )

    if selected_pi == "PI 3 — Origem Espacial das Ameaças":
        heatmap = kpis["pi3"]["heatmap"]
        fig = plot_pi3_threat_frequency_interactive(heatmap)
        fig.update_layout(height=int(500 * fig_scale))
        st.plotly_chart(fig, width="stretch")

    elif selected_pi == "PI 5 — Canal de Progressão das Ameaças":
        pi5 = pi5_threat_progression_channels(X_persona)
        fig = plot_pi5_threat_progression_channels(pi5)
        st.plotly_chart(fig, width="stretch")


# ==================================================
# DASHBOARD — TREINADOR DE GUARDA-REDES
# ==================================================
elif st.session_state.persona == "Treinador de Guarda-Redes":

    st.header("🧤 Análise do Guarda-Redes")
    st.caption("Contexto: Pós-Jogo")

    selected_pi = st.sidebar.radio(
        "Selecionar indicador (Guarda-Redes)",
        [
            "PI 1 — Distribuição Posicional",
            "PI 2 — Distância Percorrida",
            "PI 4 — Intensidade de Reação"
        ],
        key="pi_gr"
    )

    # ==================================================
    # PI 1 — Distribuição Posicional
    # ==================================================
    if selected_pi == "PI 1 — Distribuição Posicional":

       st.sidebar.markdown("### Visualização do PI 1")

       view_mode = st.sidebar.radio(
           "Modo de visualização",
           ["Pontinhos (posições)", "Densidade (padrão)"],
           index=0
        )

       zone_color_mode = st.sidebar.radio(
           "Cores das zonas",
           ["Semântico (Azul/Verde/Vermelho)", "Neutro"],
           index=0
        )


       pi1 = kpis.get("pi1")
  
       if pi1 is None or pi1.get("positions") is None or pi1["positions"].empty:
           st.warning("Dados insuficientes para análise posicional com os filtros atuais.")
           st.stop()

       fig = plot_pi1_positional_distribution_plotly(
            positions=pi1["positions"],
            mean_position=pi1["mean_position"],
            tactical_reading=pi1["tactical_reading"],
            view_mode=view_mode,
            zone_color_mode=zone_color_mode
)


       st.plotly_chart(fig, use_container_width=True)

       zone_info = pi1["zone_distribution"]
       reading = pi1["tactical_reading"]

       st.markdown("###  Leitura Tática Automática")

       st.caption(
    "Análise baseada na profundidade média do guarda-redes "
    "ao longo do jogo (não posição instantânea)."
)


       st.write(
          f"""
          **Zona Baixa:** {zone_info['baixo']:.1f}%  
          **Zona Média:** {zone_info['medio']:.1f}%  
          **Zona Alta:** {zone_info['alto']:.1f}%  
          """
        )

       st.info(f" **Interpretação:** {reading}")


    # --------------------------------------------------
    # PI 2 — Distância Percorrida
    # --------------------------------------------------
    elif selected_pi == "PI 2 — Distância Percorrida":

        distances = kpis["pi2"]["instant_distances"]

        if distances is None or len(distances) == 0:
            st.warning("Sem dados de deslocamento.")
            st.stop()

        fig = plot_pi2_distance_travelled(distances)
        st.pyplot(fig)

    # --------------------------------------------------
    # PI 4 — Intensidade de Reação
    # --------------------------------------------------
    elif selected_pi == "PI 4 — Intensidade de Reação":

        speeds = kpis["pi4"]["speed_series"]

        if speeds is None or len(speeds) == 0:
            st.warning("Sem dados de velocidade.")
            st.stop()

        fig = plot_pi4_reaction_intensity(
            speeds,
            pi1["mean_speed"],
            pi1["max_speed"]
        )
        st.pyplot(fig)
