
# Digital Goalkeeper — Projeto AVI M3

Projeto desenvolvido no âmbito da unidade curricular **Apresentação e Visualização da Informação (AVI)** — Mestrado.

O objetivo é analisar o **comportamento defensivo e o desempenho do guarda-redes** através de **KPIs espaciais e temporais**, suportados por visualizações interativas em **Streamlit**.

Criar um **painel analítico (dashboard)** que permita:
- Analisar o **posicionamento do guarda-redes**
- Avaliar **carga física e intensidade de reação**
- Identificar **origem e progressão das ameaças ofensivas**
- Suportar decisões técnicas em contexto **pós-jogo** e **treino**

O projeto segue uma abordagem orientada a:
- **Persona** (Treinador Principal vs Treinador de Guarda-Redes)
- **Contexto de análise**
- **Evidência baseada em dados**

## Indicadores implementados e que falta Implementar (KPIs)

### Treinador do Guarda-Redes
- **PI 1 — Distribuição Posicional**  
  Análise espacial da ocupação do guarda-redes e posição média.

- **PI 2 — Distância Percorrida**  Falta implementar
  Cálculo da distância acumulada ao longo do tempo.

- **PI 4 — Intensidade de Reação**  Falta implementar
  Análise da velocidade instantânea, média e máxima.

### Treinador principal
- **PI 3 — Origem Espacial das Ameaças**  
  Densidade espacial das posições da bola.
- **PI 5 — Canal de Progressão das Ameaças**  
  Identificação do corredor preferencial (esquerdo, central, direito).


## Estrutura do Projeto

avi-m3-geo-digital-goalkeeper/
│
├── app/
│   └── streamlit_app.py        # Aplicação Streamlit (dashboard)
│
├── src/
│   ├── data_loading.py         # Carregamento de datasets
│   ├── preprocessing.py       # Inferência de contexto
│   ├── kpis.py                 # Cálculo dos indicadores (PI 1–5)
│   ├── visualizations.py       # Visualizações (matplotlib / plotly)
│   └── utils.py                # Funções auxiliares
│
├── data/
│   └── raw/                    # (não incluído no GitHub)
│
├── docs/
│   └── architecture.md         # Notas de arquitetura
│
├── test_pi5.py                 # Script de teste isolado do PI 5
├── requirements.txt            # Dependências
└── README.md



## Como Executar a Aplicação

### 1. Clonar o repositório
```bash
git clone https://github.com/GoldaTeodora/avi-m3-geo-digital-goalkeeper.git
cd avi-m3-geo-digital-goalkeeper


### 2. Criar ambiente virtual

```bash
python -m venv .venv

Ativar:

* **Windows**

```bash
.venv\Scripts\activate


* **Mac / Linux**

```bash
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt


### 4. Executar o Streamlit

```bash
streamlit run app/streamlit_app.py


## 📊 Dados

Os **datasets não estão incluídos no repositório** por motivos de:

* dimensão
* boas práticas de versionamento
* contexto académico

A aplicação assume a existência de ficheiros CSV em:

data/raw/

Com colunas como:

* `#x0`, `#y0` — posição do guarda-redes
* `#vx0`, `#vy0` — velocidade
* `#ball_x`, `#ball_y` — posição da bola
