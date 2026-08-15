import streamlit as st
import duckdb
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Preditor de Futebol IA",
    page_icon="⚽",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Carregamento do Modelo e Conexão com o DuckDB
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo V3 treinado e a lista de colunas esperadas."""
    dados_modelo = joblib.load('modelo_futebol_v2.pkl')
    return dados_modelo['modelo'], dados_modelo['features']

try:
    modelo, feature_names = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo 'modelo_futebol_v2.pkl'. Verifique se executou o script de treino V3. Detalhes: {e}")
    st.stop()

def get_db_connection():
    return duckdb.connect('futebol.db', read_only=True)

con = get_db_connection()

# -----------------------------------------------------------------------------
# 3. Sidebar - Interface de Seleção
# -----------------------------------------------------------------------------
st.sidebar.title("⚽ Controle da Simulação")
st.sidebar.markdown("Escolha a competição e os times para calcular a probabilidade da IA.")

competicoes = ["Brasileirao", "Copa do Brasil", "Libertadores"]
competicao_sel = st.sidebar.selectbox("Selecione a Competição:", competicoes)

# Busca os times disponíveis na competição selecionada
query_times = f"""
    SELECT DISTINCT time 
    FROM (
        SELECT time_mandante AS time FROM partidas WHERE competicao = '{competicao_sel}'
        UNION
        SELECT time_visitante AS time FROM partidas WHERE competicao = '{competicao_sel}'
    ) ORDER BY time
"""
times_disponiveis = [row[0] for row in con.execute(query_times).fetchall()]

if not times_disponiveis:
    st.error("Nenhum time encontrado para a competição selecionada.")
    st.stop()

time_mandante = st.sidebar.selectbox("Time Mandante 🏠:", times_disponiveis, index=0)

# Garante que o visitante não seja o mesmo time
times_visitantes = [t for t in times_disponiveis if t != time_mandante]
time_visitante = st.sidebar.selectbox(
    "Time Visitante ✈️:", 
    times_visitantes, 
    index=min(1, len(times_visitantes) - 1)
)

st.sidebar.divider()
btn_simular = st.sidebar.button("🚀 Simular Partida", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# 4. Função para Extrair as Variables V3 do DuckDB
# -----------------------------------------------------------------------------
def obter_stats_modelo(con_db, time_m, time_v, comp):
    # Stats Mandante Geral (Últimos jogos)
    q_m_geral = f"""
        WITH hist AS (
            SELECT 
                data_jogo,
                CASE WHEN time_mandante = '{time_m}' THEN gols_mandante ELSE gols_visitante END as gp,
                CASE WHEN time_mandante = '{time_m}' THEN gols_visitante ELSE gols_mandante END as gc,
                CASE 
                    WHEN (time_mandante = '{time_m}' AND resultado = 'M') OR (time_visitante = '{time_m}' AND resultado = 'V') THEN 3
                    WHEN resultado = 'E' THEN 1 ELSE 0 
                END as pts
            FROM partidas
            WHERE time_mandante = '{time_m}' OR time_visitante = '{time_m}'
            ORDER BY data_jogo DESC
        )
        SELECT 
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 3)), 1.0) as pts_3j,
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 5)), 1.0) as pts_5j,
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 10)), 1.0) as pts_10j,
            COALESCE((SELECT AVG(gp - gc) FROM (SELECT gp, gc FROM hist LIMIT 5)), 0.0) as saldo_5j
    """
    
    # Stats Mandante Jogando em Casa
    q_m_casa = f"""
        SELECT 
            COALESCE(AVG(CASE WHEN resultado = 'M' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END), 1.0) as pts_casa,
            COALESCE(AVG(gols_mandante), 1.0) as gp_casa,
            COALESCE(AVG(gols_visitante), 1.0) as gc_casa
        FROM (SELECT * FROM partidas WHERE time_mandante = '{time_m}' ORDER BY data_jogo DESC LIMIT 5)
    """
    
    # Stats Visitante Geral
    q_v_geral = f"""
        WITH hist AS (
            SELECT 
                data_jogo,
                CASE WHEN time_visitante = '{time_v}' THEN gols_visitante ELSE gols_mandante END as gp,
                CASE WHEN time_visitante = '{time_v}' THEN gols_mandante ELSE gols_visitante END as gc,
                CASE 
                    WHEN (time_visitante = '{time_v}' AND resultado = 'V') OR (time_mandante = '{time_v}' AND resultado = 'M') THEN 3
                    WHEN resultado = 'E' THEN 1 ELSE 0 
                END as pts
            FROM partidas
            WHERE time_mandante = '{time_v}' OR time_visitante = '{time_v}'
            ORDER BY data_jogo DESC
        )
        SELECT 
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 3)), 1.0) as pts_3j,
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 5)), 1.0) as pts_5j,
            COALESCE((SELECT AVG(pts) FROM (SELECT pts FROM hist LIMIT 10)), 1.0) as pts_10j,
            COALESCE((SELECT AVG(gp - gc) FROM (SELECT gp, gc FROM hist LIMIT 5)), 0.0) as saldo_5j
    """
    
    # Stats Visitante Jogando Fora
    q_v_fora = f"""
        SELECT 
            COALESCE(AVG(CASE WHEN resultado = 'V' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END), 1.0) as pts_fora,
            COALESCE(AVG(gols_visitante), 1.0) as gp_fora,
            COALESCE(AVG(gols_mandante), 1.0) as gc_fora
        FROM (SELECT * FROM partidas WHERE time_visitante = '{time_v}' ORDER BY data_jogo DESC LIMIT 5)
    """

    m_g = con_db.execute(q_m_geral).fetchone()
    m_c = con_db.execute(q_m_casa).fetchone()
    v_g = con_db.execute(q_v_geral).fetchone()
    v_f = con_db.execute(q_v_fora).fetchone()

    # Dicionário com TODAS as variáveis do Modelo V3
    features_dict = {
        'mandante_descanso': 7,
        'visitante_descanso': 7,
        'diff_descanso': 0,
        'mandante_tendencia': m_g[0] - m_g[2],
        'visitante_tendencia': v_g[0] - v_g[2],
        'mandante_saldo_5j': m_g[3],
        'visitante_saldo_5j': v_g[3],
        'diff_saldo_5j': m_g[3] - v_g[3],
        'mandante_pts_casa': m_c[0],
        'visitante_pts_fora': v_f[0],
        'mandante_gp_casa': m_c[1],
        'mandante_gc_casa': m_c[2],
        'visitante_gp_fora': v_f[1],
        'visitante_gc_fora': v_f[2],
        'atq_mandante_vs_def_visitante': m_c[1] - v_f[2],
        'atq_visitante_vs_def_mandante': v_f[1] - m_c[2],
        'competicao_Brasileirao': 1 if comp == "Brasileirao" else 0,
        'competicao_Copa do Brasil': 1 if comp == "Copa do Brasil" else 0,
        'competicao_Libertadores': 1 if comp == "Libertadores" else 0
    }
    
    stats_visuais = {
        'm_pts_5j': m_g[1],
        'v_pts_5j': v_g[1],
        'm_pts_casa': m_c[0],
        'v_pts_fora': v_f[0],
        'm_gp_casa': m_c[1],
        'v_gp_fora': v_f[1]
    }
    
    return features_dict, stats_visuais

# -----------------------------------------------------------------------------
# 5. Dashboard Principal
# -----------------------------------------------------------------------------
st.title("⚽ Simulador Preditivo de Futebol IA (DuckDB + ML)")
st.markdown(f"**Confronto:** `{time_mandante}` vs `{time_visitante}` | **Competição:** `{competicao_sel}`")

features_input, stats_vis = obter_stats_modelo(con, time_mandante, time_visitante, competicao_sel)

# Exibição dos cards comparativos dos times
col_m, col_center, col_v = st.columns([2, 1, 2])

with col_m:
    st.subheader(f"🏠 {time_mandante}")
    st.metric("Aproveitamento Geral (Últimos 5j)", f"{stats_vis['m_pts_5j']:.2f} pts/jogo")
    st.metric("Desempenho em Casa", f"{stats_vis['m_pts_casa']:.2f} pts/jogo")
    st.metric("Média de Gols em Casa", f"{stats_vis['m_gp_casa']:.2f} gols/jogo")

with col_center:
    st.markdown("<h2 style='text-align: center; margin-top: 50px;'>VS</h2>", unsafe_allow_html=True)

with col_v:
    st.subheader(f"✈️ {time_visitante}")
    st.metric("Aproveitamento Geral (Últimos 5j)", f"{stats_vis['v_pts_5j']:.2f} pts/jogo")
    st.metric("Desempenho Fora de Casa", f"{stats_vis['v_pts_fora']:.2f} pts/jogo")
    st.metric("Média de Gols Fora", f"{stats_vis['v_gp_fora']:.2f} gols/jogo")

st.divider()

# -----------------------------------------------------------------------------
# 6. Cálculo da Previsão pela IA
# -----------------------------------------------------------------------------
st.subheader("🎯 Probabilidades Estimadas pelo Modelo V3")

# Alinha os dados exatamente na ordem exigida pelo modelo
df_input = pd.DataFrame([features_input])
for col in feature_names:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[feature_names]

# Calcula as probabilidades das 3 classes
probs = modelo.predict_proba(df_input)[0]
prob_empate = probs[0] * 100
prob_mandante = probs[1] * 100
prob_visitante = probs[2] * 100

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.metric(f"Vitória {time_mandante}", f"{prob_mandante:.1f}%")
    st.progress(int(prob_mandante))

with col_p2:
    st.metric("Empate", f"{prob_empate:.1f}%")
    st.progress(int(prob_empate))

with col_p3:
    st.metric(f"Vitória {time_visitante}", f"{prob_visitante:.1f}%")
    st.progress(int(prob_visitante))

st.divider()

# -----------------------------------------------------------------------------
# 7. Histórico de Confrontos Diretos (H2H)
# -----------------------------------------------------------------------------
st.subheader("📜 Retrospecto de Confrontos Diretos (H2H)")

query_h2h = f"""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN (time_mandante = '{time_mandante}' AND resultado = 'M') OR (time_visitante = '{time_mandante}' AND resultado = 'V') THEN 1 END) as vitorias_m,
        COUNT(CASE WHEN resultado = 'E' THEN 1 END) as empates,
        COUNT(CASE WHEN (time_mandante = '{time_visitante}' AND resultado = 'M') OR (time_visitante = '{time_visitante}' AND resultado = 'V') THEN 1 END) as vitorias_v
    FROM partidas
    WHERE (time_mandante = '{time_mandante}' AND time_visitante = '{time_visitante}')
       OR (time_mandante = '{time_visitante}' AND time_visitante = '{time_mandante}')
"""
h2h = con.execute(query_h2h).fetchone()

if h2h[0] > 0:
    st.write(f"Total de jogos entre as equipes registrados no banco: **{h2h[0]} partidas**")
    
    fig = go.Figure(data=[go.Pie(
        labels=[f'Vitórias {time_mandante}', 'Empates', f'Vitórias {time_visitante}'],
        values=[h2h[1], h2h[2], h2h[3]],
        hole=.4,
        marker_colors=['#2ecc71', '#f1c40f', '#e74c3c']
    )])
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Não há histórico de confrontos diretos anteriores no banco para estes dois times.")

con.close()