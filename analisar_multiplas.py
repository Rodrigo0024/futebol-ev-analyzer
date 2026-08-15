import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import poisson

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA E INICIALIZAÇÃO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gerador de Múltiplas +EV (4 Jogos)",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS E MODELO
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    if not os.path.exists('modelo_futebol_v2.pkl'):
        return None
    try:
        return joblib.load('modelo_futebol_v2.pkl')
    except Exception:
        return None

def carregar_metadados():
    if not os.path.exists('futebol.db'):
        return [], []
    try:
        con = duckdb.connect('futebol.db')
        times_m = con.execute("SELECT DISTINCT time_mandante FROM partidas WHERE time_mandante IS NOT NULL").df()['time_mandante'].tolist()
        times_v = con.execute("SELECT DISTINCT time_visitante FROM partidas WHERE time_visitante IS NOT NULL").df()['time_visitante'].tolist()
        comps = con.execute("SELECT DISTINCT competicao FROM partidas WHERE competicao IS NOT NULL").df()['competicao'].tolist()
        con.close()
        
        times = sorted(list(set([str(t).strip() for t in times_m + times_v if pd.notna(t) and str(t).strip() != ''])))
        competicoes = sorted(list(set([str(c).strip() for c in comps if pd.notna(c) and str(c).strip() != ''])))
        return times, competicoes
    except Exception:
        return [], []

dados_modelo = carregar_modelo()
lista_times, lista_competicoes = carregar_metadados()

if dados_modelo is None or not lista_times:
    st.error("⚠️ Certifique-se de que o 'modelo_futebol_v2.pkl' e o 'futebol.db' estão configurados!")
    st.stop()

modelo = dados_modelo['modelo']
features = dados_modelo['features']
elo_ratings = dados_modelo['elo_ratings']
le = dados_modelo['label_encoder']

# -----------------------------------------------------------------------------
# 3. FUNÇÃO DE PREDIÇÃO INDIVIDUAL
# -----------------------------------------------------------------------------
def obter_stats_time(time_nome, mando):
    con = duckdb.connect('futebol.db')
    if mando == 'CASA':
        query = """
            SELECT AVG(gols_mandante) as gp, 
                   AVG(CASE WHEN resultado = 'M' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END) as pts
            FROM (SELECT gols_mandante, resultado FROM partidas WHERE time_mandante = ? ORDER BY data_jogo DESC LIMIT 5)
        """
    else:
        query = """
            SELECT AVG(gols_visitante) as gp, 
                   AVG(CASE WHEN resultado = 'V' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END) as pts
            FROM (SELECT gols_visitante, resultado FROM partidas WHERE time_visitante = ? ORDER BY data_jogo DESC LIMIT 5)
        """
    res = con.execute(query, [time_nome]).df()
    con.close()
    
    gp = res['gp'].iloc[0] if not res['gp'].isna().all() else (1.3 if mando == 'CASA' else 1.0)
    pts = res['pts'].iloc[0] if not res['pts'].isna().all() else (1.4 if mando == 'CASA' else 1.0)
    return float(gp), float(pts)

def prever_partida(time_m, time_v, competicao):
    elo_m = elo_ratings.get(time_m, 1500.0)
    elo_v = elo_ratings.get(time_v, 1500.0)
    diff_elo = elo_m - elo_v

    m_gp_casa, m_pts_casa = obter_stats_time(time_m, 'CASA')
    v_gp_fora, v_pts_fora = obter_stats_time(time_v, 'FORA')

    lambda_m = max(0.4, (m_gp_casa + 1.1) / 2.0)
    lambda_v = max(0.3, (v_gp_fora + 1.2) / 2.0)

    g_arr = np.arange(7)
    pm = poisson.pmf(g_arr, lambda_m)
    pv = poisson.pmf(g_arr, lambda_v)
    joint = np.outer(pm, pv)

    poisson_pm = float(np.tril(joint, -1).sum())
    poisson_pe = float(np.trace(joint))
    poisson_pv = float(np.triu(joint, 1).sum())

    dados_jogo = {
        'diff_elo': diff_elo,
        'elo_mandante': elo_m,
        'elo_visitante': elo_v,
        'm_gp_casa': m_gp_casa,
        'm_pts_casa': m_pts_casa,
        'v_gp_fora': v_gp_fora,
        'v_pts_fora': v_pts_fora,
        'descanso_mandante': 7,
        'descanso_visitante': 7,
        'poisson_pm': poisson_pm,
        'poisson_pe': poisson_pe,
        'poisson_pv': poisson_pv
    }

    for col in features:
        if col.startswith('competicao_'):
            dados_jogo[col] = 1 if col == f'competicao_{competicao}' else 0

    df_input = pd.DataFrame([dados_jogo])
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features]

    probs = modelo.predict_proba(df_input)[0]
    mapa_prob = {cls: prob for cls, prob in zip(le.classes_, probs)}
    return mapa_prob

# -----------------------------------------------------------------------------
# 4. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🎯 Analisador de Múltiplas +EV (4 Jogos)")
st.caption("Avaliação de Valor Esperado e Dimensionamento de Stake em Bilhetes Combinados")

st.sidebar.header("💰 Gestão de Banca")
banca_total = st.sidebar.number_input("Banca Total (R$)", min_value=10.0, value=1000.0, step=50.0)
kelly_frac = st.sidebar.select_slider(
    "Fração de Kelly para Múltiplas",
    options=[0.05, 0.10, 0.20],
    value=0.05,
    format_func=lambda x: f"{x*100:.0f}% ({'Ultra Seguro' if x==0.05 else 'Moderado' if x==0.1 else 'Agressivo'})"
)

st.subheader("⚙️ Seleção das 4 Partidas e Escolha dos Palpites")

selecoes = []
col_a, col_b = st.columns(2)

for i in range(1, 5):
    col_target = col_a if i <= 2 else col_b
    with col_target:
        st.markdown(f"### ⚽ Jogo {i}")
        c1, c2 = st.columns(2)
        with c1:
            tm = st.selectbox(f"Mandante {i}", lista_times, index=min(i*2-2, len(lista_times)-1), key=f"m_{i}")
        with c2:
            tv = st.selectbox(f"Visitante {i}", lista_times, index=min(i*2-1, len(lista_times)-1), key=f"v_{i}")
        
        c3, c4 = st.columns(2)
        with c3:
            comp = st.selectbox(f"Competição {i}", lista_competicoes, key=f"c_{i}") if lista_competicoes else "Geral"
        with c4:
            palpite = st.selectbox(f"Seu Palpite {i}", ["Mandante (M)", "Empate (E)", "Visitante (V)"], key=f"p_{i}")
        
        odd_escolhida = st.number_input(f"Odd da Casa {i}", min_value=1.01, value=1.50, step=0.05, key=f"o_{i}")
        
        sigla_palpite = 'M' if 'Mandante' in palpite else ('E' if 'Empate' in palpite else 'V')
        
        selecoes.append({
            'jogo': f"{tm} x {tv}",
            'mandante': tm,
            'visitante': tv,
            'competicao': comp,
            'palpite': sigla_palpite,
            'palpite_nome': palpite,
            'odd': odd_escolhida
        })
        st.markdown("---")

# -----------------------------------------------------------------------------
# 5. CÁLCULO E RESULTADO DA MÚLTIPLA
# -----------------------------------------------------------------------------
if st.button("🚀 Calcular Múltipla de 4 Jogos", type="primary", use_container_width=True):
    prob_conjunta = 1.0
    odd_total_casa = 1.0
    detalhes_jogos = []

    for item in selecoes:
        mapa_p = prever_partida(item['mandante'], item['visitante'], item['competicao'])
        prob_ind = mapa_p[item['palpite']]
        
        prob_conjunta *= prob_ind
        odd_total_casa *= item['odd']
        
        detalhes_jogos.append({
            'Confronto': item['jogo'],
            'Palpite': item['palpite_nome'],
            'Prob. Calibrada': f"{prob_ind*100:.1f}%",
            'Odd Justa Ind.': f"{(1.0/prob_ind):.2f}",
            'Odd Casa': f"{item['odd']:.2f}"
        })

    odd_justa_combinada = 1.0 / prob_conjunta if prob_conjunta > 0 else 999.0
    ev_multipla = (prob_conjunta * odd_total_casa) - 1.0

    # Dimensionamento de Stake via Kelly Criterion
    if ev_multipla > 0 and odd_total_casa > 1.0:
        b = odd_total_casa - 1.0
        p = prob_conjunta
        q = 1.0 - p
        kelly_full = (b * p - q) / b
        stake_pct = max(0.0, kelly_full * kelly_frac)
        stake_reais = stake_pct * banca_total
    else:
        stake_pct = 0.0
        stake_reais = 0.0

    # Exibição dos resultados
    st.subheader("📋 Detalhamento Individual do Bilhete")
    st.dataframe(pd.DataFrame(detalhes_jogos), use_container_width=True, hide_index=True)

    st.subheader("📊 Resultado Final da Múltipla (+EV)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidade Conjunta", f"{prob_conjunta*100:.2f}%")
    m2.metric("Odd Justa Combinada", f"{odd_justa_combinada:.2f}")
    m3.metric("Odd Total da Casa", f"{odd_total_casa:.2f}")
    m4.metric("EV do Bilhete", f"{ev_multipla*100:+.2f}%", delta_color="normal")

    st.markdown("---")

    if ev_multipla > 0:
        st.success(f"🔥 **Múltipla com Valor Esperado Positivo (+EV de {ev_multipla*100:+.2f}%)!**")
        st.write(f"💵 **Sugestão de Aposta:** R$ {stake_reais:.2f} ({stake_pct*100:.2f}% da banca)")
        st.caption("Nota: Múltiplas exigem aportes pequenos para resistir à variância natural.")
    else:
        st.error(f"🔴 **Múltipla sem Valor Esperado (EV de {ev_multipla*100:+.2f}%)**")
        st.info("A Odd Total oferecida pela casa é menor do que a Odd Justa Calculada. Não recomendamos apostar neste bilhete.")