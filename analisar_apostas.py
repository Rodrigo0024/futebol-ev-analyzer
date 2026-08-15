import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import joblib
import subprocess
import os
from scipy.stats import poisson

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO E INICIALIZAÇÃO SEGURA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisador EV+ Futebol",
    page_icon="⚽",
    layout="wide"
)

# Inicialização global para GARANTIR que nunca ocorra NameError
lista_times = []
lista_competicoes = []

# -----------------------------------------------------------------------------
# 2. FUNÇÕES DE CARREGAMENTO (BANCO E MODELO)
# -----------------------------------------------------------------------------
def carregar_metadados():
    if not os.path.exists('futebol.db'):
        return [], []
    try:
        con = duckdb.connect('futebol.db', read_only=True)
        times_m = con.execute("SELECT DISTINCT time_mandante FROM partidas WHERE time_mandante IS NOT NULL").df()['time_mandante'].tolist()
        times_v = con.execute("SELECT DISTINCT time_visitante FROM partidas WHERE time_visitante IS NOT NULL").df()['time_visitante'].tolist()
        comps = con.execute("SELECT DISTINCT competicao FROM partidas WHERE competicao IS NOT NULL").df()['competicao'].tolist()
        con.close()
        
        times = sorted(list(set([str(t).strip() for t in times_m + times_v if pd.notna(t) and str(t).strip() != ''])))
        competicoes = sorted(list(set([str(c).strip() for c in comps if pd.notna(c) and str(c).strip() != ''])))
        return times, competicoes
    except Exception as e:
        st.error(f"⚠️ Erro ao consultar o banco de dados `futebol.db`: {e}")
        return [], []

def carregar_modelo():
    if not os.path.exists('modelo_futebol_v2.pkl'):
        return None
    try:
        return joblib.load('modelo_futebol_v2.pkl')
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar 'modelo_futebol_v2.pkl': {e}")
        return None

# Carregamento dos dados
dados_modelo = carregar_modelo()
lista_times, lista_competicoes = carregar_metadados()

# -----------------------------------------------------------------------------
# 3. VALIDAÇÃO DE PRÉ-REQUISITOS (BLOQUEIO AMIGÁVEL)
# -----------------------------------------------------------------------------
if dados_modelo is None:
    st.error("⚠️ O arquivo de modelo `modelo_futebol_v2.pkl` não foi encontrado!")
    st.info("👉 **Solução:** Execute no seu terminal: `python treinar_modelo.py` para gerar o modelo.")
    st.stop()

if not lista_times:
    st.error("⚠️ Nenhum time foi encontrado no banco de dados `futebol.db`!")
    st.info("👉 Certifique-se de que o banco `futebol.db` possui a tabela `partidas` populada.")
    st.stop()

modelo = dados_modelo['modelo']
features = dados_modelo['features']
elo_ratings = dados_modelo['elo_ratings']
le = dados_modelo['label_encoder']

# -----------------------------------------------------------------------------
# 4. FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
def obter_stats_time(time_nome, mando):
    con = duckdb.connect('futebol.db', read_only=True)
    if mando == 'CASA':
        query = """
            SELECT AVG(gols_mandante) as gp, 
                   AVG(CASE WHEN resultado = 'M' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END) as pts
            FROM (
                SELECT gols_mandante, resultado 
                FROM partidas 
                WHERE time_mandante = ? 
                ORDER BY data_jogo DESC LIMIT 5
            )
        """
    else:
        query = """
            SELECT AVG(gols_visitante) as gp, 
                   AVG(CASE WHEN resultado = 'V' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END) as pts
            FROM (
                SELECT gols_visitante, resultado 
                FROM partidas 
                WHERE time_visitante = ? 
                ORDER BY data_jogo DESC LIMIT 5
            )
        """
    res = con.execute(query, [time_nome]).df()
    con.close()
    
    gp = res['gp'].iloc[0] if not res['gp'].isna().all() else (1.3 if mando == 'CASA' else 1.0)
    pts = res['pts'].iloc[0] if not res['pts'].isna().all() else (1.4 if mando == 'CASA' else 1.0)
    return float(gp), float(pts)

def calcular_ev_e_kelly(probabilidade, odd_casa, banca_total, fracao_kelly=0.25):
    odd_justa = 1.0 / probabilidade if probabilidade > 0 else 999.0
    ev = (probabilidade * odd_casa) - 1.0
    
    if ev > 0 and odd_casa > 1.0:
        b = odd_casa - 1.0
        p = probabilidade
        q = 1.0 - p
        kelly_full = (b * p - q) / b
        stake_pct = max(0.0, kelly_full * fracao_kelly)
        stake_reais = stake_pct * banca_total
    else:
        stake_pct = 0.0
        stake_reais = 0.0
        
    return odd_justa, ev * 100.0, stake_pct * 100.0, stake_reais

# -----------------------------------------------------------------------------
# 5. BARRA LATERAL (BANCA, ATUALIZAÇÃO, EXPLICAÇÃO EV E LINKS)
# -----------------------------------------------------------------------------
st.sidebar.header("💰 Gestão de Banca")
banca_total = st.sidebar.number_input("Banca Total (R$)", min_value=10.0, value=1000.0, step=50.0)
fracao_kelly = st.sidebar.select_slider(
    "Conservadorismo (Kelly)",
    options=[0.10, 0.25, 0.50, 1.00],
    value=0.25,
    format_func=lambda x: f"{x*100:.0f}% ({'1/4 Kelly' if x==0.25 else 'Agressivo' if x==0.5 else 'Muito Seguro' if x==0.1 else 'Full Kelly'})"
)

# -----------------------------------------------------------------------------
# Explicação do Valor Esperado (+EV) na barra lateral
# -----------------------------------------------------------------------------
with st.sidebar.expander("📚 O que é o Valor Esperado (+EV)?"):
    st.markdown("""
    O **Valor Esperado (+EV)** mede se uma aposta é lucrativa a longo prazo com base na probabilidade real do evento acontecer em comparação com a cotação (Odd) oferecida pela casa.

    * **Fórmula:** $\\text{EV} = (\\text{Probabilidade} \\times \\text{Odd}) - 1$
    * **O que significa?**
      * **EV Positivo (> 0%):** A Odd da casa está pagando mais do que deveria pelo risco real calculado pelo modelo. Significa que a aposta tem **valor matemático**.
      * **EV Negativo (≤ 0%):** A Odd é baixa demais para o risco. A longo prazo, apostas assim geram prejuízo.
    * **Critério de Kelly:** Utilizado para definir o tamanho ideal (stake) que você deve investir da sua banca para maximizar o lucro minimizando o risco de quebra.
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Atualização de Dados")

if st.sidebar.button("Fetch Novos Resultados", use_container_width=True):
    with st.spinner("1/2 - Buscando resultados recentes..."):
        res_fetch = subprocess.run(["python", "atualizar_soccerdata.py"], capture_output=True, text=True)
        st.sidebar.caption("Log da Atualização:")
        st.sidebar.code(res_fetch.stdout if res_fetch.stdout else res_fetch.stderr)

    with st.spinner("2/2 - Recalibrando Rating Elo e Modelo..."):
        subprocess.run(["python", "treinar_modelo.py"], capture_output=True, text=True)

    st.sidebar.success("🎉 Atualizado!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Links Úteis")
st.sidebar.link_button("🌐 Ver Repositório / Projeto", "https://github.com/seu-usuario/seu-repositorio", use_container_width=True)

# -----------------------------------------------------------------------------
# 6. INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.title("⚽ Analisador de Apostas Esportivas (+EV)")
st.caption("Avaliador de Valor Esperado e Dimensionador de Stakes via Machine Learning Calibrado")

st.subheader("🎯 Configure o Confronto")

col1, col2, col3 = st.columns(3)

with col1:
    time_m = st.selectbox("Time Mandante", lista_times, index=0)
    odd_m = st.number_input("Odd Mandante", min_value=1.01, value=1.90, step=0.05)

with col2:
    idx_v = 1 if len(lista_times) > 1 else 0
    time_v = st.selectbox("Time Visitante", lista_times, index=idx_v)
    odd_v = st.number_input("Odd Visitante", min_value=1.01, value=4.20, step=0.05)

with col3:
    idx_c = 0 if lista_competicoes else None
    competicao_sel = st.selectbox("Competição", lista_competicoes, index=idx_c) if lista_competicoes else "Geral"
    odd_e = st.number_input("Odd Empate (E)", min_value=1.01, value=3.60, step=0.05)

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. ANÁLISE DE CONFRONTO
# -----------------------------------------------------------------------------
if st.button("🚀 Analisar Oportunidades de Valor", type="primary", use_container_width=True):
    if time_m == time_v:
        st.warning("⚠️ O time mandante e o visitante não podem ser iguais.")
        st.stop()

    with st.spinner("Calculando Probabilidades Calibradas..."):
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
                dados_jogo[col] = 1 if col == f'competicao_{competicao_sel}' else 0

        df_input = pd.DataFrame([dados_jogo])
        for col in features:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[features]

        probs = modelo.predict_proba(df_input)[0]
        mapa_prob = {cls: prob for cls, prob in zip(le.classes_, probs)}

        prob_e, prob_m, prob_v = mapa_prob['E'], mapa_prob['M'], mapa_prob['V']

    st.subheader("📊 Diagnóstico do Confronto")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Elo {time_m}", f"{elo_m:.0f}")
    m2.metric(f"Elo {time_v}", f"{elo_v:.0f}")
    m3.metric("Diferença de Elo", f"{diff_elo:+.0f}")
    m4.metric("Poisson Mandante/Visitante", f"{poisson_pm*100:.1f}% / {poisson_pv*100:.1f}%")

    st.markdown("---")
    st.subheader("📋 Relatório de Valor Esperado (+EV)")

    mercados = [
        (f"Mandante ({time_m})", prob_m, odd_m),
        ("Empate (E)", prob_e, odd_e),
        (f"Visitante ({time_v})", prob_v, odd_v)
    ]

    tabela_resultados = []
    oportunidades = []

    for nome, prob, odd_casa in mercados:
        odd_justa, ev_pct, stake_pct, stake_reais = calcular_ev_e_kelly(prob, odd_casa, banca_total, fracao_kelly)
        
        status = "🟢 APOSTAR" if ev_pct > 0 else "🔴 Sem Valor"
        
        if ev_pct > 0:
            oportunidades.append({
                'nome': nome,
                'ev': ev_pct,
                'stake_reais': stake_reais,
                'stake_pct': stake_pct
            })

        tabela_resultados.append({
            'Mercado': nome,
            'Prob. Calibrada': f"{prob*100:.1f}%",
            'Odd Justa': f"{odd_justa:.2f}",
            'Odd Casa': f"{odd_casa:.2f}",
            'EV (%)': f"{ev_pct:+.2f}%",
            'Sugestão de Aposta': f"R$ {stake_reais:.2f} ({stake_pct:.1f}%)" if ev_pct > 0 else "R$ 0.00",
            'Status': status
        })

    df_res = pd.DataFrame(tabela_resultados)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    if oportunidades:
        st.success(f"🔥 **{len(oportunidades)} Oportunidade(s) com Valor Esperado Positivo (+EV) Encontrada(s)!**")
        for op in oportunidades:
            st.write(f"👉 **{op['nome']}**: Apostar **R$ {op['stake_reais']:.2f}** ({op['stake_pct']:.1f}% da banca) | Retorno Esperado: **{op['ev']:+.1f}%**")
    else:
        st.info("⚠️ Nenhuma entrada com Valor Esperado Positivo (+EV) neste confronto com las Odds fornecidas.")