import duckdb
import pandas as pd
import numpy as np
import joblib
import time
from scipy.stats import poisson

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

tempo_inicio = time.time()

# 1. Conexão e carregamento de dados
con = duckdb.connect('futebol.db')

print("⚡ 1/5 - Carregando histórico cronológico...")

df_partidas = con.execute("""
    SELECT 
        competicao, data_jogo, time_mandante, time_visitante,
        gols_mandante, gols_visitante, resultado
    FROM partidas
    ORDER BY data_jogo ASC
""").df()

df_partidas['data_jogo'] = pd.to_datetime(df_partidas['data_jogo'])

# -----------------------------------------------------------------------------
# A. RATING ELO DINÂMICO
# -----------------------------------------------------------------------------
print("🏆 2/5 - Calculando Rating Elo Dinâmico...")

elo_ratings = {}
K_FACTOR = 32
HOME_ADVANTAGE = 65.0

def get_elo(time_nome):
    return elo_ratings.get(time_nome, 1500.0)

elo_m_list, elo_v_list = [], []

for row in df_partidas.itertuples():
    m, v = row.time_mandante, row.time_visitante
    r_m, r_v = get_elo(m), get_elo(v)
    
    elo_m_list.append(r_m)
    elo_v_list.append(r_v)
    
    e_m = 1.0 / (1.0 + 10.0 ** ((r_v - (r_m + HOME_ADVANTAGE)) / 400.0))
    res = row.resultado
    s_m = 1.0 if res == 'M' else (0.5 if res == 'E' else 0.0)
    
    elo_ratings[m] = r_m + K_FACTOR * (s_m - e_m)
    elo_ratings[v] = r_v + K_FACTOR * ((1.0 - s_m) - (1.0 - e_m))

df_partidas['elo_mandante'] = elo_m_list
df_partidas['elo_visitante'] = elo_v_list
df_partidas['diff_elo'] = df_partidas['elo_mandante'] - df_partidas['elo_visitante']

con.execute("CREATE OR REPLACE TEMPORARY TABLE partidas_elo AS SELECT * FROM df_partidas")

# -----------------------------------------------------------------------------
# B. ENGENHARIA DE RECURSOS (MANDO + FADIGA)
# -----------------------------------------------------------------------------
print("🎯 3/5 - Gerando métricas de Mando de Campo e Descanso...")

query_avancada = """
WITH historico_casa AS (
    SELECT data_jogo, time_mandante AS time, gols_mandante AS gp, gols_visitante AS gc,
           CASE WHEN resultado = 'M' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END AS pts
    FROM partidas_elo
),
historico_fora AS (
    SELECT data_jogo, time_visitante AS time, gols_visitante AS gp, gols_mandante AS gc,
           CASE WHEN resultado = 'V' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END AS pts
    FROM partidas_elo
),
stats_casa AS (
    SELECT 
        data_jogo, time,
        AVG(gp) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS gp_casa_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_casa_5j,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_m
    FROM historico_casa
),
stats_fora AS (
    SELECT 
        data_jogo, time,
        AVG(gp) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS gp_fora_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_fora_5j,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_v
    FROM historico_fora
)
SELECT 
    p.competicao, p.data_jogo, p.time_mandante, p.time_visitante,
    p.diff_elo, p.elo_mandante, p.elo_visitante,
    COALESCE(sc.gp_casa_5j, 1.3) AS m_gp_casa,
    COALESCE(sc.pts_casa_5j, 1.4) AS m_pts_casa,
    COALESCE(sf.gp_fora_5j, 1.0) AS v_gp_fora,
    COALESCE(sf.pts_fora_5j, 1.0) AS v_pts_fora,
    COALESCE(sc.descanso_m, 7) AS descanso_mandante,
    COALESCE(sf.descanso_v, 7) AS descanso_visitante,
    p.resultado
FROM partidas_elo p
LEFT JOIN stats_casa sc ON p.data_jogo = sc.data_jogo AND p.time_mandante = sc.time
LEFT JOIN stats_fora sf ON p.data_jogo = sf.data_jogo AND p.time_visitante = sf.time
ORDER BY p.data_jogo ASC;
"""

df_fatos = con.execute(query_avancada).df()
con.close()

# -----------------------------------------------------------------------------
# C. MATRIZ DE POISSON VETORIZADA
# -----------------------------------------------------------------------------
print("⚡ 4/5 - Vetorizando probabilidades de Poisson...")

lambda_m = np.maximum(0.4, (df_fatos['m_gp_casa'].values + 1.1) / 2.0)
lambda_v = np.maximum(0.3, (df_fatos['v_gp_fora'].values + 1.2) / 2.0)

g_arr = np.arange(7)
pm = poisson.pmf(g_arr[None, :], lambda_m[:, None])
pv = poisson.pmf(g_arr[None, :], lambda_v[:, None])

joint = pm[:, :, None] * pv[:, None, :]
gm_grid, gv_grid = np.meshgrid(g_arr, g_arr, indexing='ij')

df_fatos['poisson_pm'] = joint[:, gm_grid > gv_grid].sum(axis=1)
df_fatos['poisson_pe'] = joint[:, gm_grid == gv_grid].sum(axis=1)
df_fatos['poisson_pv'] = joint[:, gm_grid < gv_grid].sum(axis=1)

# -----------------------------------------------------------------------------
# D. ENSEMBLE COM RÓTULOS NUMÉRICOS E PESOS CORRIGIDOS
# -----------------------------------------------------------------------------
print("🤝 5/5 - Treinando Ensemble Votante (Gradient Boosting + Random Forest + Regressão Logística)...")

df_fatos.fillna(0, inplace=True)
df_fatos = pd.get_dummies(df_fatos, columns=['competicao'], drop_first=False)

feature_cols = [c for c in df_fatos.columns if c not in ['data_jogo', 'time_mandante', 'time_visitante', 'resultado']]

X = df_fatos[feature_cols]
y_raw = df_fatos['resultado']

# Codificação explicita dos rótulos ('E'->0, 'M'->1, 'V'->2)
le = LabelEncoder()
y = le.fit_transform(y_raw)

split_idx = int(len(df_fatos) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Mapeamento do dicionário de pesos para os rótulos inteiros [0, 1, 2]
pesos_texto = {'E': 1.25, 'M': 1.0, 'V': 1.15}
custom_weights_int = {i: pesos_texto[cls_name] for i, cls_name in enumerate(le.classes_)}

m1_gb = HistGradientBoostingClassifier(
    max_iter=180, learning_rate=0.03, max_depth=4,
    min_samples_leaf=20, l2_regularization=3.0,
    class_weight=custom_weights_int, random_state=42
)

m2_rf = RandomForestClassifier(
    n_estimators=150, max_depth=6, min_samples_leaf=15,
    class_weight=custom_weights_int, random_state=42, n_jobs=-1
)

m3_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight=custom_weights_int, C=0.5, random_state=42)
)

# Ensemble Votante por Média de Probabilidades (Soft Voting)
ensemble = VotingClassifier(
    estimators=[
        ('gb', m1_gb),
        ('rf', m2_rf),
        ('lr', m3_lr)
    ],
    voting='soft',
    weights=[2, 1, 1]
)

ensemble.fit(X_train, y_train)

# Predição e Decodificação de volta para texto ('E', 'M', 'V')
y_pred_num = ensemble.predict(X_test)
y_pred = le.inverse_transform(y_pred_num)
y_test_labels = le.inverse_transform(y_test)

acuracia = accuracy_score(y_test_labels, y_pred)
macro_f1 = f1_score(y_test_labels, y_pred, average='macro')

tempo_total = time.time() - tempo_inicio

print(f"\n✅ Ensemble treinado em {tempo_total:.2f} segundos!")
print(f"🎯 Acurácia Geral do Ensemble: {acuracia * 100:.2f}%")
print(f"⚖️ Macro F1-Score: {macro_f1 * 100:.2f}%\n")

print("--- 📋 Relatório de Classificação ---")
print(classification_report(y_test_labels, y_pred, target_names=['Empate (E)', 'Mandante (M)', 'Visitante (V)']))

print("--- 🧱 Matriz de Confusão ---")
cm = confusion_matrix(y_test_labels, y_pred, labels=['E', 'M', 'V'])
print(pd.DataFrame(cm, index=['Real E', 'Real M', 'Real V'], columns=['Prev E', 'Prev M', 'Prev V']))

joblib.dump({
    'modelo': ensemble, 
    'features': feature_cols,
    'elo_ratings': elo_ratings,
    'label_encoder': le
}, 'modelo_futebol_v2.pkl')

print("\n💾 Ensemble V5.2 salvo com sucesso em 'modelo_futebol_v2.pkl'!")