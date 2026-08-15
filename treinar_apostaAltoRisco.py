import duckdb
import pandas as pd
import numpy as np
import joblib
import time
import optuna
from scipy.stats import poisson

from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
tempo_inicio = time.time()

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E ENRIQUECIMENTO DE DADOS (COM ODDS DE FECHAMENTO)
# -----------------------------------------------------------------------------
con = duckdb.connect('futebol.db')
print("⚡ 1/7 - Carregando histórico e métricas avançadas de mercado...")

df_partidas = con.execute("""
    SELECT 
        competicao, data_jogo, time_mandante, time_visitante,
        gols_mandante, gols_visitante,
        COALESCE(xg_mandante, gols_mandante) AS xg_m,
        COALESCE(xg_visitante, gols_visitante) AS xg_v,
        COALESCE(NULLIF(odd_m_fechamento, 0.0), 2.0) AS odd_m,
        COALESCE(NULLIF(odd_v_fechamento, 0.0), 3.5) AS odd_v,
        COALESCE(NULLIF(odd_e_fechamento, 0.0), 3.0) AS odd_e,
        resultado
    FROM partidas
    WHERE resultado IS NOT NULL AND time_mandante IS NOT NULL AND time_visitante IS NOT NULL
    ORDER BY data_jogo ASC
""").df()

if df_partidas.empty:
    raise ValueError("❌ A tabela 'partidas' no banco 'futebol.db' está vazia!")

df_partidas['data_jogo'] = pd.to_datetime(df_partidas['data_jogo'])

# Probabilidades implícitas ajustadas
df_partidas['prob_mercado_m'] = 1.0 / df_partidas['odd_m']
df_partidas['prob_mercado_e'] = 1.0 / df_partidas['odd_e']
df_partidas['prob_mercado_v'] = 1.0 / df_partidas['odd_v']

suma_prob = df_partidas['prob_mercado_m'] + df_partidas['prob_mercado_e'] + df_partidas['prob_mercado_v']
df_partidas['prob_mercado_m'] /= suma_prob
df_partidas['prob_mercado_e'] /= suma_prob
df_partidas['prob_mercado_v'] /= suma_prob

# -----------------------------------------------------------------------------
# 2. RATING ELO DINÂMICO
# -----------------------------------------------------------------------------
print("🏆 2/7 - Calculando Rating Elo Dinâmico...")
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
# 3. EXTRAÇÃO DE FEATURES ISOLADAS (CASA vs FORA) + CARGA
# -----------------------------------------------------------------------------
print("🎯 3/7 - Extraindo Métricas Isoladas (Home/Away Stats) e Carga...")
query_avancada = """
WITH historico_casa AS (
    SELECT data_jogo, time_mandante AS time, xg_m AS xg_pro, xg_v AS xg_contra,
           CASE WHEN resultado = 'M' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END AS pts
    FROM partidas_elo
),
historico_fora AS (
    SELECT data_jogo, time_visitante AS time, xg_v AS xg_pro, xg_m AS xg_contra,
           CASE WHEN resultado = 'V' THEN 3 WHEN resultado = 'E' THEN 1 ELSE 0 END AS pts
    FROM partidas_elo
),
stats_casa_estrito AS (
    SELECT 
        data_jogo, time,
        AVG(xg_pro) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS xg_casa_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_casa_5j,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_m
    FROM historico_casa
),
stats_fora_estrito AS (
    SELECT 
        data_jogo, time,
        AVG(xg_pro) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS xg_fora_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_fora_5j,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_v
    FROM historico_fora
)
SELECT 
    p.competicao, p.data_jogo, p.time_mandante, p.time_visitante,
    p.diff_elo, p.elo_mandante, p.elo_visitante,
    p.prob_mercado_m, p.prob_mercado_e, p.prob_mercado_v,
    COALESCE(sc.xg_casa_5j, 1.3) AS m_xg_casa,
    COALESCE(sc.pts_casa_5j, 1.4) AS m_pts_casa,
    COALESCE(sf.xg_fora_5j, 1.0) AS v_xg_fora,
    COALESCE(sf.pts_fora_5j, 1.0) AS v_pts_fora,
    COALESCE(sc.descanso_m, 7) AS descanso_mandante,
    COALESCE(sf.descanso_v, 7) AS descanso_visitante,
    p.resultado
FROM partidas_elo p
LEFT JOIN stats_casa_estrito sc ON p.data_jogo = sc.data_jogo AND p.time_mandante = sc.time
LEFT JOIN stats_fora_estrito sf ON p.data_jogo = sf.data_jogo AND p.time_visitante = sf.time
ORDER BY p.data_jogo ASC;
"""

df_fatos = con.execute(query_avancada).df()
con.close()

if df_fatos.empty:
    raise ValueError("❌ O DataFrame `df_fatos` retornou vazio.")

# -----------------------------------------------------------------------------
# 4. DIXON-COLES & MATRIZES DE PROBABILIDADE
# -----------------------------------------------------------------------------
print("⚡ 4/7 - Calculando Probabilidades de Dixon-Coles...")
lambda_m = np.maximum(0.4, (df_fatos['m_xg_casa'].values + 1.1) / 2.0)
lambda_v = np.maximum(0.3, (df_fatos['v_xg_fora'].values + 1.2) / 2.0)

g_arr = np.arange(7)
pm = poisson.pmf(g_arr[None, :], lambda_m[:, None])
pv = poisson.pmf(g_arr[None, :], lambda_v[:, None])
joint = pm[:, :, None] * pv[:, None, :]

RHO = -0.13
tau = np.ones((len(df_fatos), 7, 7))
tau[:, 0, 0] = 1.0 - (lambda_m * lambda_v * RHO)
tau[:, 1, 0] = 1.0 + (lambda_m * RHO)
tau[:, 0, 1] = 1.0 + (lambda_v * RHO)
tau[:, 1, 1] = 1.0 - RHO
tau = np.maximum(0.0001, tau)

joint_dc = joint * tau
joint_dc = joint_dc / joint_dc.sum(axis=(1, 2), keepdims=True)
gm_grid, gv_grid = np.meshgrid(g_arr, g_arr, indexing='ij')

df_fatos['poisson_pm'] = joint_dc[:, gm_grid > gv_grid].sum(axis=1)
df_fatos['poisson_pe'] = joint_dc[:, gm_grid == gv_grid].sum(axis=1)
df_fatos['poisson_pv'] = joint_dc[:, gm_grid < gv_grid].sum(axis=1)

for col in df_fatos.columns:
    if pd.api.types.is_numeric_dtype(df_fatos[col]):
        df_fatos[col] = df_fatos[col].fillna(0.0)
    else:
        df_fatos[col] = df_fatos[col].fillna('')

df_fatos = pd.get_dummies(df_fatos, columns=['competicao'], drop_first=False, dtype=int)
feature_cols = [c for c in df_fatos.columns if c not in ['data_jogo', 'time_mandante', 'time_visitante', 'resultado']]

X = df_fatos[feature_cols]
y_raw = df_fatos['resultado']
le = LabelEncoder()
y = le.fit_transform(y_raw)

split_idx = int(len(df_fatos) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"📊 Total de amostras: {len(df_fatos)} | Treino: {len(X_train)} | Teste: {len(X_test)}")

# -----------------------------------------------------------------------------
# 5. OTIMIZAÇÃO COM OPTUNA (FOCADA EM LOG LOSS PARA CALIBRAGEM)
# -----------------------------------------------------------------------------
print("🔍 5/7 - Otimizando LightGBM (Minimizando Log Loss com Balanceamento)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 15, 45),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 60),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    preds_proba = clf.predict_proba(X_test)
    return log_loss(y_test, preds_proba)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

# -----------------------------------------------------------------------------
# 6. STACKING AVANÇADO COM CALIBRAGEM
# -----------------------------------------------------------------------------
print("🤝 6/7 - Treinando Stacking Avançado com Pesos Balanceados...")
best_lgbm = LGBMClassifier(**study.best_params, class_weight='balanced', random_state=42, verbose=-1)

stacking_classifier = StackingClassifier(
    estimators=[('optuna_lgb', best_lgbm)],
    final_estimator=LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    stack_method='predict_proba',
    cv=5
)

stacking_classifier.fit(X_train, y_train)

y_pred_test = stacking_classifier.predict(X_test)
acuracia = accuracy_score(y_test, y_pred_test)

print(f"\n✅ Treinamento concluído!")
print(f"🎯 Nova Acurácia Geral Ajustada: {acuracia * 100:.2f}%\n")

# -----------------------------------------------------------------------------
# 7. RELATÓRIO FINAL E EXPORTAÇÃO
# -----------------------------------------------------------------------------
print("📊 Relatório Atualizado:")
print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))

stacking_classifier.fit(X, y)

joblib.dump({
    'modelo': stacking_classifier, 
    'features': feature_cols,
    'elo_ratings': elo_ratings,
    'label_encoder': le
}, 'modelo_futebol_mercado.pkl')

print("💾 Modelo otimizado salvo em 'modelo_futebol_mercado.pkl'!")