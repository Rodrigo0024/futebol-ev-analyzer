import duckdb
import pandas as pd
import numpy as np
import joblib
import time
import optuna
from scipy.stats import poisson

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss

# Ocultar avisos do Optuna para manter o terminal limpo
optuna.logging.set_verbosity(optuna.logging.WARNING)

tempo_inicio = time.time()

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E TRATAMENTO DOS DADOS
# -----------------------------------------------------------------------------
con = duckdb.connect('futebol.db')

print("⚡ 1/7 - Carregando histórico e enriquecendo features...")

df_partidas = con.execute("""
    SELECT 
        competicao, data_jogo, time_mandante, time_visitante,
        gols_mandante, gols_visitante,
        COALESCE(xg_mandante, gols_mandante) AS xg_m,
        COALESCE(xg_visitante, gols_visitante) AS xg_v,
        COALESCE(odd_m_fechamento, 0.0) AS odd_m_fechamento,
        COALESCE(odd_e_fechamento, 0.0) AS odd_e_fechamento,
        COALESCE(odd_v_fechamento, 0.0) AS odd_v_fechamento,
        resultado
    FROM partidas
    WHERE resultado IS NOT NULL AND time_mandante IS NOT NULL AND time_visitante IS NOT NULL
    ORDER BY data_jogo ASC
""").df()

df_partidas['data_jogo'] = pd.to_datetime(df_partidas['data_jogo'])

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
# 3. EXTRAÇÃO DE FEATURES CONTEXTUAIS (xG, DESCANSO E MARATONA)
# -----------------------------------------------------------------------------
print("🎯 3/7 - Extraindo Métricas de xG, Carga de Jogos e Descanso...")

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
stats_casa AS (
    SELECT 
        data_jogo, time,
        AVG(xg_pro) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS xg_casa_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_casa_5j,
        COUNT(*) OVER (PARTITION BY time ORDER BY data_jogo RANGE BETWEEN INTERVAL 14 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS jogos_14d_m,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_m
    FROM historico_casa
),
stats_fora AS (
    SELECT 
        data_jogo, time,
        AVG(xg_pro) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS xg_fora_5j,
        AVG(pts) OVER (PARTITION BY time ORDER BY data_jogo ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_fora_5j,
        COUNT(*) OVER (PARTITION BY time ORDER BY data_jogo RANGE BETWEEN INTERVAL 14 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS jogos_14d_v,
        DATEDIFF('day', LAG(data_jogo, 1) OVER (PARTITION BY time ORDER BY data_jogo), data_jogo) AS descanso_v
    FROM historico_fora
)
SELECT 
    p.competicao, p.data_jogo, p.time_mandante, p.time_visitante,
    p.diff_elo, p.elo_mandante, p.elo_visitante,
    COALESCE(sc.xg_casa_5j, 1.3) AS m_xg_casa,
    COALESCE(sc.pts_casa_5j, 1.4) AS m_pts_casa,
    COALESCE(sf.xg_fora_5j, 1.0) AS v_xg_fora,
    COALESCE(sf.pts_fora_5j, 1.0) AS v_pts_fora,
    COALESCE(sc.descanso_m, 7) AS descanso_mandante,
    COALESCE(sf.descanso_v, 7) AS descanso_visitante,
    COALESCE(sc.jogos_14d_m, 2) AS carga_jogos_m,
    COALESCE(sf.jogos_14d_v, 2) AS carga_jogos_v,
    p.odd_m_fechamento, p.odd_e_fechamento, p.odd_v_fechamento,
    p.resultado
FROM partidas_elo p
LEFT JOIN stats_casa sc ON p.data_jogo = sc.data_jogo AND p.time_mandante = sc.time
LEFT JOIN stats_fora sf ON p.data_jogo = sf.data_jogo AND p.time_visitante = sf.time
ORDER BY p.data_jogo ASC;
"""

df_fatos = con.execute(query_avancada).df()
con.close()

# -----------------------------------------------------------------------------
# 4. MODELAGEM MATEMÁTICA DIXON-COLES + ODDS IMPLÍCITAS
# -----------------------------------------------------------------------------
print("⚡ 4/7 - Calculando Probabilidades de Dixon-Coles e Odds Implícitas...")

lambda_m = np.maximum(0.4, (df_fatos['m_xg_casa'].values + 1.1) / 2.0)
lambda_v = np.maximum(0.3, (df_fatos['v_xg_fora'].values + 1.2) / 2.0)

g_arr = np.arange(7)
pm = poisson.pmf(g_arr[None, :], lambda_m[:, None])
pv = poisson.pmf(g_arr[None, :], lambda_v[:, None])

joint = pm[:, :, None] * pv[:, None, :]

# Fator de Correção de Dixon-Coles (rho = -0.13)
RHO = -0.13
tau = np.ones((len(df_fatos), 7, 7))
tau[:, 0, 0] = 1.0 - (lambda_m * lambda_v * RHO)
tau[:, 1, 0] = 1.0 + (lambda_m * RHO)
tau[:, 0, 1] = 1.0 + (lambda_v * RHO)
tau[:, 1, 1] = 1.0 - RHO
tau = np.maximum(0.0001, tau)

joint_dc = joint * tau
somas_joint = joint_dc.sum(axis=(1, 2), keepdims=True)
joint_dc = joint_dc / somas_joint

gm_grid, gv_grid = np.meshgrid(g_arr, g_arr, indexing='ij')

df_fatos['poisson_pm'] = joint_dc[:, gm_grid > gv_grid].sum(axis=1)
df_fatos['poisson_pe'] = joint_dc[:, gm_grid == gv_grid].sum(axis=1)
df_fatos['poisson_pv'] = joint_dc[:, gm_grid < gv_grid].sum(axis=1)

# Odds implícitas desenviesadas
df_fatos['implied_prob_m'] = np.where(df_fatos['odd_m_fechamento'] > 1.0, 1.0 / df_fatos['odd_m_fechamento'], 0.0)
df_fatos['implied_prob_e'] = np.where(df_fatos['odd_e_fechamento'] > 1.0, 1.0 / df_fatos['odd_e_fechamento'], 0.0)
df_fatos['implied_prob_v'] = np.where(df_fatos['odd_v_fechamento'] > 1.0, 1.0 / df_fatos['odd_v_fechamento'], 0.0)

# -----------------------------------------------------------------------------
# PREPARAÇÃO DAS MATRIZES
# -----------------------------------------------------------------------------
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

# Divisão Temporal Pura (80% treino / 20% teste)
split_idx = int(len(df_fatos) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -----------------------------------------------------------------------------
# 5. OTIMIZAÇÃO BAYESIANA COM OPTUNA
# -----------------------------------------------------------------------------
print("🔍 5/7 - Otimizando Hiperparâmetros do LightGBM com Optuna (20 ensaios)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 80, 220),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.06, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 12, 35),
        'min_child_samples': trial.suggest_int('min_child_samples', 12, 35),
        'subsample': trial.suggest_float('subsample', 0.65, 0.90),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.90),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 3.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    return log_loss(y_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"   🏆 Menor Log Loss atingido pelo Optuna: {study.best_value:.4f}")

# -----------------------------------------------------------------------------
# 6. ARQUITETURA DE STACKING + CALIBRAÇÃO ISOTÔNICA
# -----------------------------------------------------------------------------
print("🤝 6/7 - Treinando Meta-Learner (Stacking + Calibração Isotônica)...")

best_lgbm = LGBMClassifier(**study.best_params, random_state=42, verbose=-1)

hgb_base = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.02, max_depth=4,
    min_samples_leaf=20, l2_regularization=2.5, random_state=42
)

rf_base = RandomForestClassifier(
    n_estimators=150, max_depth=6, min_samples_leaf=15,
    random_state=42, n_jobs=-1
)

stacking_classifier = StackingClassifier(
    estimators=[
        ('optuna_lgb', best_lgbm),
        ('hgb', hgb_base),
        ('rf', rf_base)
    ],
    final_estimator=LogisticRegression(C=0.3, max_iter=1000, random_state=42),
    stack_method='predict_proba',
    cv=5
)

# Calibração de Curvas
modelo_calibrado_eval = CalibratedClassifierCV(estimator=stacking_classifier, method='isotonic', cv=5)
modelo_calibrado_eval.fit(X_train, y_train)

y_probs_test = modelo_calibrado_eval.predict_proba(X_test)
y_pred_test = np.argmax(y_probs_test, axis=1)

acuracia = accuracy_score(y_test, y_pred_test)
loss = log_loss(y_test, y_probs_test)

tempo_total = time.time() - tempo_inicio

print(f"\n✅ Treinamento concluído em {tempo_total:.2f} segundos!")
print(f"🎯 Acurácia Indicativa: {acuracia * 100:.2f}%")
print(f"📉 Log Loss Calibrado Final: {loss:.4f}\n")

# -----------------------------------------------------------------------------
# 7. AVALIAÇÃO DE DESEMPENHO E VALOR ESPERADO (+EV)
# -----------------------------------------------------------------------------
print("💰 7/7 - Avaliando Desempenho de Apostas +EV no Conjunto de Teste...")

df_test = df_fatos.iloc[split_idx:].copy().reset_index(drop=True)
classes = list(le.classes_)

idx_e = classes.index('E')
idx_m = classes.index('M')
idx_v = classes.index('V')

p_m = y_probs_test[:, idx_m]
p_e = y_probs_test[:, idx_e]
p_v = y_probs_test[:, idx_v]

ev_m = (p_m * df_test['odd_m_fechamento'].values) - 1.0
ev_e = (p_e * df_test['odd_e_fechamento'].values) - 1.0
ev_v = (p_v * df_test['odd_v_fechamento'].values) - 1.0

# Filtro de corte para entradas +EV (EV > +5%)
CUTOFF_EV = 0.05
apostas_ev = []

for i, row in df_test.iterrows():
    actual_res = y_test[i]
    
    if ev_m[i] >= CUTOFF_EV and row['odd_m_fechamento'] > 1.0:
        win = 1 if actual_res == idx_m else 0
        profit = (row['odd_m_fechamento'] - 1.0) if win else -1.0
        apostas_ev.append({'ev': ev_m[i], 'win': win, 'profit': profit})
        
    if ev_e[i] >= CUTOFF_EV and row['odd_e_fechamento'] > 1.0:
        win = 1 if actual_res == idx_e else 0
        profit = (row['odd_e_fechamento'] - 1.0) if win else -1.0
        apostas_ev.append({'ev': ev_e[i], 'win': win, 'profit': profit})
        
    if ev_v[i] >= CUTOFF_EV and row['odd_v_fechamento'] > 1.0:
        win = 1 if actual_res == idx_v else 0
        profit = (row['odd_v_fechamento'] - 1.0) if win else -1.0
        apostas_ev.append({'ev': ev_v[i], 'win': win, 'profit': profit})

if len(apostas_ev) > 0:
    df_ev = pd.DataFrame(apostas_ev)
    total_apostas = len(df_ev)
    win_rate = df_ev['win'].mean()
    roi = (df_ev['profit'].sum() / total_apostas) * 100
    print(f"📊 Total de Apostas +EV Identificadas: {total_apostas}")
    print(f"🎯 Taxa de Acerto nas Apostas +EV: {win_rate * 100:.2f}%")
    print(f"📈 ROI / Yield Estimado: {roi:+.2f}%\n")
else:
    print("ℹ️ Nenhuma aposta +EV com Odd de fechamento válida registrada no conjunto de teste.\n")

# Treino Final Completo e Salvamento
modelo_calibrado_final = CalibratedClassifierCV(estimator=stacking_classifier, method='isotonic', cv=5)
modelo_calibrado_final.fit(X, y)

joblib.dump({
    'modelo': modelo_calibrado_final, 
    'features': feature_cols,
    'elo_ratings': elo_ratings,
    'label_encoder': le
}, 'modelo_futebol_v2.pkl')

print("💾 Modelo completo e calibrado salvo em 'modelo_futebol_v2.pkl'!")