import duckdb
import time

print("⚡ Iniciando teste de performance com DuckDB...")

# Conecta ao DuckDB em memória (ultra-rápido)
con = duckdb.connect()

# 1. Gerando 100.000 chutes fictícios com coordenadas X (0 a 120m) e Y (0 a 80m)
# Consideramos o centro do gol na posição X = 120 e Y = 40
inicio = time.time()

con.execute("""
CREATE TABLE chutes AS
SELECT 
    range AS id_chute,
    'Jogador_' || (range % 100) AS jogador,
    random() * 120 AS x,
    random() * 80 AS y,
    CASE WHEN random() > 0.5 THEN 'Pé Direito' ELSE 'Pé Esquerdo' END AS pe,
    CASE WHEN random() > 0.9 THEN 1 ELSE 0 END AS gol
FROM range(100000);
""")

# 2. Consulta SQL em escala: calcula a distância até o gol e filtra chutes perigosos
df_resultado = con.execute("""
SELECT 
    id_chute,
    jogador,
    x,
    y,
    pe,
    -- Cálculo da distância até o centro do gol (120, 40)
    ROUND(SQRT(POWER(120 - x, 2) + POWER(40 - y, 2)), 2) AS distancia_gol_metros,
    gol
FROM chutes
WHERE x >= 100 -- Seleciona apenas chutes do campo de ataque (últimos 20 metros)
ORDER BY distancia_gol_metros ASC
""").df()

fim = time.time()

# 3. Exibindo os resultados
print(f"✅ 100.000 registros processados e filtrados em: {fim - inicio:.4f} segundos!\n")
print("--- Primeiras linhas dos dados processados pelo DuckDB ---")
print(df_resultado.head())