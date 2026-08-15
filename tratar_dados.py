import duckdb

# Conecta (ou cria) o banco de dados em arquivo
con = duckdb.connect('futebol.db')

print("🧹 Processando, limpando e consolidando os dados no DuckDB...\n")

# SQL de transformação e unificação
query_limpeza = """
CREATE OR REPLACE TABLE partidas AS
WITH partidas_bruto AS (
    -- 1. Brasileirão
    SELECT 
        'Brasileirao' AS competicao,
        CAST(datetime AS TIMESTAMP) AS data_jogo,
        TRY_CAST(season AS INT) AS temporada,
        home_team AS time_mandante,
        away_team AS time_visitante,
        TRY_CAST(regexp_extract(CAST(home_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_mandante,
        TRY_CAST(regexp_extract(CAST(away_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_visitante,
        home_team_state AS uf_mandante,
        away_team_state AS uf_visitante,
        CAST(round AS VARCHAR) AS fase_rodada
    FROM 'Data/Brasileirao_Matches.csv'

    UNION ALL

    -- 2. Copa do Brasil
    SELECT 
        'Copa do Brasil' AS competicao,
        CAST(datetime AS TIMESTAMP) AS data_jogo,
        TRY_CAST(season AS INT) AS temporada,
        home_team AS time_mandante,
        away_team AS time_visitante,
        TRY_CAST(regexp_extract(CAST(home_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_mandante,
        TRY_CAST(regexp_extract(CAST(away_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_visitante,
        NULL AS uf_mandante,
        NULL AS uf_visitante,
        CAST(round AS VARCHAR) AS fase_rodada
    FROM 'Data/Brazilian_Cup_Matches.csv'

    UNION ALL

    -- 3. Libertadores
    SELECT 
        'Libertadores' AS competicao,
        TRY_CAST(datetime AS TIMESTAMP) AS data_jogo,
        TRY_CAST(season AS INT) AS temporada,
        home_team AS time_mandante,
        away_team AS time_visitante,
        TRY_CAST(regexp_extract(CAST(home_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_mandante,
        TRY_CAST(regexp_extract(CAST(away_goal AS VARCHAR), '^([0-9]+)', 1) AS INT) AS gols_visitante,
        NULL AS uf_mandante,
        NULL AS uf_visitante,
        stage AS fase_rodada
    FROM 'Data/Libertadores_Matches.csv'
)
SELECT 
    *,
    (gols_mandante + gols_visitante) AS total_gols,
    CASE 
        WHEN gols_mandante > gols_visitante THEN 'M'  -- Mandante Venceu
        WHEN gols_mandante < gols_visitante THEN 'V'  -- Visitante Venceu
        ELSE 'E'                                      -- Empate
    END AS resultado
FROM partidas_bruto
WHERE gols_mandante IS NOT NULL 
  AND gols_visitante IS NOT NULL;  -- Remove jogos não disputados ou nulos
"""

con.execute(query_limpeza)

# Resumo do banco gerado
total_limpo = con.execute("SELECT COUNT(*) FROM partidas").fetchone()[0]
print(f"✅ Tratamento concluído com sucesso!")
print(f"📊 Total de partidas válidas salvas na tabela 'partidas': {total_limpo:,}\n")

# Exibe distribuição de resultados
df_resumo = con.execute("""
    SELECT 
        competicao,
        COUNT(*) as total_jogos,
        ROUND(AVG(gols_mandante + gols_visitante), 2) as media_gols,
        ROUND(COUNT(CASE WHEN resultado = 'M' THEN 1 END) * 100.0 / COUNT(*), 1) as pct_vitoria_mandante,
        ROUND(COUNT(CASE WHEN resultado = 'E' THEN 1 END) * 100.0 / COUNT(*), 1) as pct_empate,
        ROUND(COUNT(CASE WHEN resultado = 'V' THEN 1 END) * 100.0 / COUNT(*), 1) as pct_vitoria_visitante
    FROM partidas
    GROUP BY competicao
    ORDER BY total_jogos DESC
""").df()

print("--- ⚽ Resumo Estatístico por Competição ---")
print(df_resumo.to_string(index=False))

con.close()