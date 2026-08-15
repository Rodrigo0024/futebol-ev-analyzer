import duckdb

con = duckdb.connect('futebol.db')

print("🧹 Limpando espaços sobressalentes e padronizando nomes...")

# 1. Remover espaços em branco invisíveis no início e fim dos nomes
con.execute("UPDATE partidas SET time_mandante = TRIM(time_mandante);")
con.execute("UPDATE partidas SET time_visitante = TRIM(time_visitante);")

# 2. Dicionário de padronização (Mapeie variações conhecidas para o nome oficial)
mapeamento_times = {
    'Palmeiras - SP': 'Palmeiras',
    'Palmeiras-SP': 'Palmeiras',
    'São Paulo - SP': 'São Paulo',
    'Corinthians - SP': 'Corinthians',
    'Santos - SP': 'Santos',
    'Flamengo - RJ': 'Flamengo',
    'Fluminense - RJ': 'Fluminense',
    'Vasco - RJ': 'Vasco',
    'Botafogo - RJ': 'Botafogo',
    'Sao Paulo-SP': 'Sao Paulo',
    # Adicione outros se notar na sua lista
}

for var, oficial in mapeamento_times.items():
    con.execute(f"UPDATE partidas SET time_mandante = '{oficial}' WHERE time_mandante = '{var}';")
    con.execute(f"UPDATE partidas SET time_visitante = '{oficial}' WHERE time_visitante = '{var}';")

con.close()
print("✅ Banco de dados limpo e unificado com sucesso!")