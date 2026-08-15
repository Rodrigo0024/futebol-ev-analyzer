import duckdb

con = duckdb.connect()

print("🔍 Inspecionando a estrutura dos arquivos na pasta 'Data'...\n")

arquivos = [
    'Brasileirao_Matches.csv',
    'Brazilian_Cup_Matches.csv',
    'Libertadores_Matches.csv'
]

# 1. Inspeciona a estrutura de cada competição individualmente
for arq in arquivos:
    caminho = f"Data/{arq}"
    print(f"--- 📋 Colunas do arquivo: {arq} ---")
    try:
        df_colunas = con.execute(f"DESCRIBE SELECT * FROM '{caminho}'").df()
        print(df_colunas[['column_name', 'column_type']].to_string(index=False))
        
        qtd = con.execute(f"SELECT COUNT(*) FROM '{caminho}'").fetchone()[0]
        print(f"👉 Total de jogos: {qtd:,}\n")
    except Exception as e:
        print(f"Erro ao ler {arq}: {e}\n")

# 2. Consolidação usando 'union_by_name=True' (DuckDB alinha as colunas correspondentes)
print("--- 📊 Total de Jogos Consolidados ---")
total_partidas = con.execute("""
    SELECT COUNT(*) 
    FROM read_csv_auto('Data/*.csv', union_by_name=True)
""").fetchone()[0]

print(f"Total geral de partidas em todas as competições: {total_partidas:,}")