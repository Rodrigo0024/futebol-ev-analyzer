import duckdb
import pandas as pd

def importar_jogos_csv(caminho_csv):
    df_novos = pd.read_csv(caminho_csv)
    
    # Garante a coluna de resultado se não existir
    if 'resultado' not in df_novos.columns:
        df_novos['resultado'] = df_novos.apply(
            lambda r: 'M' if r['gols_mandante'] > r['gols_visitante'] 
            else ('V' if r['gols_mandante'] < r['gols_visitante'] else 'E'), axis=1
        )

    con = duckdb.connect('futebol.db')
    con.execute("INSERT INTO partidas SELECT * FROM df_novos")
    con.close()
    
    print(f"🚀 {len(df_novos)} novos jogos importados com sucesso para o banco de dados!")

# Executar importação
importar_jogos_csv('novos_jogos_rodada.csv')