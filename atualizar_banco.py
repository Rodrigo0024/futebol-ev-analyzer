import duckdb
import pandas as pd

def adicionar_partida(competicao, data_jogo, time_mandante, time_visitante, gols_mandante, gols_visitante):
    # Determinar resultado
    if gols_mandante > gols_visitante:
        res = 'M'
    elif gols_mandante < gols_visitante:
        res = 'V'
    else:
        res = 'E'

    con = duckdb.connect('futebol.db')
    
    # Inserção no banco
    con.execute("""
        INSERT INTO partidas (competicao, data_jogo, time_mandante, time_visitante, gols_mandante, gols_visitante, resultado)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [competicao, data_jogo, time_mandante, time_visitante, gols_mandante, gols_visitante, res])
    
    con.close()
    print(f"✅ Jogo adicionado com sucesso: {time_mandante} {gols_mandante} x {gols_visitante} {time_visitante}")

# --- EXEMPLO DE USO ---
if __name__ == "__main__":
    # Adicione os jogos recentes da rodada aqui:
    adicionar_partida(
        competicao='SerieA',
        data_jogo='2026-08-10',
        time_mandante='Flamengo',
        time_visitante='Palmeiras',
        gols_mandante=2,
        gols_visitante=1
    )