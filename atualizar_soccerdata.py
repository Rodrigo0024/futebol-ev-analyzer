import duckdb
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# DICIONÁRIO DE PADRONIZAÇÃO DE NOMES DE TIMES
# -----------------------------------------------------------------------------
MAPA_TIMES = {
    'Atlético Mineiro': 'Atletico-MG',
    'Atlético-MG': 'Atletico-MG',
    'Athletico Paranaense': 'Athletico-PR',
    'América Mineiro': 'America-MG',
    'América-MG': 'America-MG',
    'Bragantino': 'Red Bull Bragantino',
    'RB Bragantino': 'Red Bull Bragantino',
}

def padronizar_time(nome):
    return MAPA_TIMES.get(str(nome).strip(), str(nome).strip())

# -----------------------------------------------------------------------------
# 1. ATUALIZAÇÃO DO BRASILEIRÃO
# -----------------------------------------------------------------------------
def atualizar_brasileirao():
    print("🌐 Buscando dados atualizados do Brasileirão Série A...")
    
    # URL corrigida com o nome exato da base de dados oficial
    url = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv"
    
    try:
        df_raw = pd.read_csv(url)
        
        # Converte todas as colunas para minúsculo para evitar divergência de maiúsculas/minúsculas
        df_raw.columns = df_raw.columns.str.lower()
        
        # Filtra apenas os jogos que já foram encerrados (com placar preenchido)
        df_jogos = df_raw.dropna(subset=['mandante_placar', 'visitante_placar']).copy()
        
        df_novos = pd.DataFrame()
        df_novos['competicao'] = 'SerieA'
        
        # Tratamento flexível de datas
        df_novos['data_jogo'] = pd.to_datetime(df_jogos['data'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
        df_novos['time_mandante'] = df_jogos['mandante'].apply(padronizar_time)
        df_novos['time_visitante'] = df_jogos['visitante'].apply(padronizar_time)
        df_novos['gols_mandante'] = df_jogos['mandante_placar'].astype(int)
        df_novos['gols_visitante'] = df_jogos['visitante_placar'].astype(int)

        df_novos = df_novos.dropna(subset=['data_jogo'])

        # Classificação do resultado: M (Mandante), E (Empate), V (Visitante)
        df_novos['resultado'] = np.where(
            df_novos['gols_mandante'] > df_novos['gols_visitante'], 'M',
            np.where(df_novos['gols_mandante'] < df_novos['gols_visitante'], 'V', 'E')
        )
        
        salvar_no_banco(df_novos)

    except Exception as e:
        print(f"❌ Erro ao processar o Brasileirão: {e}")

# -----------------------------------------------------------------------------
# 2. SALVAR NO DUCKDB SEM DUPLICATAS
# -----------------------------------------------------------------------------
def salvar_no_banco(df_novos):
    con = duckdb.connect('futebol.db')
    con.register('df_temp', df_novos)

    query = """
        INSERT INTO partidas (competicao, data_jogo, time_mandante, time_visitante, gols_mandante, gols_visitante, resultado)
        SELECT t.competicao, t.data_jogo, t.time_mandante, t.time_visitante, t.gols_mandante, t.gols_visitante, t.resultado
        FROM df_temp t
        WHERE NOT EXISTS (
            SELECT 1 FROM partidas p
            WHERE p.data_jogo = t.data_jogo
              AND p.time_mandante = t.time_mandante
              AND p.time_visitante = t.time_visitante
        )
    """
    linhas_antes = con.execute("SELECT COUNT(*) FROM partidas").fetchone()[0]
    con.execute(query)
    linhas_depois = con.execute("SELECT COUNT(*) FROM partidas").fetchone()[0]
    con.close()

    print(f"✅ Sucesso! {linhas_depois - linhas_antes} novos jogos foram salvos no banco `futebol.db`!")

# -----------------------------------------------------------------------------
# EXECUÇÃO
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    atualizar_brasileirao()