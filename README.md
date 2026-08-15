# ⚽ Futebol EV+ Analyzer & Machine Learning Predictor

Aplicativo web interativo desenvolvido em **Python** e **Streamlit** para análise quantitativa de partidas de futebol. O sistema cruza **Modelos Preditivos de Machine Learning**, **Ratings Elo** e **Distribuição de Poisson** para calcular probabilidades reais e identificar oportunidades de apostas com **Valor Esperado Positivo (+EV)**, aplicando gestão de banca via Critério de Kelly.

---

## 🚀 Principais Tecnologias e Conceitos Aprendidos

Este projeto serviu como um laboratório prático de engenharia de dados, modelagem estatística e desenvolvimento full-stack em Python. Abaixo estão as principais tecnologias e fundamentos aplicados:

### 1. 🦆 DuckDB (Banco de Dados Analítico)
* **O que é:** Um banco de dados SQL relacional em memória e orientado a colunas, altamente otimizado para cargas de trabalho analíticas (OLAP).
* **Como foi aplicado:** Utilizado como o motor central de dados (`futebol.db`). Permite consultas extremamente rápidas e eficientes para extrair histórico de partidas, médias móveis de gols e metadados de campeonatos diretamente no disco sem sobrecarregar a memória RAM.

### 2. 🤖 Machine Learning & Modelos Preditivos
* **O que é:** O uso de algoritmos capazes de aprender padrões a partir de dados históricos para fazer previsões sobre eventos futuros.
* **Como foi aplicado:** 
  * Carga e inferência em tempo real utilizando um modelo serializado (`joblib`).
  * Uso do método `.predict_proba()` para transformar um vetor de características (*features*) complexas em probabilidades calibradas para os resultados (Vitória do Mandante, Empate, Vitória do Visitante).

### 3. 📊 Engenharia de Recursos (*Feature Engineering*) & Estatística Esportiva
* **Rating Elo:** Implementação de um sistema dinâmico de pontuação de força para os times, simulando o modelo clássico de xadrez adaptado ao futebol.
* **Distribuição de Poisson:** Aplicação da estatística de Poisson para modelar a probabilidade de ocorrência de gols com base nas taxas de ataque e defesa recentes de cada equipe.
* **Médias Móveis:** Extração de indicadores de desempenho recentes (gols pró e pontos nos últimos 5 jogos) como variáveis de entrada para o modelo preditivo.

### 4. 💰 Gestão de Risco Quantitativa (+EV & Critério de Kelly)
* **Cálculo de Valor Esperado (EV):** Comparação matemática entre a probabilidade real calculada pelo modelo e as cotações (Odds) oferecidas pelas casas de apostas:
  $$\text{EV} = (\text{Probabilidade} \times \text{Odd}) - 1$$
* **Critério de Kelly Fracionado:** Implementação de um algoritmo de alocação de capital para definir de forma matemática a porcentagem exata da banca (*stake*) que deve ser investida, maximizando o crescimento a longo prazo enquanto mitiga o risco de ruína.

### 5. 🌐 Streamlit (Interface Web em Python)
* **O que é:** Framework moderno para construção de aplicações web orientadas a dados usando puramente Python.
* **Como foi aplicado:** Construção de uma interface reativa e intuitiva contendo seletores de confrontos, painéis de métricas (`st.metric`), tabelas dinâmicas, barras laterais de gestão de banca e *expanders* informativos.

---

## 🛠️ Stack Tecnológica

* **Linguagem:** Python 🐍
* **Interface Web:** Streamlit
* **Banco de Dados:** DuckDB
* **Machine Learning & Serialização:** Scikit-Learn, Joblib
* **Computação Científica & Manipulação:** Pandas, NumPy, SciPy (Poisson)

---



## ⚙️ Como Instalar e Executar o Projeto

Siga os passos abaixo para rodar o projeto localmente na sua máquina:

1. **Clone o repositório:**
   ```bash
   git clone git@github.com:Rodrigo0024/futebol-ev-analyzer.git
   cd futebol-ev-analyzer
2. **Instale as dependências necessárias:**  
pip install -r requirements.txt
   
