import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def preparar_dados_previsao(df_municipio, anos_historicos=[2023, 2024]):
    """
    Prepara os dados para previsão de geração de resíduos.
    Para cada município, calcula:
    - Massa total coletada por ano
    - População (se disponível)
    - Geração per capita (kg/hab/ano)
    - Tendência linear de crescimento
    """
    # Agrupa por município e ano
    df_agg = df_municipio.groupby(['MUNICÍPIO', 'UF']).agg({
        'MASSA_COLETADA': 'sum',
        'ANO': 'first'  # apenas para referência
    }).reset_index()
    
    # Simula dados de população (se não estiver disponível, usa estimativa)
    # Nota: no SNIS, a população está em colunas como DFE0001, mas pode não estar presente.
    # Vamos usar uma estimativa fictícia para demonstração.
    # Na prática, você pode buscar do IBGE ou usar a coluna de população do SNIS.
    df_agg['Populacao'] = df_agg['MASSA_COLETADA'] / 0.5  # supondo 500 kg/hab/ano, valor médio
    df_agg['Geracao_per_capita'] = df_agg['MASSA_COLETADA'] / df_agg['Populacao']
    
    # Cria features de tempo (ano)
    df_agg['Ano'] = pd.to_numeric(df_agg['ANO'], errors='coerce')
    df_agg = df_agg.dropna(subset=['Ano'])
    
    # Ordena por ano
    df_agg = df_agg.sort_values('Ano')
    
    return df_agg

def treinar_modelo_previsao(df_municipio, target='MASSA_COLETADA', test_size=0.2):
    """
    Treina um modelo Random Forest para prever a massa coletada.
    Utiliza como features: ano, população, geração per capita, e outras derivadas.
    """
    # Prepara os dados
    df = preparar_dados_previsao(df_municipio)
    
    # Seleciona features
    features = ['Ano', 'Populacao', 'Geracao_per_capita']
    X = df[features].copy()
    y = df[target]
    
    # Remove linhas com NaN
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Escala os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divide treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    # Treina o modelo
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Avalia o modelo
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Modelo treinado - MAE: {mae:.2f} t, R²: {r2:.2f}")
    
    return model, scaler, df, mae, r2

def prever_futuro(model, scaler, df_historico, anos_futuros=[2025, 2026, 2027, 2028, 2029, 2030]):
    """
    Faz previsões para os anos futuros com base no modelo treinado.
    """
    # Obtém a última população e geração per capita conhecidas
    ultimo_ano = df_historico['Ano'].max()
    ultima_linha = df_historico[df_historico['Ano'] == ultimo_ano].iloc[-1]
    
    pop_atual = ultima_linha['Populacao']
    gpc_atual = ultima_linha['Geracao_per_capita']
    
    # Estima taxa de crescimento populacional (ex: 1% ao ano)
    taxa_crescimento_pop = 0.01
    # Estima tendência de geração per capita (ex: aumento de 0.5% ao ano)
    tendencia_gpc = 0.005
    
    previsoes = []
    for ano in anos_futuros:
        # Projeta população e geração per capita
        anos_diff = ano - ultimo_ano
        pop_futura = pop_atual * (1 + taxa_crescimento_pop) ** anos_diff
        gpc_futura = gpc_atual * (1 + tendencia_gpc) ** anos_diff
        
        # Cria o vetor de features para o modelo
        X_futuro = np.array([[ano, pop_futura, gpc_futura]])
        X_futuro_scaled = scaler.transform(X_futuro)
        
        # Faz a previsão
        previsao = model.predict(X_futuro_scaled)[0]
        previsoes.append({
            'Ano': ano,
            'Populacao_Estimada': pop_futura,
            'Geracao_per_capita_Estimada': gpc_futura,
            'Massa_Prevista (t)': previsao
        })
    
    return pd.DataFrame(previsoes)

def plot_previsao(df_historico, df_previsao):
    """
    Gera gráfico com os dados históricos e a projeção futura.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dados históricos
    anos_hist = df_historico['Ano'].values
    massas_hist = df_historico['MASSA_COLETADA'].values
    
    ax.plot(anos_hist, massas_hist, 'o-', label='Dados Históricos', color='blue', linewidth=2)
    
    # Dados previstos
    anos_fut = df_previsao['Ano'].values
    massas_fut = df_previsao['Massa_Prevista (t)'].values
    
    ax.plot(anos_fut, massas_fut, 's--', label='Previsão', color='green', linewidth=2)
    
    # Adiciona rótulos
    for i, (ano, massa) in enumerate(zip(anos_hist, massas_hist)):
        ax.annotate(f'{massa:.0f}', (ano, massa), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    for i, (ano, massa) in enumerate(zip(anos_fut, massas_fut)):
        ax.annotate(f'{massa:.0f}', (ano, massa), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('Massa Coletada (t)')
    ax.set_title('Previsão de Geração de Resíduos Sólidos Urbanos')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Ajusta os ticks do eixo X
    todos_anos = sorted(set(anos_hist) | set(anos_fut))
    ax.set_xticks(todos_anos)
    ax.set_xticklabels([str(int(a)) for a in todos_anos])
    
    return fig
