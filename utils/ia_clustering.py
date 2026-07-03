import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preparar_dados_clusterizacao(df_municipios):
    """
    Prepara os dados dos municípios para clusterização.
    Calcula indicadores como:
    - Massa total coletada
    - Massa per capita
    - % de resíduos orgânicos na coleta seletiva
    - % destinado a aterro
    - % destinado a compostagem
    - Número de rotas de coleta
    """
    # Agrupa por município
    agg = df_municipios.groupby(['MUNICÍPIO', 'UF']).agg({
        'MASSA_COLETADA': 'sum',
        'TIPO_COLETA_EXECUTADA': 'count',  # número de rotas
        'DESTINO': lambda x: ','.join(x.unique())  # destinos concatenados
    }).reset_index()
    
    # Renomeia colunas
    agg.columns = ['MUNICÍPIO', 'UF', 'Massa_Total', 'Num_Rotas', 'Destinos']
    
    # Calcula indicadores adicionais
    # % de rotas que vão para aterro (MCF > 0)
    # % de rotas que vão para compostagem
    def calc_indicadores(grupo):
        destinos = grupo['DESTINO'].str.lower()
        total = len(grupo)
        pct_aterro = destinos.str.contains('aterro').sum() / total * 100
        pct_compostagem = destinos.str.contains('compostagem').sum() / total * 100
        return pd.Series({
            'Pct_Aterro': pct_aterro,
            'Pct_Compostagem': pct_compostagem
        })
    
    # Aplica os indicadores por município
    indicadores = df_municipios.groupby(['MUNICÍPIO', 'UF']).apply(calc_indicadores).reset_index()
    
    # Junta com os dados agregados
    df_cluster = agg.merge(indicadores, on=['MUNICÍPIO', 'UF'])
    
    # Seleciona as features para clusterização
    features = ['Massa_Total', 'Num_Rotas', 'Pct_Aterro', 'Pct_Compostagem']
    X = df_cluster[features].copy()
    
    # Trata valores faltantes
    X = X.fillna(0)
    
    # Remove linhas com Massa_Total = 0 (municípios sem dados)
    X = X[X['Massa_Total'] > 0]
    
    return X, df_cluster

def clusterizar_municipios(X, n_clusters=4, random_state=42):
    """
    Aplica K-Means clusterização nos dados dos municípios.
    Retorna: labels, modelo KMeans, scaler
    """
    # Padroniza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplica K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    return labels, kmeans, scaler

def aplicar_pca(X, n_components=2, random_state=42):
    """
    Aplica PCA para redução de dimensionalidade (visualização 2D).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def plot_clusters(X_pca, labels, df_cluster):
    """
    Gera gráfico de dispersão dos clusters.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cria um DataFrame para plotagem
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': labels,
        'Município': df_cluster['MUNICÍPIO'],
        'UF': df_cluster['UF']
    })
    
    # Plota os pontos coloridos por cluster
    scatter = ax.scatter(
        df_plot['PC1'], 
        df_plot['PC2'],
        c=df_plot['Cluster'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Adiciona labels para alguns pontos (opcional: os maiores municípios)
    top_n = df_cluster.nlargest(10, 'Massa_Total')
    for _, row in top_n.iterrows():
        idx = df_cluster[df_cluster['MUNICÍPIO'] == row['MUNICÍPIO']].index[0]
        ax.annotate(
            f"{row['MUNICÍPIO'][:15]}",
            (X_pca[idx, 0], X_pca[idx, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('Clusterização de Municípios por Perfil de Resíduos')
    ax.legend(*scatter.legend_elements(), title='Cluster')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig

def resumo_clusters(df_cluster, labels):
    """
    Retorna um resumo estatístico por cluster.
    """
    df_cluster['Cluster'] = labels
    resumo = df_cluster.groupby('Cluster').agg({
        'MUNICÍPIO': 'count',
        'Massa_Total': ['mean', 'median', 'sum'],
        'Num_Rotas': 'mean',
        'Pct_Aterro': 'mean',
        'Pct_Compostagem': 'mean'
    }).round(2)
    
    resumo.columns = ['Quantidade', 'Massa_Media', 'Massa_Mediana', 'Massa_Total_Cluster', 
                      'Rotas_Media', 'Pct_Aterro_Media', 'Pct_Compostagem_Media']
    
    return resumo
