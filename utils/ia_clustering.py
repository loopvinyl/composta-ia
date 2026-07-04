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
    Detecta automaticamente as colunas de UF e destino.
    """
    df = df_municipios.copy()
    
    # --- DETECÇÃO DA COLUNA UF ---
    col_uf = None
    for col in df.columns:
        if 'uf' in col.lower():
            col_uf = col
            break
    if col_uf is None:
        df['UF'] = 'BR'
        col_uf = 'UF'

    # --- DETECÇÃO DA COLUNA DESTINO ---
    col_destino = None
    for col in df.columns:
        if any(palavra in col.lower() for palavra in ['destino', 'unidade', 'tipo']):
            col_destino = col
            break
    if col_destino is None:
        # Se não encontrar, usa a última coluna que parece ter texto (fallback)
        for col in df.columns:
            if df[col].dtype == 'object':
                col_destino = col
                break
        if col_destino is None:
            col_destino = df.columns[-1]

    # --- AGRUPAMENTO POR MUNICÍPIO E UF ---
    # Grupo para calcular soma da massa e contagem de rotas
    grupo = df.groupby(['MUNICÍPIO', col_uf])
    
    # Soma da massa
    massa_total = grupo['MASSA_COLETADA'].sum().reset_index()
    massa_total.rename(columns={'MASSA_COLETADA': 'Massa_Total'}, inplace=True)
    
    # Contagem de rotas (número de linhas por município)
    num_rotas = grupo.size().reset_index(name='Num_Rotas')
    
    # Concatenação de destinos (tratamento robusto)
    def concat_destinos(series):
        # Converte para string, remove NaN, espaços vazios e junta
        strings = series.dropna().astype(str).str.strip()
        strings = strings[strings != '']
        return ','.join(strings.unique()) if not strings.empty else ''
    
    destinos = grupo[col_destino].apply(concat_destinos).reset_index(name='Destinos')
    
    # --- JUNÇÃO DOS DATAFRAMES ---
    # Merge passo a passo
    df_cluster = massa_total.merge(num_rotas, on=['MUNICÍPIO', col_uf])
    df_cluster = df_cluster.merge(destinos, on=['MUNICÍPIO', col_uf])
    
    # Renomeia a coluna UF para padronizar
    df_cluster.rename(columns={col_uf: 'UF'}, inplace=True)
    
    # --- CÁLCULO DE INDICADORES (percentuais de destino) ---
    # Função para calcular percentuais para cada município
    def calc_indicadores(grupo_municipio):
        destinos_series = grupo_municipio[col_destino].dropna().astype(str).str.lower()
        total = len(grupo_municipio)
        if total == 0:
            return pd.Series({'Pct_Aterro': 0, 'Pct_Compostagem': 0})
        pct_aterro = destinos_series.str.contains('aterro').sum() / total * 100
        pct_compostagem = destinos_series.str.contains('compostagem').sum() / total * 100
        return pd.Series({
            'Pct_Aterro': pct_aterro,
            'Pct_Compostagem': pct_compostagem
        })
    
    indicadores = df.groupby(['MUNICÍPIO', col_uf]).apply(calc_indicadores).reset_index()
    indicadores.rename(columns={col_uf: 'UF'}, inplace=True)
    
    # Junta os indicadores ao df_cluster
    df_cluster = df_cluster.merge(indicadores, on=['MUNICÍPIO', 'UF'])
    
    # --- SELEÇÃO DAS FEATURES PARA CLUSTERIZAÇÃO ---
    features = ['Massa_Total', 'Num_Rotas', 'Pct_Aterro', 'Pct_Compostagem']
    X = df_cluster[features].copy()
    
    # Trata valores faltantes e zeros
    X = X.fillna(0)
    X = X[X['Massa_Total'] > 0]  # remove municípios sem dados
    
    return X, df_cluster

def clusterizar_municipios(X, n_clusters=4, random_state=42):
    """Aplica K-Means clusterização nos dados dos municípios."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans, scaler

def aplicar_pca(X, n_components=2, random_state=42):
    """Aplica PCA para redução de dimensionalidade (visualização 2D)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def plot_clusters(X_pca, labels, df_cluster):
    """Gera gráfico de dispersão dos clusters."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': labels,
        'Município': df_cluster['MUNICÍPIO'],
        'UF': df_cluster['UF']
    })
    
    scatter = ax.scatter(
        df_plot['PC1'], 
        df_plot['PC2'],
        c=df_plot['Cluster'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Anota os 10 maiores municípios
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
    """Retorna um resumo estatístico por cluster."""
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
