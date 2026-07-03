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
    # --- DETECÇÃO DA COLUNA UF ---
    col_uf = None
    for col in df_municipios.columns:
        if 'uf' in col.lower():
            col_uf = col
            break
    if col_uf is None:
        df_municipios = df_municipios.copy()
        df_municipios['UF'] = 'BR'
        col_uf = 'UF'

    # --- DETECÇÃO DA COLUNA DESTINO ---
    col_destino = None
    for col in df_municipios.columns:
        if any(palavra in col.lower() for palavra in ['destino', 'unidade', 'tipo']):
            col_destino = col
            break
    if col_destino is None:
        col_destino = 'DESTINO' if 'DESTINO' in df_municipios.columns else df_municipios.columns[-1]

    # --- FUNÇÃO DE AGREGAÇÃO ROBUSTA PARA DESTINO ---
    def concat_destinos(series):
        strings = series.dropna().astype(str).str.strip()
        strings = strings[strings != '']
        return ','.join(strings.unique()) if not strings.empty else ''

    # --- AGRUPAMENTO POR MUNICÍPIO E UF (com as_index=False para evitar índices) ---
    agg = df_municipios.groupby(['MUNICÍPIO', col_uf], as_index=False).agg({
        'MASSA_COLETADA': 'sum',
        'TIPO_COLETA_EXECUTADA': 'count',  # número de rotas
        col_destino: concat_destinos       # destinos concatenados
    })
    
    # Renomeia colunas (garantindo que o número de colunas seja compatível)
    # O agg terá 4 colunas: ['MUNICÍPIO', col_uf, 'MASSA_COLETADA', 'TIPO_COLETA_EXECUTADA', col_destino]? 
    # Na verdade, o agg terá 5 colunas: MUNICÍPIO, col_uf, MASSA_COLETADA (sum), TIPO_COLETA_EXECUTADA (count), col_destino (concat)
    # Então são 5 colunas, mas vamos verificar.
    # Vamos renomear dinamicamente:
    new_cols = ['MUNICÍPIO', 'UF', 'Massa_Total', 'Num_Rotas', 'Destinos']
    # Mas o nome da coluna UF pode ser diferente (col_uf), então renomeamos depois.
    agg.rename(columns={col_uf: 'UF'}, inplace=True)
    # Agora renomeamos as outras colunas
    agg.rename(columns={
        'MASSA_COLETADA': 'Massa_Total',
        'TIPO_COLETA_EXECUTADA': 'Num_Rotas',
        col_destino: 'Destinos'
    }, inplace=True)
    
    # O DataFrame agora tem colunas: ['MUNICÍPIO', 'UF', 'Massa_Total', 'Num_Rotas', 'Destinos']
    
    # --- CÁLCULO DE INDICADORES ---
    def calc_indicadores(grupo):
        destinos_series = grupo[col_destino].dropna().astype(str).str.lower()
        total = len(grupo)
        if total == 0:
            return pd.Series({'Pct_Aterro': 0, 'Pct_Compostagem': 0})
        pct_aterro = destinos_series.str.contains('aterro').sum() / total * 100
        pct_compostagem = destinos_series.str.contains('compostagem').sum() / total * 100
        return pd.Series({
            'Pct_Aterro': pct_aterro,
            'Pct_Compostagem': pct_compostagem
        })
    
    # Aplica os indicadores por município (usando as_index=False)
    indicadores = df_municipios.groupby(['MUNICÍPIO', col_uf], as_index=False).apply(calc_indicadores).reset_index(drop=True)
    # O resultado terá colunas: ['MUNICÍPIO', col_uf, 'Pct_Aterro', 'Pct_Compostagem']
    indicadores.rename(columns={col_uf: 'UF'}, inplace=True)
    
    # Junta com os dados agregados
    df_cluster = agg.merge(indicadores, on=['MUNICÍPIO', 'UF'])
    
    # Seleciona as features para clusterização
    features = ['Massa_Total', 'Num_Rotas', 'Pct_Aterro', 'Pct_Compostagem']
    X = df_cluster[features].copy()
    
    # Trata valores faltantes e zeros
    X = X.fillna(0)
    X = X[X['Massa_Total'] > 0]  # remove municípios sem dados
    
    return X, df_cluster

def clusterizar_municipios(X, n_clusters=4, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans, scaler

def aplicar_pca(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def plot_clusters(X_pca, labels, df_cluster):
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
