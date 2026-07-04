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
    Retorna X (features) e df_cluster com as mesmas linhas (filtrados por Massa_Total > 0).
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
        for col in df.columns:
            if df[col].dtype == 'object':
                col_destino = col
                break
        if col_destino is None:
            col_destino = df.columns[-1]

    # --- FUNÇÃO DE AGREGAÇÃO ROBUSTA PARA DESTINO ---
    def concat_destinos(series):
        strings = series.dropna().astype(str).str.strip()
        strings = strings[strings != '']
        return ','.join(strings.unique()) if not strings.empty else ''

    # --- AGRUPAMENTO POR MUNICÍPIO E UF ---
    grupo = df.groupby(['MUNICÍPIO', col_uf])
    
    massa_total = grupo['MASSA_COLETADA'].sum().reset_index()
    massa_total.rename(columns={'MASSA_COLETADA': 'Massa_Total'}, inplace=True)
    
    num_rotas = grupo.size().reset_index(name='Num_Rotas')
    
    destinos = grupo[col_destino].apply(concat_destinos).reset_index(name='Destinos')
    
    df_cluster = massa_total.merge(num_rotas, on=['MUNICÍPIO', col_uf])
    df_cluster = df_cluster.merge(destinos, on=['MUNICÍPIO', col_uf])
    
    df_cluster.rename(columns={col_uf: 'UF'}, inplace=True)
    
    # --- CÁLCULO DE INDICADORES ---
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
    
    df_cluster = df_cluster.merge(indicadores, on=['MUNICÍPIO', 'UF'])
    
    # --- SELEÇÃO DAS FEATURES ---
    features = ['Massa_Total', 'Num_Rotas', 'Pct_Aterro', 'Pct_Compostagem']
    X = df_cluster[features].copy()
    
    # --- FILTRO: REMOVE MUNICÍPIOS SEM MASSA (Massa_Total = 0) ---
    mask = X['Massa_Total'] > 0
    X = X.loc[mask].copy()
    df_cluster = df_cluster.loc[mask].copy()  # <-- CORREÇÃO AQUI: aplica o mesmo filtro
    
    return X, df_cluster

def clusterizar_municipios(X, n_clusters=4, random_state=42):
    """Aplica K-Means clusterização."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans, scaler

def aplicar_pca(X, n_components=2, random_state=42):
    """Aplica PCA para visualização 2D."""
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
    Espera que df_cluster tenha as colunas necessárias e que labels seja um array.
    """
    df = df_cluster.copy()
    df['Cluster'] = labels
    resumo = df.groupby('Cluster').agg({
        'MUNICÍPIO': 'count',
        'Massa_Total': ['mean', 'median', 'sum'],
        'Num_Rotas': 'mean',
        'Pct_Aterro': 'mean',
        'Pct_Compostagem': 'mean'
    }).round(2)
    
    resumo.columns = ['Quantidade', 'Massa_Media', 'Massa_Mediana', 'Massa_Total_Cluster', 
                      'Rotas_Media', 'Pct_Aterro_Media', 'Pct_Compostagem_Media']
    return resumo

def descrever_clusters(df_cluster, labels):
    """
    Gera uma descrição textual para cada cluster com base nas médias das variáveis.
    Espera que df_cluster tenha as colunas necessárias e que labels seja um array.
    """
    df = df_cluster.copy()
    df['Cluster'] = labels
    descricoes = {}
    
    for cluster in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == cluster]
        media_massa = subset['Massa_Total'].mean()
        media_rotas = subset['Num_Rotas'].mean()
        media_aterro = subset['Pct_Aterro'].mean()
        media_compostagem = subset['Pct_Compostagem'].mean()
        n_municipios = len(subset)
        
        desc = f"**Cluster {cluster+1}** – {n_municipios} municípios\n\n"
        
        # Perfil de massa
        if media_massa > 100000:
            desc += "📊 **Massa de resíduos:** Alta (acima de 100 mil t/ano). "
        elif media_massa > 20000:
            desc += "📊 **Massa de resíduos:** Média (entre 20 mil e 100 mil t/ano). "
        else:
            desc += "📊 **Massa de resíduos:** Baixa (menos de 20 mil t/ano). "
        
        # Perfil de destinação
        if media_compostagem > 50:
            desc += "♻️ **Compostagem:** Alta (acima de 50% das rotas). "
            if media_aterro < 20:
                desc += "🚮 **Aterro:** Baixo (menos de 20%). Este cluster já tem uma boa infraestrutura de compostagem. "
                recomendacao = "**Recomendação:** Manter e expandir a coleta seletiva, e incentivar a compostagem doméstica."
            else:
                desc += "🚮 **Aterro:** Moderado. Há espaço para aumentar a compostagem. "
                recomendacao = "**Recomendação:** Ampliar a coleta seletiva de orgânicos e investir em usinas de compostagem."
        elif media_aterro > 70:
            desc += "🚮 **Aterro:** Muito alto (acima de 70% das rotas). "
            desc += "♻️ **Compostagem:** Baixa (menos de 30%). Este cluster depende fortemente de aterros. "
            recomendacao = "**Recomendação:** Prioridade máxima para implantar coleta seletiva de orgânicos e compostagem."
        else:
            desc += "🚮 **Aterro:** Moderado (entre 30% e 70%). "
            desc += "♻️ **Compostagem:** Moderada. Há potencial para melhorar. "
            recomendacao = "**Recomendação:** Fortalecer a coleta seletiva e buscar parcerias para compostagem comunitária."
        
        if media_rotas > 10:
            desc += f"🛣️ **Rotas de coleta:** {media_rotas:.1f} (muitas rotas, boa cobertura). "
        elif media_rotas > 3:
            desc += f"🛣️ **Rotas de coleta:** {media_rotas:.1f} (número médio de rotas). "
        else:
            desc += f"🛣️ **Rotas de coleta:** {media_rotas:.1f} (poucas rotas, pode indicar baixa capilaridade). "
        
        desc += "\n\n" + recomendacao
        
        descricoes[cluster] = desc
    
    return descricoes
