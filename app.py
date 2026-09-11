# =====================================================================
# COMPOSTA.IA - SCRIPT COMPLETO PARA GOOGLE COLAB
# =====================================================================
# Replica TODAS as funcionalidades do aplicativo Streamlit:
# - Aba 1: Estatísticas tradicionais (SNIS), Pareto, distribuição por estado/destino
# - Aba 2: Insights com IA (classificação PLN, clusterização, projeções, cenários)
# - Aba 3: Diagnóstico de emissões (baseline UNFCCC, limiares SBCE, projeção contínua)
#
# Gera CSVs com todos os resultados e gráficos em PNG.
# =====================================================================

# 1. INSTALAÇÃO DE DEPENDÊNCIAS
!pip install pandas numpy openpyxl scikit-learn scipy matplotlib seaborn requests yfinance beautifulsoup4 -q

# 2. IMPORTAÇÃO DE BIBLIOTECAS
import os
import re
import glob
import unicodedata
import pickle
import warnings
from datetime import datetime
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
warnings.filterwarnings('ignore')

# =====================================================================
# 3. CONSTANTES E PARÂMETROS (IGUAIS AO APP)
# =====================================================================
GWP_CH4 = 28.0
GWP_N2O = 265.0
PHI_APPLICATION_B = 0.85
OX_SOIL_COVER = 0.383
F_METHANE_FRACTION = 0.5
ANOS_PROJECAO = 20
DIAS_PROJECAO = ANOS_PROJECAO * 365

# =====================================================================
# 4. FUNÇÕES AUXILIARES (CÓPIA EXATA DO APP)
# =====================================================================

def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto.strip()

def classificar_destino_regra(texto):
    if not texto or pd.isna(texto):
        return "Indefinido"
    t_lower = texto.lower()
    if any(p in t_lower for p in ['aterro sanitario', 'aterro sanitário', 'as']):
        return "Aterro Sanitário"
    if any(p in t_lower for p in ['aterro controlado']):
        return "Aterro Controlado"
    if any(p in t_lower for p in ['lixao', 'vazadouro', 'lixão']):
        return "Lixão"
    if any(p in t_lower for p in ['compostagem', 'compost']):
        return "Compostagem"
    if any(p in t_lower for p in ['transbordo', 'transbord']):
        return "Transbordo"
    if any(p in t_lower for p in ['reciclagem', 'triagem', 'cooperativa']):
        return "Reciclagem"
    return "Outros"

class ClassificadorDestinoIA:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.classes = None

    def treinar_com_dados_snis(self, df, col_texto):
        textos = df[col_texto].dropna().astype(str).tolist()
        if not textos:
            return
        textos_norm = [normalizar_texto(t) for t in textos]
        labels = [classificar_destino_regra(t) for t in textos]
        dados = [(t, l) for t, l in zip(textos_norm, labels) if l != "Indefinido"]
        if not dados:
            return
        X, y = zip(*dados)
        self.vectorizer = TfidfVectorizer(max_features=500)
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_vec, y)
        self.classes = self.classifier.classes_

    def prever(self, texto, threshold=0.3):
        if self.classifier is None:
            return classificar_destino_regra(texto)
        texto_norm = normalizar_texto(texto)
        if not texto_norm:
            return "Indefinido"
        X = self.vectorizer.transform([texto_norm])
        probs = self.classifier.predict_proba(X)[0]
        if max(probs) < threshold:
            return classificar_destino_regra(texto)
        return self.classifier.predict(X)[0]

def encontrar_coluna(df, padroes):
    for col in df.columns:
        for padrao in padroes:
            if padrao.lower() in col.lower():
                return col
    return None

def calcular_doc_k_ponderado(df_municipio):
    doc_map = {'GTR1501':0.15,'GTR1502':0.0,'GTR1503':0.0,'GTR1504':0.0,'GTR1505':0.40,'GTR1506':0.24,'GTR1507':0.10}
    docf_map = {'GTR1501':0.7,'GTR1502':0.0,'GTR1503':0.0,'GTR1504':0.0,'GTR1505':0.5,'GTR1506':0.5,'GTR1507':0.1}
    k_map = {'GTR1501':0.17,'GTR1502':0.0,'GTR1503':0.0,'GTR1504':0.0,'GTR1505':0.07,'GTR1506':0.07,'GTR1507':0.035}
    cols = [col for col in doc_map.keys() if col in df_municipio.columns]
    if not cols:
        return 0.15, 0.5, 0.07
    pct = pd.to_numeric(df_municipio[cols], errors='coerce').fillna(0)
    total = pct.sum().sum()
    if total <= 0:
        return 0.15, 0.5, 0.07
    doc = sum(pct[col].sum() * doc_map.get(col,0) for col in cols) / total
    docf = sum(pct[col].sum() * docf_map.get(col,0) for col in cols) / total
    k = sum(pct[col].sum() * k_map.get(col,0) for col in cols) / total
    return max(0.01,min(0.5,doc)), max(0.05,min(0.9,docf)), max(0.01,min(0.5,k))

def determinar_mcf_por_destino(destino):
    if pd.isna(destino):
        return 0.0
    norm = normalizar_texto(destino)
    if "aterro sanitario" in norm:
        return 0.8
    elif "aterro controlado" in norm:
        return 0.4
    elif "lixao" in norm or "vazadouro" in norm:
        return 0.4
    return 0.0

def calcular_co2eq_aterro_20anos(massa_t, mcf, k, doc, docf):
    if massa_t <=0 or mcf<=0: return 0.0
    massa_kg = massa_t*1000
    ch4_pot = doc*docf*mcf*F_METHANE_FRACTION*(16/12)*(1-OX_SOIL_COVER)*PHI_APPLICATION_B
    frac = 1 - np.exp(-k*ANOS_PROJECAO)
    return (massa_kg * ch4_pot * frac * GWP_CH4) / 1000.0

def calcular_co2eq_compostagem(massa_t):
    if massa_t<=0: return 0.0
    massa_kg = massa_t*1000
    return (massa_kg*0.002*GWP_CH4 + massa_kg*0.0002*GWP_N2O)/1000.0

def calcular_evitado_por_municipio(df):
    resultados = []
    mask = df['TIPO_COLETA_EXECUTADA'].astype(str).str.contains("seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)
    df_org = df[mask].copy()
    if df_org.empty:
        return pd.DataFrame()
    for mun in df_org['MUNICÍPIO'].unique():
        df_mun = df[df['MUNICÍPIO']==mun].copy()
        df_mun_org = df_org[df_org['MUNICÍPIO']==mun].copy()
        massa_org = df_mun_org['MASSA_COLETADA'].sum()
        if massa_org == 0: continue
        doc, docf, k = calcular_doc_k_ponderado(df_mun)
        df_mun['MCF'] = df_mun['DESTINO'].apply(determinar_mcf_por_destino)
        df_aterro = df_mun[df_mun['MCF']>0].copy()
        if not df_aterro.empty:
            mcf_medio = (df_aterro['MASSA_COLETADA']*df_aterro['MCF']).sum() / df_aterro['MASSA_COLETADA'].sum()
        else:
            mcf_medio = 0.8
        co2_aterro = calcular_co2eq_aterro_20anos(massa_org, mcf_medio, k, doc, docf)
        co2_compost = calcular_co2eq_compostagem(massa_org)
        resultados.append({'MUNICÍPIO': mun, 'Massa_Org_Seletiva': massa_org, 'Evitado_Total': co2_aterro - co2_compost})
    return pd.DataFrame(resultados)

def projetar_emissao_continua(massa_anual_t, mcf, k, doc, docf, anos=20):
    if massa_anual_t <=0 or mcf<=0:
        return pd.DataFrame()
    ch4_pot = doc*docf*mcf*F_METHANE_FRACTION*(16/12)*(1-OX_SOIL_COVER)*PHI_APPLICATION_B
    fator = ch4_pot * GWP_CH4
    results = []
    acum = 0.0
    for ano in range(1, anos+1):
        emissao_ano = 0.0
        for i in range(1, ano+1):
            idade = ano - i + 1
            frac = np.exp(-k*(idade-1)) - np.exp(-k*idade)
            emissao_ano += massa_anual_t * fator * frac
        acum += emissao_ano
        results.append({'Ano': datetime.now().year + ano, 'Emissao_Anual': emissao_ano, 'Emissao_Acumulada': acum})
    return pd.DataFrame(results)

def obter_cotacao_carbono():
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            if 10 < preco < 200:
                return preco, True
    except: pass
    return 85.50, False

def obter_cotacao_euro():
    try:
        resp = requests.get("https://economia.awesomeapi.com.br/last/EUR-BRL", timeout=5)
        if resp.status_code == 200:
            return float(resp.json()['EURBRL']['bid']), True
    except: pass
    return 5.50, False

# =====================================================================
# 5. FUNÇÃO PARA LER OS ARQUIVOS EXCEL (ROBUSTA)
# =====================================================================
def carregar_snis(caminho):
    xl = pd.ExcelFile(caminho)
    # Tenta encontrar as abas corretas
    aba_coleta = None
    aba_caract = None
    for nome in xl.sheet_names:
        n = nome.lower()
        if 'coleta' in n and 'destin' in n:
            aba_coleta = nome
        if 'resíduo' in n or 'caracter' in n or 'gtr' in n:
            aba_caract = nome
    if aba_coleta is None or aba_caract is None:
        # Fallback: usa as duas primeiras abas
        if len(xl.sheet_names) >= 2:
            aba_coleta = xl.sheet_names[0]
            aba_caract = xl.sheet_names[1]
        else:
            return None
    # Tenta header=12 (padrão do SNIS) ou 0
    df_coleta = None
    for h in [12, 13, 0]:
        try:
            df_temp = pd.read_excel(caminho, sheet_name=aba_coleta, header=h)
            if 'Cod_IBGE' in df_temp.columns or 'Município' in df_temp.columns or 'municipio' in df_temp.columns:
                df_coleta = df_temp
                break
        except: continue
    if df_coleta is None:
        return None
    df_caract = None
    for h in [12, 13, 0]:
        try:
            df_temp = pd.read_excel(caminho, sheet_name=aba_caract, header=h)
            cols = [c for c in df_temp.columns if isinstance(c,str) and c.upper().startswith('GTR150')]
            if cols:
                df_caract = df_temp[['Cod_IBGE'] + cols] if 'Cod_IBGE' in df_temp else df_temp
                break
        except: continue
    if df_caract is None:
        df_caract = pd.DataFrame()
    # Merge
    if 'Cod_IBGE' in df_coleta and 'Cod_IBGE' in df_caract:
        df = pd.merge(df_coleta, df_caract, on='Cod_IBGE', how='left')
    else:
        df = df_coleta
    return df

# =====================================================================
# 6. PROCESSAMENTO PRINCIPAL (LOOP POR ANO)
# =====================================================================
pasta = '/content/'
arquivos = glob.glob(os.path.join(pasta, '*.xlsx'))
print(f"📁 Encontrados {len(arquivos)} arquivos .xlsx em {pasta}\n")

for arquivo in arquivos:
    nome_base = os.path.basename(arquivo).replace('.xlsx', '')
    print(f"\n{'='*60}")
    print(f"🔍 PROCESSANDO: {nome_base}")
    print('='*60)

    try:
        # 6.1 Carregar dados
        df = carregar_snis(arquivo)
        if df is None or df.empty:
            print("❌ Erro: não foi possível carregar o arquivo.")
            continue

        # 6.2 Identificar colunas
        col_mun = encontrar_coluna(df, ['município','municipio','nom_mun'])
        col_uf = encontrar_coluna(df, ['uf','estado','sigla'])
        col_tipo = encontrar_coluna(df, ['tipo de coleta','tipo coleta','coleta','gtr1001'])
        col_massa = encontrar_coluna(df, ['massa coletada','massa (t)','massa total','quantidade coletada','gtr1008'])
        col_destino = encontrar_coluna(df, ['destino','unidade','local de destinação','gtr1011'])
        if None in [col_mun, col_uf, col_tipo, col_massa, col_destino]:
            print("❌ Colunas não encontradas. Pulando.")
            continue

        df = df.rename(columns={col_mun:'MUNICÍPIO', col_tipo:'TIPO_COLETA_EXECUTADA',
                               col_massa:'MASSA_COLETADA', col_uf:'UF', col_destino:'DESTINO'})
        df['MASSA_COLETADA'] = pd.to_numeric(df['MASSA_COLETADA'], errors='coerce').fillna(0)
        if 'DFE0001' in df.columns:
            df.rename(columns={'DFE0001':'POPULACAO_TOTAL'}, inplace=True)
        elif 'POPULACAO_TOTAL' not in df.columns:
            for c in df.columns:
                if 'popula' in c.lower():
                    df.rename(columns={c:'POPULACAO_TOTAL'}, inplace=True); break

        print(f"✅ Dados carregados: {df.shape[0]} linhas, {df['MUNICÍPIO'].nunique()} municípios")

        # 6.3 Classificador IA
        classificador = ClassificadorDestinoIA()
        classificador.treinar_com_dados_snis(df, 'DESTINO')
        df['DESTINO_IA'] = df['DESTINO'].apply(lambda x: classificador.prever(x, 0.3))

        # =============================================================
        # 7. ABAS DO APP - GERAÇÃO DE TODOS OS RESULTADOS
        # =============================================================

        # ---------- 7.1 Aba 1: Análise Tradicional (SNIS) ----------
        print("📊 Gerando Análise Tradicional...")

        # Estatísticas gerais
        total_mun = df['MUNICÍPIO'].nunique()
        df_temp = df.copy()
        df_temp['MCF'] = df_temp['DESTINO'].apply(determinar_mcf_por_destino)
        mun_com_aterro = df_temp[df_temp['MCF']>0]['MUNICÍPIO'].nunique()
        estatisticas_gerais = pd.DataFrame({
            'Ano': [nome_base],
            'Total_Municipios': [total_mun],
            'Municipios_com_Aterro': [mun_com_aterro],
            'Municipios_sem_Aterro': [total_mun - mun_com_aterro],
            'Massa_Total_RSU_t': [df['MASSA_COLETADA'].sum()]
        })
        estatisticas_gerais.to_csv(os.path.join(pasta, f'estatisticas_gerais_{nome_base}.csv'), index=False)

        # Per capita e Pareto (massa)
        df_massa_mun = df.groupby('MUNICÍPIO').agg({'MASSA_COLETADA':'sum', 'POPULACAO_TOTAL':'first'}).reset_index()
        df_massa_mun = df_massa_mun[(df_massa_mun['MASSA_COLETADA']>0) & (df_massa_mun['POPULACAO_TOTAL']>0)].copy()
        if not df_massa_mun.empty:
            df_massa_mun['per_capita'] = (df_massa_mun['MASSA_COLETADA'] / df_massa_mun['POPULACAO_TOTAL']) * 1000
            stats_percapita = {
                'Media_kg_hab': df_massa_mun['per_capita'].mean(),
                'Mediana_kg_hab': df_massa_mun['per_capita'].median(),
                'Q1_kg_hab': df_massa_mun['per_capita'].quantile(0.25),
                'Q3_kg_hab': df_massa_mun['per_capita'].quantile(0.75)
            }
            pd.DataFrame([stats_percapita]).to_csv(os.path.join(pasta, f'percapita_stats_{nome_base}.csv'), index=False)

            # Gráfico Pareto da massa
            df_ord = df_massa_mun.sort_values('MASSA_COLETADA', ascending=False)
            df_ord['pct_acum'] = df_ord['MASSA_COLETADA'].cumsum() / df_ord['MASSA_COLETADA'].sum() * 100
            df_ord['pct_mun'] = (np.arange(len(df_ord))+1) / len(df_ord) * 100
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(df_ord['pct_mun'], df_ord['pct_acum'], color='#1f77b4', linewidth=2)
            ax.axhline(80, color='red', linestyle='--')
            ax.set_xlabel('% acumulado de municípios'); ax.set_ylabel('% acumulado da massa')
            ax.set_title(f'Concentração da Massa de RSU – {nome_base}')
            ax.grid(True, linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(pasta, f'pareto_massa_{nome_base}.png'), dpi=150)
            plt.close()

        # Destinação agregada
        agg_destino = df.groupby('DESTINO_IA')['MASSA_COLETADA'].sum().reset_index().sort_values('MASSA_COLETADA', ascending=False)
        agg_destino.to_csv(os.path.join(pasta, f'destinacao_ia_{nome_base}.csv'), index=False)

        # Coleta seletiva de orgânicos - Ranking
        mask_org = df['TIPO_COLETA_EXECUTADA'].astype(str).str.contains("seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)
        df_org = df[mask_org].copy()
        if not df_org.empty:
            ranking = df_org.groupby('MUNICÍPIO').agg({'MASSA_COLETADA':'sum', 'UF':'first'}).reset_index()
            ranking = ranking.sort_values('MASSA_COLETADA', ascending=False)
            ranking.to_csv(os.path.join(pasta, f'ranking_org_seletiva_{nome_base}.csv'), index=False)

        # ---------- 7.2 Aba 2: Insights com IA ----------
        print("🧠 Gerando Insights com IA...")

        # Distribuição nacional da IA
        contagem_ia = df['DESTINO_IA'].value_counts().reset_index()
        contagem_ia.columns = ['Destino_IA', 'Quantidade']
        contagem_ia.to_csv(os.path.join(pasta, f'distribuicao_ia_{nome_base}.csv'), index=False)

        # Clusterização (K-Means)
        print("   - Executando clusterização...")
        # Preparar dados para cluster
        df_cluster = df.groupby('MUNICÍPIO').agg({
            'MASSA_COLETADA':'sum',
            'UF':'first',
            'POPULACAO_TOTAL':'first'
        }).reset_index()
        df_cluster = df_cluster[df_cluster['MASSA_COLETADA']>0].copy()
        if not df_cluster.empty and len(df_cluster) >= 5:
            # Cria features: massa per capita, % orgânico, % para aterro
            df_cluster['per_capita'] = (df_cluster['MASSA_COLETADA'] / df_cluster['POPULACAO_TOTAL']) * 1000
            df_cluster = df_cluster.fillna(0)
            # % de orgânico na coleta seletiva (aproximado)
            mask_org_mun = df[mask_org].groupby('MUNICÍPIO')['MASSA_COLETADA'].sum().reset_index().rename(columns={'MASSA_COLETADA':'Massa_Org'})
            df_cluster = pd.merge(df_cluster, mask_org_mun, on='MUNICÍPIO', how='left').fillna(0)
            df_cluster['pct_org'] = (df_cluster['Massa_Org'] / df_cluster['MASSA_COLETADA']) * 100
            # % para aterro
            df_mcf = df.groupby('MUNICÍPIO').apply(lambda x: (x['MASSA_COLETADA'] * x['DESTINO'].apply(determinar_mcf_por_destino)).sum() / x['MASSA_COLETADA'].sum() if x['MASSA_COLETADA'].sum()>0 else 0).reset_index(name='MCF_medio')
            df_cluster = pd.merge(df_cluster, df_mcf, on='MUNICÍPIO', how='left').fillna(0)

            features = df_cluster[['per_capita', 'pct_org', 'MCF_medio']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            if len(X_scaled) >= 4:
                n_clusters = min(4, len(X_scaled)//2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
                # PCA para visualização
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df_cluster['PCA1'] = X_pca[:,0]; df_cluster['PCA2'] = X_pca[:,1]
                df_cluster[['MUNICÍPIO','UF','Cluster','PCA1','PCA2','MASSA_COLETADA','per_capita','pct_org','MCF_medio']].to_csv(
                    os.path.join(pasta, f'cluster_municipios_{nome_base}.csv'), index=False)
                # Gráfico de clusters
                fig, ax = plt.subplots(figsize=(10,8))
                for cl in sorted(df_cluster['Cluster'].unique()):
                    subset = df_cluster[df_cluster['Cluster']==cl]
                    ax.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cl}', alpha=0.7, s=50)
                ax.set_title(f'Clusterização de Municípios – {nome_base}')
                ax.legend(); ax.grid(True, linestyle=':')
                plt.tight_layout()
                plt.savefig(os.path.join(pasta, f'cluster_plot_{nome_base}.png'), dpi=150)
                plt.close()

        # Projeção per capita (Brasil)
        print("   - Projeção per capita...")
        massa_atual = df['MASSA_COLETADA'].sum()
        pop_atual = df['POPULACAO_TOTAL'].sum() if 'POPULACAO_TOTAL' in df else 210000000
        if massa_atual > 0 and pop_atual > 0:
            taxa_pop = 0.01
            anos_proj = 10
            per_capita = massa_atual / pop_atual
            proj = []
            pop = pop_atual
            for i in range(1, anos_proj+1):
                pop = pop * (1 + taxa_pop)
                massa = pop * per_capita
                proj.append({'Ano': datetime.now().year + i, 'Populacao': pop, 'Massa_ton': massa})
            df_proj = pd.DataFrame(proj)
            df_proj.to_csv(os.path.join(pasta, f'projecao_percapita_{nome_base}.csv'), index=False)

        # Simulação de cenários (compostagem)
        print("   - Simulação de cenários...")
        df_evitado = calcular_evitado_por_municipio(df)
        if not df_evitado.empty:
            total_evitado = df_evitado['Evitado_Total'].sum()
            total_massa_org = df_evitado['Massa_Org_Seletiva'].sum()
            evitado_por_t = total_evitado / total_massa_org if total_massa_org > 0 else 0.5
            preco, _ = obter_cotacao_carbono()
            cambio, _ = obter_cotacao_euro()
            # Simula 10 anos
            massa_aterro = df[df['DESTINO'].apply(determinar_mcf_por_destino)>0]['MASSA_COLETADA'].sum()
            if massa_aterro > 0 and evitado_por_t > 0:
                resultados_sim = []
                for ano in range(1, 11):
                    fator = (1 + 0.15)**(ano-1)
                    massa_proj = massa_aterro * fator
                    co2_evitado = massa_proj * evitado_por_t
                    receita = co2_evitado * preco * cambio
                    resultados_sim.append({'Ano': datetime.now().year+ano, 'Massa_Desviada_t': massa_proj,
                                          'Receita_Anual_BRL': receita, 'Receita_Acumulada_BRL': receita if ano==1 else 0})
                df_sim = pd.DataFrame(resultados_sim)
                df_sim['Receita_Acumulada_BRL'] = df_sim['Receita_Anual_BRL'].cumsum()
                df_sim.to_csv(os.path.join(pasta, f'simulacao_receita_{nome_base}.csv'), index=False)

        # Cenários de expansão (cobertura)
        print("   - Cenários de expansão...")
        df_total = df.groupby('MUNICÍPIO').agg({'MASSA_COLETADA':'sum', 'UF':'first'}).reset_index()
        df_total.rename(columns={'MASSA_COLETADA':'Massa_Total'}, inplace=True)
        mask_org = df['TIPO_COLETA_EXECUTADA'].astype(str).str.contains("seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)
        df_seletiva = df[mask_org].groupby('MUNICÍPIO')['MASSA_COLETADA'].sum().reset_index().rename(columns={'MASSA_COLETADA':'Massa_Seletiva'})
        df_cobertura = pd.merge(df_total, df_seletiva, on='MUNICÍPIO', how='left').fillna(0)
        df_cobertura['Pct_Cobertura'] = (df_cobertura['Massa_Seletiva'] / df_cobertura['Massa_Total']) * 100
        df_cobertura['Possui_Seletiva'] = df_cobertura['Massa_Seletiva'] > 0
        df_cobertura.to_csv(os.path.join(pasta, f'cobertura_seletiva_{nome_base}.csv'), index=False)
        # Estatísticas de cobertura
        cobertura_stats = {
            'Total_Municipios': len(df_cobertura),
            'Com_Seletiva': df_cobertura['Possui_Seletiva'].sum(),
            'Sem_Seletiva': (~df_cobertura['Possui_Seletiva']).sum(),
            'Massa_Total_Brasil_t': df_cobertura['Massa_Total'].sum(),
            'Massa_Seletiva_Brasil_t': df_cobertura['Massa_Seletiva'].sum(),
            'Pct_Nacional': (df_cobertura['Massa_Seletiva'].sum() / df_cobertura['Massa_Total'].sum())*100 if df_cobertura['Massa_Total'].sum()>0 else 0
        }
        pd.DataFrame([cobertura_stats]).to_csv(os.path.join(pasta, f'cobertura_stats_{nome_base}.csv'), index=False)

        # ---------- 7.3 Aba 3: Diagnóstico de Emissões ----------
        print("🔥 Gerando Diagnóstico de Emissões...")

        # Emissões por município
        emissoes = []
        for mun in df['MUNICÍPIO'].unique():
            df_mun = df[df['MUNICÍPIO']==mun].copy()
            doc, docf, k = calcular_doc_k_ponderado(df_mun)
            df_mun['MCF'] = df_mun['DESTINO'].apply(determinar_mcf_por_destino)
            df_aterro = df_mun[(df_mun['MCF']>0) & (df_mun['MASSA_COLETADA']>0)].copy()
            if df_aterro.empty:
                continue
            massa = df_aterro['MASSA_COLETADA'].sum()
            mcf_medio = (df_aterro['MASSA_COLETADA'] * df_aterro['MCF']).sum() / massa
            co2_20 = calcular_co2eq_aterro_20anos(massa, mcf_medio, k, doc, docf)
            emissao_anual = co2_20 / 20.0
            pop = df_mun['POPULACAO_TOTAL'].iloc[0] if 'POPULACAO_TOTAL' in df_mun else 0
            pop = pop if pd.notna(pop) and pop>0 else 0
            intensidade = emissao_anual / massa if massa>0 else 0
            per_capita = (emissao_anual * 1000) / pop if pop>0 else 0
            uf = df_mun['UF'].iloc[0] if 'UF' in df_mun else 'N/A'
            emissoes.append({
                'MUNICÍPIO': mun, 'UF': uf,
                'Massa_Aterro_t': massa,
                'MCF_Medio': mcf_medio,
                'DOC_Medio': doc, 'DOCF_Medio': docf, 'k_Medio': k,
                'Emissao_Media_Anual_tCO2e': emissao_anual,
                'Intensidade_tCO2e_por_t': intensidade,
                'Emissao_per_capita_kgCO2e': per_capita
            })
        df_emissoes = pd.DataFrame(emissoes)
        df_emissoes.to_csv(os.path.join(pasta, f'emissoes_municipios_{nome_base}.csv'), index=False)

        if not df_emissoes.empty:
            # Limiares SBCE (estático)
            acima_10k = df_emissoes[df_emissoes['Emissao_Media_Anual_tCO2e'] > 10000]
            acima_25k = df_emissoes[df_emissoes['Emissao_Media_Anual_tCO2e'] > 25000]
            limiares = pd.DataFrame({
                'Cenario': ['Estatico_1_deposito'],
                'Acima_10k': [len(acima_10k)],
                'Acima_25k': [len(acima_25k)]
            })

            # Projeção contínua (Brasil)
            massa_total_br = df_emissoes['Massa_Aterro_t'].sum()
            if massa_total_br > 0:
                mcf_br = (df_emissoes['Massa_Aterro_t'] * df_emissoes['MCF_Medio']).sum() / massa_total_br
                doc_br = (df_emissoes['Massa_Aterro_t'] * df_emissoes['DOC_Medio']).sum() / massa_total_br
                docf_br = (df_emissoes['Massa_Aterro_t'] * df_emissoes['DOCF_Medio']).sum() / massa_total_br
                k_br = (df_emissoes['Massa_Aterro_t'] * df_emissoes['k_Medio']).sum() / massa_total_br
                df_cont = projetar_emissao_continua(massa_total_br, mcf_br, k_br, doc_br, docf_br)
                df_cont.to_csv(os.path.join(pasta, f'projecao_continua_20anos_{nome_base}.csv'), index=False)

                # Limiares contínuos (ano 20)
                df_cont_ano20 = df_emissoes.copy()
                def calc_cont_ano20(row):
                    return projetar_emissao_continua(row['Massa_Aterro_t'], row['MCF_Medio'],
                                                     row['k_Medio'], row['DOC_Medio'], row['DOCF_Medio']).iloc[-1]['Emissao_Anual'] if row['Massa_Aterro_t']>0 else 0
                df_cont_ano20['Emissao_Continua_Ano20'] = df_cont_ano20.apply(calc_cont_ano20, axis=1)
                acima_10k_cont = df_cont_ano20[df_cont_ano20['Emissao_Continua_Ano20'] > 10000]
                acima_25k_cont = df_cont_ano20[df_cont_ano20['Emissao_Continua_Ano20'] > 25000]
                limiares_cont = pd.DataFrame({
                    'Cenario': ['Continuo_20_anos'],
                    'Acima_10k': [len(acima_10k_cont)],
                    'Acima_25k': [len(acima_25k_cont)]
                })
                limiares = pd.concat([limiares, limiares_cont], ignore_index=True)
                df_cont_ano20[['MUNICÍPIO','UF','Emissao_Continua_Ano20','Massa_Aterro_t']].to_csv(
                    os.path.join(pasta, f'limiares_continuos_detalhado_{nome_base}.csv'), index=False)
            limiares.to_csv(os.path.join(pasta, f'limiares_sbce_{nome_base}.csv'), index=False)

            # Pareto das emissões
            df_ord_emis = df_emissoes.sort_values('Emissao_Media_Anual_tCO2e', ascending=False)
            df_ord_emis['pct_acum'] = df_ord_emis['Emissao_Media_Anual_tCO2e'].cumsum() / df_ord_emis['Emissao_Media_Anual_tCO2e'].sum() * 100
            df_ord_emis['pct_mun'] = (np.arange(len(df_ord_emis))+1) / len(df_ord_emis) * 100
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(df_ord_emis['pct_mun'], df_ord_emis['pct_acum'], color='#d62728', linewidth=2)
            ax.axhline(80, color='red', linestyle='--')
            ax.set_xlabel('% acumulado de municípios'); ax.set_ylabel('% acumulado das emissões')
            ax.set_title(f'Concentração das Emissões de Metano – {nome_base}')
            ax.grid(True, linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(pasta, f'pareto_emissoes_{nome_base}.png'), dpi=150)
            plt.close()

            # Top 20 emissores
            top20 = df_emissoes.nlargest(20, 'Emissao_Media_Anual_tCO2e')[['MUNICÍPIO','UF','Emissao_Media_Anual_tCO2e','Intensidade_tCO2e_por_t']]
            top20.to_csv(os.path.join(pasta, f'top20_emissores_{nome_base}.csv'), index=False)

            # Matriz de decisão (Massa x Intensidade)
            fig, ax = plt.subplots(figsize=(10,8))
            med_massa = df_emissoes['Massa_Aterro_t'].median()
            med_int = df_emissoes['Intensidade_tCO2e_por_t'].median()
            cores = {'Crítico':'red', 'Ineficiente':'orange', 'Referência':'green', 'Baixa Prioridade':'blue'}
            for _, row in df_emissoes.iterrows():
                if row['Massa_Aterro_t'] >= med_massa and row['Intensidade_tCO2e_por_t'] >= med_int:
                    cat = 'Crítico'
                elif row['Massa_Aterro_t'] < med_massa and row['Intensidade_tCO2e_por_t'] >= med_int:
                    cat = 'Ineficiente'
                elif row['Massa_Aterro_t'] >= med_massa and row['Intensidade_tCO2e_por_t'] < med_int:
                    cat = 'Referência'
                else:
                    cat = 'Baixa Prioridade'
                ax.scatter(row['Massa_Aterro_t'], row['Intensidade_tCO2e_por_t'], color=cores[cat], alpha=0.6, s=30, label=cat if cat not in [l.get_text() for l in ax.get_legend_handles_labels()[1]] else "")
            ax.axvline(med_massa, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(med_int, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Massa em Aterro (t/ano)'); ax.set_ylabel('Intensidade (tCO₂e/t)')
            ax.set_title(f'Matriz de Decisão – {nome_base}')
            ax.legend()
            ax.grid(True, linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(pasta, f'matriz_decisao_{nome_base}.png'), dpi=150)
            plt.close()

        print(f"✅ PROCESSAMENTO CONCLUÍDO para {nome_base}")
        print(f"📁 Arquivos gerados em: {pasta}")

    except Exception as e:
        print(f"❌ ERRO FATAL em {nome_base}: {e}")
        import traceback
        traceback.print_exc()

# =====================================================================
# 8. RESUMO CONSOLIDADO
# =====================================================================
print("\n" + "="*60)
print("📋 RESUMO CONSOLIDADO - TODOS OS ANOS PROCESSADOS")
print("="*60)

csv_files = glob.glob(os.path.join(pasta, '*.csv'))
if csv_files:
    print(f"\nTotal de arquivos CSV gerados: {len(csv_files)}")
    # Lista os arquivos agrupados
    for f in sorted(csv_files):
        print(f"  - {os.path.basename(f)}")
    print("\n✅ Todos os resultados estão disponíveis na pasta /content/")
    print("📥 Clique no ícone de pasta à esquerda para baixar os arquivos.")
else:
    print("Nenhum arquivo CSV foi gerado.")

print("\n🏁 FIM DO PROCESSAMENTO")
