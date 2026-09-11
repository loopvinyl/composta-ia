#B1

import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import requests
import os
from bs4 import BeautifulSoup
import re
from scipy.signal import fftconvolve
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf

# ========== IMPORTAÇÃO DA IA ==========
from utils.ia_classificacao import ClassificadorDestinoIA, classificar_destino_regra, normalizar_texto

# ========== CONFIGURAÇÃO DA PÁGINA ==========
st.set_page_config(
    page_title="Composta.IA - Potencial de Compostagem e Créditos de Carbono",
    layout="wide"
)

st.title("🌱 Composta.IA - Potencial de Compostagem e Créditos de Carbono (UNFCCC)")
st.markdown("""
Este aplicativo interpreta os **tipos de coleta executada** informados pelos municípios no SINISA
e avalia o **potencial técnico para compostagem** de resíduos sólidos urbanos,
utilizando **Inteligência Artificial** para padronizar os dados e a **metodologia UNFCCC A6.4-AMT-003** para o cálculo de emissões.

**Ferramenta de apoio à gestão pública** – desenvolvida para subsidiar o SINISA e políticas de resíduos sólidos.
""")

#B2
# =========================================================
# SELEÇÃO DE ANO
# =========================================================
ano_selecionado = st.selectbox(
    "Selecione o ano de referência:",
    ["2023", "2024"],
    index=1
)

# =========================================================
# URLs atualizadas para apontar para os dados dentro do seu repositório
# =========================================================
URLS_POR_ANO = {
    "2023": "https://raw.githubusercontent.com/loopvinyl/composta-ia/main/data/rsuBrasil_2023.xlsx",
    "2024": "https://raw.githubusercontent.com/loopvinyl/composta-ia/main/data/rsuBrasil_2024.xlsx"
}

# =========================================================
# FUNÇÕES DE COTAÇÃO
# =========================================================
def obter_cotacao_carbono():
    """Obtém cotação do carbono via Yahoo Finance, fallback €85,50."""
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            if 10 < preco < 200:
                return preco, "€", "Carbon Futures (CO2.L)", True, "Yahoo Finance"
    except:
        pass
    return 85.50, "€", "Referência", False, "Referência"

def obter_cotacao_euro_real():
    """Cotação EUR/BRL com APIs públicas."""
    try:
        resp = requests.get("https://economia.awesomeapi.com.br/last/EUR-BRL", timeout=10)
        if resp.status_code == 200:
            return float(resp.json()['EURBRL']['bid']), "R$", True, "AwesomeAPI"
    except:
        pass
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=10)
        if resp.status_code == 200:
            return resp.json()['rates']['BRL'], "R$", True, "ExchangeRate-API"
    except:
        pass
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas, preco_ton, moeda, taxa_cambio=1):
    return emissoes_evitadas * preco_ton * taxa_cambio

# Inicialização das cotações no session_state
if 'preco_carbono' not in st.session_state:
    preco, moeda, _, _, _ = obter_cotacao_carbono()
    st.session_state.preco_carbono = preco
    st.session_state.moeda_carbono = moeda
if 'taxa_cambio' not in st.session_state:
    cambio, moeda_r, _, _ = obter_cotacao_euro_real()
    st.session_state.taxa_cambio = cambio
    st.session_state.moeda_real = moeda_r

# B3

# =========================================================
# FORMATAÇÕES
# =========================================================
def formatar_br(numero, auto_precision=True, casas_override=None):
    if pd.isna(numero) or numero is None:
        return "N/A"
    try:
        numero = float(numero)
        if casas_override is not None:
            decimais = casas_override
        elif auto_precision:
            decimais = 2 if abs(numero) >= 1 else 4
        else:
            decimais = 2
        numero_arredondado = round(numero, decimais)
        if decimais == 0:
            return f"{numero_arredondado:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            formato = f"{{:,.{decimais}f}}"
            return formato.format(numero_arredondado).replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "N/A"

def formatar_numero_br(valor, decimais=None, auto_precision=True):
    if decimais is not None:
        return formatar_br(valor, auto_precision=False, casas_override=decimais)
    return formatar_br(valor, auto_precision=auto_precision, casas_override=None)

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(".", ",")

def formatar_massa_br(valor):
    if pd.isna(valor) or valor is None:
        return "Não informado"
    return f"{formatar_br(valor)} t"

def formatar_eixo_abreviado(x, pos):
    """Formata números grandes para exibir como Mi (milhões) ou Bi (bilhões)."""
    if x == 0:
        return "0"
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f} Bi"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f} Mi"
    if abs(x) >= 1e3:
        return f"{x/1e3:.1f} k"
    return f"{x:.0f}"

#B4

# =========================================================
# PARÂMETROS UNFCCC A6.4-AMT-003-v01.0 (2025) – Application B (Tropical Wet)
# =========================================================
GWP_CH4 = 28.0
GWP_N2O = 265.0
PHI_APPLICATION_B = 0.85
OX_SOIL_COVER = 0.383
F_METHANE_FRACTION = 0.5
MCF_DEFAULT_BULK = 0.8

ANOS_PROJECAO = 20
DIAS_PROJECAO = ANOS_PROJECAO * 365
T_ORGANICO = 25.0
DOC_PADRAO = 0.15
K_PADRAO = 0.07

# =========================================================
# FUNÇÃO PARA CALCULAR DOC, DOC_f e k PONDERADOS (VIA SINISA)
# =========================================================
def calcular_doc_k_ponderado(df_municipio):
    """
    Calcula DOC médio, DOC_f (fração que realmente se decompõe) e k (decay rate)
    com base na caracterização dos resíduos (colunas GTR1501 a GTR1507).
    Segue as Tabelas 7 e 10 da UNFCCC A6.4-AMT-003 (Tropical Wet).
    """
    doc_map = {
        'GTR1501': 0.15,
        'GTR1502': 0.00,
        'GTR1503': 0.00,
        'GTR1504': 0.00,
        'GTR1505': 0.40,
        'GTR1506': 0.24,
        'GTR1507': 0.10
    }
    docf_map = {
        'GTR1501': 0.7,
        'GTR1502': 0.0,
        'GTR1503': 0.0,
        'GTR1504': 0.0,
        'GTR1505': 0.5,
        'GTR1506': 0.5,
        'GTR1507': 0.1
    }
    k_map = {
        'GTR1501': 0.17,
        'GTR1502': 0.0,
        'GTR1503': 0.0,
        'GTR1504': 0.0,
        'GTR1505': 0.07,
        'GTR1506': 0.07,
        'GTR1507': 0.035
    }

    cols = [col for col in doc_map.keys() if col in df_municipio.columns]
    if not cols:
        return 0.15, 0.5, 0.07

    pct = pd.to_numeric(df_municipio[cols], errors='coerce').fillna(0)
    total_pct = pct.sum().sum()
    if total_pct <= 0:
        return 0.15, 0.5, 0.07

    doc_pond = sum(pct[col].sum() * doc_map.get(col, 0) for col in cols) / total_pct
    docf_pond = sum(pct[col].sum() * docf_map.get(col, 0) for col in cols) / total_pct
    k_pond = sum(pct[col].sum() * k_map.get(col, 0) for col in cols) / total_pct

    doc_pond = max(0.01, min(0.5, doc_pond))
    docf_pond = max(0.05, min(0.9, docf_pond))
    k_pond = max(0.01, min(0.5, k_pond))
    return doc_pond, docf_pond, k_pond

# =========================================================
# FUNÇÃO: FRAÇÃO ORGÂNICA DE REFERÊNCIA (Pimentel e Capanema, 2025)
# =========================================================
def calcular_fracao_organica_nacional(df):
    """
    Retorna a fração orgânica de referência para o Brasil.

    Referência: Pimentel e Capanema (2025), citados na tese (p. 44):
    "os resíduos orgânicos correspondem a mais da metade do total coletado
    nas cidades brasileiras, configurando um recurso estratégico que, se
    adequadamente tratado, pode ser convertido em fertilizantes, energia e
    insumos para a agricultura."

    Como muitos municípios não realizaram estudo de caracterização dos RSU
    nos últimos 5 anos (não preenchendo GTR1501/GTR1505 no SINISA), adota-se
    o valor conservador de 50% (piso de "mais da metade") para a fração
    orgânica dos RSU coletados no Brasil.

    O parâmetro 'df' é mantido na assinatura para compatibilidade e
    possíveis usos futuros (ex.: ponderação municipal quando houver dados).
    """
    return 0.50

# =========================================================
# FUNÇÃO DE CÁLCULO – ATERRO (BASELINE UNFCCC) - MODELO ANUAL (EQUAÇÃO 1)
# =========================================================
def calcular_co2eq_aterro_20anos(massa_t_ano, mcf, k_ano, doc_pond, docf_pond):
    if massa_t_ano <= 0 or mcf <= 0:
        return 0.0
    massa_kg = massa_t_ano * 1000
    ch4_pot_por_kg = (doc_pond * docf_pond * mcf * F_METHANE_FRACTION * (16/12) *
                      (1 - OX_SOIL_COVER) * PHI_APPLICATION_B)
    frac_decomposta = 1 - np.exp(-k_ano * ANOS_PROJECAO)
    ch4_total_kg = massa_kg * ch4_pot_por_kg * frac_decomposta
    co2eq_total_t = (ch4_total_kg * GWP_CH4) / 1000.0
    return co2eq_total_t

# =========================================================
# FUNÇÃO DE CÁLCULO – COMPOSTAGEM (UNFCCC TOOL13 / AMS-III.F)
# =========================================================
def calcular_co2eq_compostagem_UNFCCC(massa_t_ano):
    if massa_t_ano <= 0:
        return 0.0
    massa_kg = massa_t_ano * 1000
    ch4_kg = massa_kg * 0.002
    n2o_kg = massa_kg * 0.0002
    co2eq_t = (ch4_kg * GWP_CH4 + n2o_kg * GWP_N2O) / 1000.0
    return co2eq_t

def determinar_mcf_por_destino(destino, tipo_residuo='organico'):
    if pd.isna(destino):
        return 0.0
    destino_norm = normalizar_texto(destino)
    if "ATERRO SANITARIO" in destino_norm:
        if "GERENCIADO" in destino_norm or "COLETA" in destino_norm or "BIOGÁS" in destino_norm:
            mcf_base = 1.0
        else:
            mcf_base = 0.8
    elif "ATERRO CONTROLADO" in destino_norm:
        mcf_base = 0.4
    elif "LIXAO" in destino_norm or "VAZADOURO" in destino_norm:
        mcf_base = 0.4
    else:
        mcf_base = 0.0
    return mcf_base

# =========================================================
# FUNÇÃO: CALCULAR EVITADO POR MUNICÍPIO (PARÂMETROS ESPECÍFICOS) - CORRIGIDA
# =========================================================
@st.cache_data
def calcular_evitado_por_municipio(df, col_destino, col_massa):
    """
    Calcula o evitado (tCO2e) para cada município com coleta seletiva orgânica,
    utilizando os parâmetros específicos (DOC/k/MCF) de cada município.
    Retorna DataFrame com colunas: MUNICÍPIO, Massa_Org_Seletiva, Evitado_Total
    """
    resultados = []
    mask_org = df['TIPO_COLETA_EXECUTADA'].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True
    )
    df_org = df[mask_org].copy()
    if df_org.empty:
        return pd.DataFrame(columns=['MUNICÍPIO', 'Massa_Org_Seletiva', 'Evitado_Total'])
    for mun in df_org['MUNICÍPIO'].unique():
        df_mun = df[df['MUNICÍPIO'] == mun].copy()
        df_mun_org = df_org[df_org['MUNICÍPIO'] == mun].copy()
        massa_org = df_mun_org['MASSA_COLETADA'].sum()
        if massa_org == 0:
            continue
        doc, docf, k = calcular_doc_k_ponderado(df_mun)
        df_mun['MCF'] = df_mun[col_destino].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
        df_aterro = df_mun[df_mun['MCF'] > 0].copy()
        if not df_aterro.empty:
            massa_aterro = df_aterro['MASSA_COLETADA'].sum()
            mcf_medio = (df_aterro['MASSA_COLETADA'] * df_aterro['MCF']).sum() / massa_aterro if massa_aterro > 0 else 0.8
        else:
            mcf_medio = 0.8
        co2_aterro_org = calcular_co2eq_aterro_20anos(massa_org, mcf_medio, k, doc, docf)
        co2_compost_org = calcular_co2eq_compostagem_UNFCCC(massa_org)
        evitado = co2_aterro_org - co2_compost_org
        resultados.append({
            'MUNICÍPIO': mun,
            'Massa_Org_Seletiva': massa_org,
            'Evitado_Total': evitado
        })
    return pd.DataFrame(resultados)


#B5

# =========================================================
# FUNÇÕES DE PROJEÇÃO PER CAPITA E SIMULAÇÃO
# =========================================================
def projetar_residuos_per_capita(populacao_atual, massa_anual_atual,
                                 taxa_crescimento_pop=0.01, anos=10):
    if populacao_atual <= 0 or massa_anual_atual <= 0:
        raise ValueError("População e massa devem ser maiores que zero.")
    per_capita = massa_anual_atual / populacao_atual
    resultados = []
    pop = populacao_atual
    massa = massa_anual_atual
    for i in range(1, anos + 1):
        pop = pop * (1 + taxa_crescimento_pop)
        massa = pop * per_capita
        resultados.append({
            'Ano': datetime.now().year + i,
            'Populacao_Projetada': pop,
            'Massa_Projetada_ton': massa
        })
    return pd.DataFrame(resultados)

def plot_projecao_residuos(df_proj):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('População (habitantes)', color='blue')
    ax1.plot(df_proj['Ano'], df_proj['Populacao_Projetada'], 'o-', color='blue', linewidth=2, label='População')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Massa de Resíduos (toneladas/ano)', color='green')
    ax2.plot(df_proj['Ano'], df_proj['Massa_Projetada_ton'], 's-', color='green', linewidth=2, label='Massa')
    ax2.tick_params(axis='y', labelcolor='green')
    for i, row in df_proj.iterrows():
        ax1.annotate(formatar_br(row['Populacao_Projetada'], auto_precision=False, casas_override=0),
                    (row['Ano'], row['Populacao_Projetada']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
        ax2.annotate(formatar_br(row['Massa_Projetada_ton'], auto_precision=False, casas_override=0),
                    (row['Ano'], row['Massa_Projetada_ton']),
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='green')
    plt.title('Projeção de População e Geração de Resíduos', fontsize=14)
    fig.tight_layout()
    return fig

def simular_cenarios_compostagem(massa_aterro_ano,
                                 co2_evitado_por_tonelada,
                                 preco_carbono_atual,
                                 taxa_cambio,
                                 anos_projecao=10,
                                 taxa_crescimento_compostagem=0.10,
                                 inflacao_carbono=0.02):
    if massa_aterro_ano <= 0:
        raise ValueError("Massa de aterro deve ser maior que zero.")
    resultados = []
    massa_estatica = massa_aterro_ano
    for ano in range(1, anos_projecao + 1):
        fator_desvio = (1 + taxa_crescimento_compostagem) ** (ano - 1)
        massa_projetada = massa_aterro_ano * fator_desvio
        preco_atualizado = preco_carbono_atual * (1 + inflacao_carbono) ** (ano - 1)
        co2_evitado_estatico = massa_estatica * co2_evitado_por_tonelada
        co2_evitado_projetado = massa_projetada * co2_evitado_por_tonelada
        receita_estatico_brl = co2_evitado_estatico * preco_atualizado * taxa_cambio
        receita_projetado_brl = co2_evitado_projetado * preco_atualizado * taxa_cambio
        ganho_incremental = receita_projetado_brl - receita_estatico_brl
        resultados.append({
            'Ano': datetime.now().year + ano,
            'Massa_Desviada_Acumulada(t)': massa_projetada,
            'Receita_Anual_BRL': receita_projetado_brl,
            'Ganho_Adicional_BRL': ganho_incremental
        })
    df = pd.DataFrame(resultados)
    df['Receita_Acumulada_BRL'] = df['Receita_Anual_BRL'].cumsum()
    return df

def plot_simulacao_compostagem(df_sim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sim['Ano'], df_sim['Receita_Acumulada_BRL'], 'o-', color='green', linewidth=2, label='Receita Acumulada')
    ax.fill_between(df_sim['Ano'], 0, df_sim['Receita_Acumulada_BRL'], alpha=0.3, color='lightgreen')
    for i, row in df_sim.iterrows():
        ax.annotate(f"R$ {formatar_br(row['Receita_Acumulada_BRL'], auto_precision=False, casas_override=0)}",
                    (row['Ano'], row['Receita_Acumulada_BRL']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Receita Acumulada (R$)')
    ax.set_title('Projeção de Ganhos com Créditos de Carbono (Compostagem)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return fig

# =========================================================
# NOVA FUNÇÃO: PROJEÇÃO CONTÍNUA DE 20 ANOS (DEPÓSITOS ANUAIS REPETIDOS) - CORRIGIDA
# =========================================================
def projetar_emissao_continua(massa_anual_t, mcf, k, doc, docf, anos=20):
    """
    Projeta a emissão acumulada ao longo de 20 anos, considerando que a mesma
    quantidade de resíduos é depositada TODO ANO.
    
    Retorna um DataFrame com as colunas:
    - Ano
    - Emissao_Anual (tCO2e emitidas naquele ano específico)
    - Emissao_Acumulada (tCO2e totais acumuladas até aquele ano)
    """
    # GARANTE QUE A MASSA ESTÁ EM TONELADAS (se for muito pequena, assume que está em kt)
    if massa_anual_t > 0 and massa_anual_t < 1000:
        # Se for menor que 1000, provavelmente está em kt, converte para t
        massa_anual_t = massa_anual_t * 1000
        st.warning(f"⚠️ Massa convertida de kt para t: {massa_anual_t/1000:.0f} kt → {massa_anual_t:,.0f} t")
    
    if massa_anual_t <= 0 or mcf <= 0:
        return pd.DataFrame(columns=['Ano', 'Emissao_Anual', 'Emissao_Acumulada'])
    
    # Fator de emissão potencial (tCO2e por tonelada de resíduo depositado)
    # ch4_pot_kg é kg CH4 por kg de resíduo.
    # Multiplicando por GWP_CH4 obtemos kg CO2e por kg de resíduo.
    # Como 1 tonelada = 1000 kg, o fator em tCO2e por tonelada é o MESMO VALOR NUMÉRICO
    # (porque a tonelada tem 1000 kg, e o fator já é por kg).
    # Portanto, NÃO se divide por 1000 aqui.
    ch4_pot_kg = (doc * docf * mcf * F_METHANE_FRACTION * (16/12) *
                  (1 - OX_SOIL_COVER) * PHI_APPLICATION_B)
    fator_tco2_por_ton = ch4_pot_kg * GWP_CH4  # <--- CORREÇÃO: removido o /1000
    
    resultados = []
    emissao_acumulada_total = 0.0
    
    for ano in range(1, anos + 1):
        # Soma das emissões de TODAS as camadas depositadas até este ano
        emissao_ano = 0.0
        for i in range(1, ano + 1):
            anos_decomp = ano - i + 1  # idade da camada i no ano atual
            
            # Fração que se decompõe EXATAMENTE neste ano (diferença entre acumulado)
            fracao_ano = np.exp(-k * (anos_decomp - 1)) - np.exp(-k * anos_decomp)
            
            # Emissão desta camada específica neste ano
            emissao_camada = massa_anual_t * fator_tco2_por_ton * fracao_ano
            emissao_ano += emissao_camada
        
        emissao_acumulada_total += emissao_ano
        resultados.append({
            'Ano': datetime.now().year + ano,
            'Emissao_Anual': emissao_ano,
            'Emissao_Acumulada': emissao_acumulada_total
        })
    
    return pd.DataFrame(resultados)


#B6

# =========================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS - VERSÃO ROBUSTA (2023 e 2024)
# =========================================================
@st.cache_data
def load_data(ano):
    url = URLS_POR_ANO[ano]
    df_coleta = pd.read_excel(url, sheet_name="Manejo_Coleta_e_Destinação", header=12)
    df_caract = pd.read_excel(url, sheet_name="Manejo_Resíduos_Sólidos_Urbanos", header=12)

    # =========================================================
    # POPULAÇÃO DO MUNICÍPIO = COLUNA J (10ª coluna, índice 9)
    # da aba "Manejo_Resíduos_Sólidos_Urbanos", a partir da linha 14
    # (após header=12, esses valores já são dados do DataFrame)
    # =========================================================
    if df_caract.shape[1] >= 10:
        nome_col_j = df_caract.columns[9]
        df_caract = df_caract.rename(columns={nome_col_j: 'POPULACAO_TOTAL'})
        df_caract['POPULACAO_TOTAL'] = pd.to_numeric(
            df_caract['POPULACAO_TOTAL'], errors='coerce'
        ).fillna(0)
    else:
        st.warning("⚠️ A aba 'Manejo_Resíduos_Sólidos_Urbanos' não possui 10 colunas. Coluna J (população) não encontrada.")
        df_caract['POPULACAO_TOTAL'] = 0

    # =========================================================
    # POPULAÇÃO TOTAL DO BRASIL = SOMA BRUTA DA COLUNA J
    # (SEM FILTROS — todos os municípios cadastrados na aba)
    # =========================================================
    populacao_brasil_snis = df_caract['POPULACAO_TOTAL'].sum()

    # Total de municípios cadastrados na aba (para o card comparativo)
    total_municipios_caract = df_caract['Cod_IBGE'].nunique() if 'Cod_IBGE' in df_caract.columns else len(df_caract)

    cols_caract = ['Cod_IBGE', 'POPULACAO_TOTAL',
                   'GTR1501', 'GTR1502', 'GTR1503', 'GTR1504',
                   'GTR1505', 'GTR1506', 'GTR1507']
    cols_existentes = [col for col in cols_caract if col in df_caract.columns]
    df_caract_filtrado = df_caract[cols_existentes]
    df = pd.merge(df_coleta, df_caract_filtrado, on='Cod_IBGE', how='left')

    return df, populacao_brasil_snis, total_municipios_caract

df, POPULACAO_BRASIL_SNIS, TOTAL_MUNICIPIOS_CADASTRO = load_data(ano_selecionado)

# =========================================================
# MAPEAMENTO INTELIGENTE DE COLUNAS (por nome, não por índice)
# =========================================================
def encontrar_coluna(df, padroes):
    """
    Procura no DataFrame uma coluna cujo nome contenha (case insensitive)
    qualquer uma das strings em 'padroes'. Retorna o nome da primeira que encontrar.
    """
    for col in df.columns:
        for padrao in padroes:
            if padrao.lower() in col.lower():
                return col
    return None  # não encontrou

# Mapeamento usando palavras-chave (baseado nos nomes reais dos arquivos)
COL_MUNICIPIO = encontrar_coluna(df, ['município', 'municipio', 'nom_mun'])
COL_UF = encontrar_coluna(df, ['uf', 'estado', 'sigla'])
COL_CODIGO_ROTA = encontrar_coluna(df, ['código rota', 'codigo rota', 'rota', 'gtr1000'])
COL_TIPO_COLETA = encontrar_coluna(df, ['tipo de coleta', 'tipo coleta', 'coleta', 'gtr1001'])
COL_MASSA = encontrar_coluna(df, ['massa coletada', 'massa (t)', 'massa total', 'quantidade coletada', 'gtr1008'])
COL_DESTINO = encontrar_coluna(df, ['destino', 'unidade', 'local de destinação', 'gtr1011'])

# Se algum não for encontrado, exibe mensagem de erro e para
if None in [COL_MUNICIPIO, COL_UF, COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_MASSA, COL_DESTINO]:
    st.error("❌ Não foi possível identificar todas as colunas necessárias no arquivo. Verifique a estrutura do SINISA.")
    st.stop()

# Renomeia para padronização
df = df.rename(columns={
    COL_MUNICIPIO: "MUNICÍPIO",
    COL_TIPO_COLETA: "TIPO_COLETA_EXECUTADA",
    COL_MASSA: "MASSA_COLETADA",
    COL_UF: "UF",
    COL_DESTINO: "DESTINO"
})

# Atualiza as variáveis com os novos nomes (já estão padronizados)
COL_MUNICIPIO = "MUNICÍPIO"
COL_TIPO_COLETA = "TIPO_COLETA_EXECUTADA"
COL_MASSA = "MASSA_COLETADA"
COL_UF = "UF"
COL_DESTINO = "DESTINO"

# Força a coluna de massa a ser numérica (já que pode vir como texto)
df['MASSA_COLETADA'] = pd.to_numeric(df['MASSA_COLETADA'], errors='coerce').fillna(0)

# =========================================================
# CLASSIFICAÇÃO AUXILIAR DE COLETA
# =========================================================
def classificar_coleta(texto):
    if pd.isna(texto):
        return ("Não informado", False, False, "Tipo não informado")
    t = str(texto).lower()
    palavras = {
        "compostagem": ("Orgânico direto", True, True, "Coleta para compostagem"),
        "vermicompostagem": ("Orgânico direto", True, True, "Coleta para vermicompostagem"),
        "poda": ("Orgânico direto", True, True, "Resíduo vegetal limpo"),
        "galhada": ("Orgânico direto", True, True, "Resíduo vegetal limpo"),
        "verde": ("Orgânico direto", True, True, "Resíduo vegetal limpo"),
        "orgânica": ("Orgânico direto", True, True, "Orgânico segregado"),
        "domiciliar": ("Orgânico potencial", True, False, "Exige triagem"),
        "varrição": ("Inapto", False, False, "Alta contaminação"),
        "seletiva": ("Não orgânico", False, False, "Recicláveis")
    }
    for p, c in palavras.items():
        if p in t:
            return c
    return ("Indefinido", False, False, "Não classificado")

df_clean = df.dropna(subset=[COL_MUNICIPIO])
df_clean[COL_MUNICIPIO] = df_clean[COL_MUNICIPIO].astype(str).str.strip()
municipios = ["BRASIL – Todos os municípios"] + sorted(df_clean[COL_MUNICIPIO].unique())
municipio = st.selectbox("Selecione o município:", municipios)
df_mun = df_clean.copy() if municipio == municipios[0] else df_clean[df_clean[COL_MUNICIPIO] == municipio]

#B7

# =========================================================
# INICIALIZAÇÃO DA IA (PLN)
# =========================================================
with st.spinner("🤖 Inicializando o modelo de Inteligência Artificial..."):
    classificador_ia = ClassificadorDestinoIA()
    try:
        classificador_ia.carregar_ou_treinar(df_clean, col_texto=COL_DESTINO)
        st.success("✅ IA carregada com sucesso!")
    except Exception as e:
        st.warning(f"⚠️ Modelo não encontrado. Treinando com dados atuais... (pode levar alguns segundos)")
        classificador_ia.treinar_com_dados_snis(df_clean, col_texto=COL_DESTINO)
        st.success("✅ IA treinada e salva com sucesso!")

# =========================================================
# CRIAÇÃO DAS ABAS
# =========================================================
tab_tradicional, tab_ia, tab_diagnostico = st.tabs([
    "📊 Análise Tradicional (SINISA)",
    "🤖 Insights com Inteligência Artificial",
    "🔥 Diagnóstico de Emissões (Baseline)"
])


#B8

# =========================================================
# ABA TRADICIONAL (com cards de estatísticas no Panorama Nacional)
# =========================================================
with tab_tradicional:
    st.subheader(f"🇧🇷 Brasil — Síntese Nacional de RSU ({ano_selecionado})" if municipio == municipios[0] else f"📍 {municipio} - Ano {ano_selecionado}")

    if municipio == municipios[0]:
        st.markdown("---")
        st.markdown("### 📊 Panorama Nacional de Geração de Resíduos")
        st.markdown(f"**Dados do SINISA – {ano_selecionado}**")

        # =========================================================
        # CARDS DE ESTATÍSTICAS GERAIS
        # =========================================================
        total_municipios_snis = df_clean['MUNICÍPIO'].nunique()

        df_temp = df_clean.copy()
        df_temp['MCF'] = df_temp[COL_DESTINO].apply(
            lambda x: determinar_mcf_por_destino(x, 'organico') if pd.notna(x) else 0.0
        )
        municipios_com_aterro = df_temp[df_temp['MCF'] > 0]['MUNICÍPIO'].nunique()
        municipios_sem_aterro = total_municipios_snis - municipios_com_aterro

        # =========================================================
        # PAINEL DE MUNICÍPIOS — 4 CARDS
        # Explicita a diferença entre:
        #  (a) municípios que REPORTARAM coleta (aba Manejo_Coleta_e_Destinação)
        #  (b) TOTAL de municípios cadastrados no SINISA (aba Manejo_Resíduos_Sólidos_Urbanos)
        # =========================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏙️ Municípios que reportaram coleta", total_municipios_snis,
                    help="Municípios presentes na aba 'Manejo_Coleta_e_Destinação' (declararam pelo menos uma rota de coleta).")
        col2.metric("🇧🇷 Total de municípios no SINISA", TOTAL_MUNICIPIOS_CADASTRO,
                    help="Municípios cadastrados na aba 'Manejo_Resíduos_Sólidos_Urbanos' (todos os 5.570 municípios do Brasil).")
        col3.metric("🗑️ Municípios com envio para aterro", municipios_com_aterro,
                    help="Municípios que possuem pelo menos uma rota de coleta cujo destino final é aterro sanitário, controlado ou lixão.")
        col4.metric("📭 Sem envio para aterro (ou dados zerados)", municipios_sem_aterro)

        st.caption(f"""
        ℹ️ **Diferença importante:**
        - O SINISA {ano_selecionado} possui **{TOTAL_MUNICIPIOS_CADASTRO} municípios cadastrados** (aba de caracterização — todos os {TOTAL_MUNICIPIOS_CADASTRO} do Brasil).
        - **{total_municipios_snis}** reportaram efetivamente **rotas de coleta** (aba de coleta).
        - A diferença de **{TOTAL_MUNICIPIOS_CADASTRO - total_municipios_snis} municípios** são cidades que **não declararam nenhuma rota de coleta** — possivelmente dados ausentes ou não se aplicam.

        *Municípios com aterro = aqueles que possuem pelo menos uma rota de coleta cujo destino final é aterro sanitário, controlado ou lixão.*
        """)
        st.markdown("---")

        ocultar_transbordo_panorama = st.checkbox(
            "Ocultar transbordos no panorama",
            value=False,
            key="ocultar_transbordo_panorama",
            help="Exclui rotas cujo destino é 'Transbordo' para evitar dupla contagem e alinhar com a visão consolidada."
        )
        with st.spinner("Calculando estatísticas nacionais..."):
            df_panorama = df_clean.copy()
            if ocultar_transbordo_panorama:
                df_panorama = df_panorama[~df_panorama[COL_DESTINO].apply(
                    lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
                )]
            df_massa_mun = df_panorama.groupby('MUNICÍPIO').agg({
                'MASSA_COLETADA': 'sum',
                'POPULACAO_TOTAL': 'first'
            }).reset_index()
            df_massa_mun = df_massa_mun[
                (df_massa_mun['POPULACAO_TOTAL'] > 0) &
                (df_massa_mun['MASSA_COLETADA'] > 0)
            ].copy()
            if not df_massa_mun.empty:
                df_massa_mun['per_capita_kg'] = (df_massa_mun['MASSA_COLETADA'] / df_massa_mun['POPULACAO_TOTAL']) * 1000
                media = df_massa_mun['per_capita_kg'].mean()
                mediana = df_massa_mun['per_capita_kg'].median()
                q1 = df_massa_mun['per_capita_kg'].quantile(0.25)
                q3 = df_massa_mun['per_capita_kg'].quantile(0.75)
                minimo = df_massa_mun['per_capita_kg'].min()
                maximo = df_massa_mun['per_capita_kg'].max()

                # =========================================================
                # PER CAPITA NACIONAL AGREGADO (PONDERADO)
                # Massa total ÷ População total × 1000
                # - Massa: dos municípios que reportaram (respeitando "Ocultar transbordos")
                # - População: SOMA BRUTA da coluna J (todos os municípios cadastrados
                #   na aba 'Manejo_Resíduos_Sólidos_Urbanos' — sem filtros)
                # =========================================================
                massa_total_brasil_pc = df_massa_mun['MASSA_COLETADA'].sum()
                pop_total_brasil_pc = POPULACAO_BRASIL_SNIS  # 5.570 municípios, sem filtro
                per_capita_nacional = (massa_total_brasil_pc / pop_total_brasil_pc) * 1000 if pop_total_brasil_pc > 0 else 0

                # =========================================================
                # INDICADORES REAIS (SINISA) E ESTIMADOS (fração orgânica 50%)
                # =========================================================
                per_capita_dia = per_capita_nacional / 365.0 if per_capita_nacional > 0 else 0.0

                # Fração orgânica de referência (50%) — Pimentel e Capanema (2025):
                # "os resíduos orgânicos correspondem a mais da metade do total
                #  coletado nas cidades brasileiras". Adota-se o piso conservador 50%.
                fracao_organica_nacional = calcular_fracao_organica_nacional(df_panorama)

                per_capita_organico_ano = per_capita_nacional * fracao_organica_nacional
                per_capita_organico_dia = per_capita_dia * fracao_organica_nacional

                df_ordenado = df_massa_mun.sort_values('MASSA_COLETADA', ascending=False).copy()
                df_ordenado['massa_acumulada'] = df_ordenado['MASSA_COLETADA'].cumsum()
                massa_total = df_ordenado['MASSA_COLETADA'].sum()
                df_ordenado['pct_acumulado'] = (df_ordenado['massa_acumulada'] / massa_total) * 100
                df_ate_80 = df_ordenado[df_ordenado['pct_acumulado'] <= 80]
                pct_municipios_80 = (len(df_ate_80) / len(df_ordenado)) * 100
                df_ate_50 = df_ordenado[df_ordenado['pct_acumulado'] <= 50]
                pct_municipios_50 = (len(df_ate_50) / len(df_ordenado)) * 100

                # =========================================================
                # PAINEL 1 — DADOS REAIS DO SINISA (4 cards)
                # ⚖️ Massa total coletada | 👥 População | 📊 Per capita anual | 📆 Per capita diário
                # =========================================================
                st.markdown("##### 📋 Indicadores extraídos diretamente do SINISA")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "⚖️ Massa total coletada",
                    f"{formatar_br(massa_total_brasil_pc, auto_precision=False, casas_override=0)} t",
                    help="Soma da massa dos municípios que reportaram coleta (influenciada pela opção 'Ocultar transbordos')."
                )
                col2.metric(
                    "👥 População total (SINISA)",
                    f"{formatar_br(pop_total_brasil_pc, auto_precision=False, casas_override=0)} hab",
                    help=f"Soma da coluna J de TODOS os {TOTAL_MUNICIPIOS_CADASTRO} municípios cadastrados na aba 'Manejo_Resíduos_Sólidos_Urbanos' (sem filtros)."
                )
                col3.metric(
                    "📊 Per capita anual",
                    f"{formatar_br(per_capita_nacional, auto_precision=False, casas_override=0)} kg/hab/ano",
                    help="Massa total ÷ População total (SINISA) × 1000. Reflete a realidade nacional (municípios grandes pesam mais)."
                )
                col4.metric(
                    "📆 Per capita diário",
                    f"{formatar_br(per_capita_dia, auto_precision=False, casas_override=2)} kg/hab/dia",
                    help="Per capita anual ÷ 365. Indicador clássico de geração diária de RSU (≈ 1 kg/hab/dia)."
                )
                st.caption("📌 Valores obtidos de: https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/saneamento/sinisa")

                st.markdown("")

                # =========================================================
                # PAINEL 2 — ESTIMATIVAS (2 cards) em container com borda
                # 🌱 Orgânico anual | 🌱 Orgânico diário
                # =========================================================
                with st.container(border=True):
                    st.markdown("##### 🔬 Estimativas — fração orgânica de referência")
                    st.caption(
                        f"Os dois indicadores abaixo **não** vêm diretamente da planilha do SINISA. "
                        f"São **estimativas** calculadas multiplicando-se os indicadores reais acima "
                        f"pela fração orgânica de referência de "
                        f"{formatar_br(fracao_organica_nacional*100, auto_precision=False, casas_override=0)}% "
                        f"conforme Pimentel e Capanema (2025): os resíduos orgânicos correspondem "
                        f"a mais da metade do total coletado nas cidades brasileiras\" "
                        f"(restos de comida, vegetais e frutas). Adota-se o piso conservador de 50% "
                        f"porque muitos municípios não realizaram estudo de caracterização dos RSU "
                        f"nos últimos 5 anos e, portanto, não preencheram as colunas GTR1501/GTR1505 no SINISA."
                    )
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "🌱 Orgânico anual (estimado)",
                        f"{formatar_br(per_capita_organico_ano, auto_precision=False, casas_override=0)} kg/hab/ano",
                        help=(
                            f"Per capita anual × fração orgânica de referência "
                            f"({formatar_br(fracao_organica_nacional*100, auto_precision=False, casas_override=0)}%). "
                            "Referência: Pimentel e Capanema (2025)."
                        )
                    )
                    col2.metric(
                        "🌱 Orgânico diário (estimado)",
                        f"{formatar_br(per_capita_organico_dia*1000, auto_precision=False, casas_override=0)} g/hab/dia",
                        help=(
                            "Per capita diário × fração orgânica de referência (50%). "
                            "Equivale a ≈ 500 g de restos de comida, vegetais e frutas por habitante por dia."
                        )
                    )
                    st.caption("⚠️ Estimativas baseadas em referência bibliográfica, **não** substituem o estudo de caracterização de cada município.")

                # Gráfico de concentração (Pareto)
                fig_conc, ax_conc = plt.subplots(figsize=(12, 7))
                df_ordenado['pct_municipios'] = (np.arange(len(df_ordenado)) + 1) / len(df_ordenado) * 100
                ax_conc.plot(df_ordenado['pct_municipios'], df_ordenado['pct_acumulado'], color='#1f77b4', linewidth=3, label='Concentração real da massa')
                ax_conc.axhline(y=80, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='80% da massa total')
                ax_conc.axvline(x=pct_municipios_80, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
                ax_conc.annotate(f'{pct_municipios_80:.1f}% dos municípios\nconcentram 80% da massa', xy=(pct_municipios_80, 80), xytext=(pct_municipios_80 + 15, 60), arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=11, color='red', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
                ax_conc.plot([0, 100], [0, 100], color='gray', linestyle=':', alpha=0.5, label='Igualdade perfeita (referência)')
                ax_conc.set_xlabel('Percentual acumulado de municípios (%)', fontsize=12)
                ax_conc.set_ylabel('Percentual acumulado da massa total (%)', fontsize=12)
                ax_conc.set_title(f'Concentração da Massa de RSU – Brasil ({ano_selecionado})', fontsize=14)
                ax_conc.grid(True, linestyle=':', alpha=0.4)
                ax_conc.legend(loc='lower right')
                ax_conc.set_xlim(0, 100)
                ax_conc.set_ylim(0, 100)
                ax_conc.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
                ax_conc.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
                plt.tight_layout()
                st.pyplot(fig_conc)
                plt.close(fig_conc)

                legenda_extra = " (transbordos ocultados)" if ocultar_transbordo_panorama else ""

                st.caption(f"""
                📌 **Interpretação:** A curva demonstra que os **{formatar_br(pct_municipios_80, auto_precision=False, casas_override=1)}% maiores municípios** (em massa) concentram **80% de todo o RSU do Brasil{legenda_extra}**.

                **Per capita nacional:** {formatar_br(per_capita_nacional, auto_precision=False, casas_override=0)} kg/hab/ano
                = {formatar_br(per_capita_dia, auto_precision=False, casas_override=2)} kg/hab/dia
                — Massa total ({formatar_br(massa_total_brasil_pc, auto_precision=False, casas_override=0)} t) ÷ População total SINISA ({formatar_br(pop_total_brasil_pc, auto_precision=False, casas_override=0)} hab) × 1000.

                **Fração orgânica de referência (estimativa):** {formatar_br(fracao_organica_nacional*100, auto_precision=False, casas_override=0)}%
                (Pimentel e Capanema, 2025 — "mais da metade do total coletado nas cidades brasileiras",
                considerando restos de comida, vegetais e frutas)
                → **{formatar_br(per_capita_organico_ano, auto_precision=False, casas_override=0)} kg orgânico/hab/ano**
                ou **{formatar_br(per_capita_organico_dia*1000, auto_precision=False, casas_override=0)} g orgânico/hab/dia** (≈ 500 g/dia).
                """)
            else:
                st.warning("Dados insuficientes para calcular estatísticas nacionais.")

    st.markdown("---")
    st.subheader(f"🗺️ Para onde o resíduo está indo? (Destinação Final, {ano_selecionado})")
    ocultar_transbordo = st.checkbox("Ocultar transbordos", value=False)
    df_mun_dest = df_mun.copy()
    if ocultar_transbordo:
        df_mun_dest = df_mun_dest[~df_mun_dest[COL_DESTINO].apply(
            lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
        )]
    df_mun_dest["MASSA_FLOAT"] = pd.to_numeric(df_mun_dest[COL_MASSA], errors="coerce").fillna(0)
    massa_total_geral = df_mun_dest["MASSA_FLOAT"].sum()
    st.markdown(f"### Total de resíduos coletados: **{formatar_br(massa_total_geral, auto_precision=False, casas_override=0)} t**")
    st.markdown("#### 📊 Distribuição dos principais destinos")
    df_mun_dest['destino_agrupado'] = df_mun_dest[COL_DESTINO].apply(
        lambda x: classificador_ia.prever(x, threshold=0.3) if pd.notna(x) else "Indefinido"
    )
    agg_grafico = df_mun_dest.groupby('destino_agrupado')['MASSA_FLOAT'].sum().reset_index()
    agg_grafico = agg_grafico.sort_values('MASSA_FLOAT', ascending=False).head(8)
    fig_dest, ax_dest = plt.subplots(figsize=(10, 8))
    cores = plt.cm.Set3(np.linspace(0, 1, len(agg_grafico)))
    wedges, texts, autotexts = ax_dest.pie(
        agg_grafico['MASSA_FLOAT'],
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        colors=cores,
        textprops={'fontsize': 9},
        pctdistance=0.7,
    )
    ax_dest.legend(wedges, agg_grafico['destino_agrupado'],
                   title="Destino",
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   fontsize=9)
    ax_dest.axis('equal')
    plt.tight_layout()
    st.pyplot(fig_dest)
    plt.close(fig_dest)
    st.caption("📌 Classificação dos destinos feita pela IA (PLN) para padronizar as variações textuais do SINISA.")

    st.markdown("#### 📋 Detalhamento por rota de coleta")
    tabela_destino = df_mun_dest[[COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_DESTINO, "MASSA_FLOAT"]].copy()
    tabela_destino = tabela_destino.rename(columns={
        COL_CODIGO_ROTA: "Código Rota",
        COL_TIPO_COLETA: "Tipo de Coleta",
        COL_DESTINO: "Tipo de Unidade (SINISA)",
        "MASSA_FLOAT": "Massa (t)"
    })
    tabela_destino["%"] = (tabela_destino["Massa (t)"] / massa_total_geral) * 100 if massa_total_geral > 0 else 0
    tabela_destino["Massa (t)"] = tabela_destino["Massa (t)"].apply(formatar_numero_br)
    tabela_destino["%"] = tabela_destino["%"].apply(lambda x: formatar_numero_br(x, 1))
    st.dataframe(
        tabela_destino[["Código Rota", "Tipo de Coleta", "Tipo de Unidade (SINISA)", "Massa (t)", "%"]],
        use_container_width=True
    )
    st.caption("📌 Os dados refletem fielmente os registros do SINISA. A classificação dos destinos é feita pela IA.")

    if municipio == municipios[0]:
        st.markdown("---")
        st.subheader(f"📊 Distribuição dos resíduos por tipo de destino ({ano_selecionado})")
        ocultar_transbordo_dist = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_dist")
        df_dist = df_mun_dest.copy()
        if ocultar_transbordo_dist:
            df_dist = df_dist[~df_dist[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
        massa_total_dist = df_dist["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de resíduos coletados: **{formatar_br(massa_total_dist, auto_precision=False, casas_override=0)} t**")
        agg_destino = df_dist.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_destino = agg_destino.sort_values("MASSA_FLOAT", ascending=False)
        agg_destino["Percentual (%)"] = (agg_destino["MASSA_FLOAT"] / massa_total_dist) * 100 if massa_total_dist > 0 else 0
        agg_destino["Massa (t)"] = agg_destino["MASSA_FLOAT"].apply(formatar_numero_br)
        agg_destino["Percentual (%)"] = agg_destino["Percentual (%)"].apply(lambda x: formatar_numero_br(x, 2))
        st.dataframe(
            agg_destino.rename(columns={COL_DESTINO: "Tipo de Unidade (SINISA)"})[["Tipo de Unidade (SINISA)", "Massa (t)", "Percentual (%)"]],
            use_container_width=True
        )
        st.markdown("#### 📊 Principais destinos (gráfico)")
        top_destinos = agg_destino.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_destinos[COL_DESTINO], top_destinos["MASSA_FLOAT"], color='steelblue')
        ax.set_xlabel('Massa (t)')
        ax.set_title('Top 10 destinos de resíduos')
        ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Nota: a soma das massas pode exceder o total coletado devido a duplicidades nas rotas (ex.: transbordo e destino final).")

        st.markdown("---")
        st.subheader(f"🏳️ Coleta de RSU pelos estados do Brasil ({ano_selecionado})")
        ocultar_transbordo_est = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_est")
        df_estados = df_mun_dest.copy()
        if ocultar_transbordo_est:
            df_estados = df_estados[~df_estados[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
        massa_total_est = df_estados["MASSA_FLOAT"].sum()
        agg_estados = df_estados.groupby("UF")["MASSA_FLOAT"].sum().reset_index()
        agg_estados = agg_estados.sort_values("MASSA_FLOAT", ascending=False)
        agg_estados["%"] = (agg_estados["MASSA_FLOAT"] / massa_total_est) * 100 if massa_total_est > 0 else 0
        agg_estados["% acumulado"] = agg_estados["%"].cumsum()
        agg_estados["Massa (t)"] = agg_estados["MASSA_FLOAT"].apply(formatar_numero_br)
        agg_estados["%"] = agg_estados["%"].apply(lambda x: formatar_numero_br(x, 2))
        agg_estados["% acumulado"] = agg_estados["% acumulado"].apply(lambda x: formatar_numero_br(x, 2))
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                agg_estados.rename(columns={"UF": "Estado"})[["Estado", "Massa (t)", "%", "% acumulado"]],
                use_container_width=True
            )
        with col2:
            fig, ax = plt.subplots(figsize=(6, 8))
            top_estados = agg_estados.head(10)
            ax.barh(top_estados["UF"], top_estados["MASSA_FLOAT"], color='forestgreen')
            ax.set_xlabel('Massa (t)')
            ax.set_title('Top 10 estados')
            ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if municipio == municipios[0]:
        st.markdown("---")
        st.header(f"🏆 Mapeamento de Coleta Seletiva de Orgânicos ({ano_selecionado})")
        st.markdown("""
        Lista de todos os municípios que declararam possuir **coleta seletiva de resíduos orgânicos**,
        com a massa coletada e a **receita potencial anual com créditos de carbono** (compostagem - UNFCCC).
        """)
        with st.spinner("Consultando dados..."):
            mask_organicos = df_clean[COL_TIPO_COLETA].astype(str).str.contains(
                "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)
            df_org_ranking = df_clean[mask_organicos].copy()
            if df_org_ranking.empty:
                st.info("Nenhum município registrou coleta seletiva de resíduos orgânicos.")
            else:
                df_org_ranking["MASSA_FLOAT_RANK"] = pd.to_numeric(df_org_ranking[COL_MASSA], errors="coerce").fillna(0)
                num_municipios = df_org_ranking[COL_MUNICIPIO].nunique()
                total_massa_org = df_org_ranking["MASSA_FLOAT_RANK"].sum()
                massa_compostagem = df_org_ranking[df_org_ranking[COL_DESTINO].str.contains("COMPOSTAGEM", case=False, na=False)]["MASSA_FLOAT_RANK"].sum()
                massa_aterro = df_org_ranking[df_org_ranking[COL_DESTINO].str.contains("ATERRO", case=False, na=False)]["MASSA_FLOAT_RANK"].sum()
                if total_massa_org > 0:
                    pct_comp = (massa_compostagem / total_massa_org) * 100
                    pct_aterro = (massa_aterro / total_massa_org) * 100
                else:
                    pct_comp = pct_aterro = 0.0
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Municípios com coleta seletiva", num_municipios)
                col_m2.metric("Massa p/ Compostagem", f"{formatar_br(pct_comp, auto_precision=False, casas_override=1)}%")
                col_m3.metric("Massa p/ Aterro", f"{formatar_br(pct_aterro, auto_precision=False, casas_override=1)}%")
                ranking_data = df_org_ranking.groupby([COL_MUNICIPIO, "UF", COL_DESTINO])["MASSA_FLOAT_RANK"].sum().reset_index()
                mapeamento = []
                preco = st.session_state.preco_carbono
                cambio = st.session_state.taxa_cambio
                for (mun, uf), grupo in ranking_data.groupby([COL_MUNICIPIO, "UF"]):
                    massa_total_local = grupo["MASSA_FLOAT_RANK"].sum()
                    destinos = ", ".join(sorted(grupo[COL_DESTINO].unique()))
                    grupo["MCF"] = grupo[COL_DESTINO].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
                    grupo_aterro = grupo[grupo["MCF"] > 0]
                    massa_aterro_local = grupo_aterro["MASSA_FLOAT_RANK"].sum()
                    if massa_aterro_local > 0:
                        mcf_medio = (grupo_aterro["MASSA_FLOAT_RANK"] * grupo_aterro["MCF"]).sum() / massa_aterro_local
                    else:
                        mcf_medio = 0.8
                    receita_anual = 0.0
                    if massa_aterro_local > 0:
                        df_mun_caract = df_clean[df_clean[COL_MUNICIPIO] == mun]
                        doc_pond, docf_pond, k_pond = calcular_doc_k_ponderado(df_mun_caract)
                        co2eq_aterro = calcular_co2eq_aterro_20anos(massa_aterro_local, mcf_medio, k_pond, doc_pond, docf_pond)
                        co2eq_compostagem = calcular_co2eq_compostagem_UNFCCC(massa_aterro_local)
                        evitado_20anos = co2eq_aterro - co2eq_compostagem
                        receita_anual = (evitado_20anos / ANOS_PROJECAO) * preco * cambio
                    massa_total_municipio = df_clean[df_clean[COL_MUNICIPIO] == mun]['MASSA_COLETADA'].sum()
                    pct_org = (massa_total_local / massa_total_municipio) * 100 if massa_total_municipio > 0 else 0
                    mapeamento.append({
                        "Município": mun,
                        "UF": uf,
                        "Massa Total (t/ano)": massa_total_local,
                        "Massa para Aterro (t/ano)": massa_aterro_local,
                        "% da massa total": pct_org,
                        "Tipo(s) de Unidade (SINISA)": destinos,
                        "Receita Potencial (R$/ano)": receita_anual
                    })
                df_mapeamento = pd.DataFrame(mapeamento).sort_values("Massa Total (t/ano)", ascending=False)
                st.dataframe(
                    df_mapeamento.style.format({
                        "Massa Total (t/ano)": lambda x: formatar_numero_br(x, None),
                        "Massa para Aterro (t/ano)": lambda x: formatar_numero_br(x, None),
                        "% da massa total": lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%',
                        "Receita Potencial (R$/ano)": lambda x: f"R$ {formatar_numero_br(x, None)}"
                    }),
                    use_container_width=True,
                    height=600
                )
                st.caption("""
                - **Baseline (aterro)**: alinhado à UNFCCC A6.4-AMT-003 (Application B) – CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28.
                - **Cenário de compostagem**: UNFCCC TOOL13 / AMS-III.F – CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265.
                - **DOC e k**: calculados dinamicamente a partir da caracterização dos resíduos do SINISA (quando disponível).
                - **MCF**: ponderado pelos diferentes destinos (aterro sanitário, controlado, lixão) de acordo com a Tabela 8 do anexo.
                - **% da massa total**: percentual da massa total de RSU do município que é composta por orgânicos da coleta seletiva.
                - Receita potencial anual considerando o preço atual do carbono.
                """)

    st.markdown("---")
    st.subheader(f"♻️ Destinação da Coleta Seletiva de Resíduos Orgânicos ({ano_selecionado})")
    df_organicos = df_mun_dest[df_mun_dest[COL_TIPO_COLETA].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)].copy()
    if not df_organicos.empty:
        df_organicos["MASSA_FLOAT"] = pd.to_numeric(df_organicos[COL_MASSA], errors="coerce").fillna(0)
        ocultar_transbordo_org = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_org")
        df_mun_org = df_mun_dest.copy()
        if ocultar_transbordo_org:
            df_organicos = df_organicos[~df_organicos[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
            df_mun_org = df_mun_org[~df_mun_org[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
        total_organicos = df_organicos["MASSA_FLOAT"].sum()
        massa_total_geral_org = df_mun_org["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de orgânicos coletados seletivamente: **{formatar_br(total_organicos, auto_precision=False, casas_override=2)} t**")
        st.markdown("#### 📊 Composição da destinação dos orgânicos")
        agg_org_pie = df_organicos.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_org_pie = agg_org_pie.sort_values("MASSA_FLOAT", ascending=False)
        fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
        cores_pie = plt.cm.Set3(np.linspace(0, 1, len(agg_org_pie)))
        wedges, texts, autotexts = ax_pie.pie(
            agg_org_pie["MASSA_FLOAT"],
            labels=None,
            autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
            startangle=90,
            colors=cores_pie,
            textprops={'fontsize': 9},
            pctdistance=0.7,
        )
        ax_pie.legend(wedges, agg_org_pie[COL_DESTINO],
                      title="Destino",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=9)
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)
        st.markdown("#### 📋 Tabela – Destino da coleta de recicláveis orgânicos")
        agg_org = df_organicos.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_org = agg_org.sort_values("MASSA_FLOAT", ascending=False)
        agg_org["% do tipo"] = (agg_org["MASSA_FLOAT"] / total_organicos) * 100 if total_organicos > 0 else 0
        agg_org["% do total no ano"] = (agg_org["MASSA_FLOAT"] / massa_total_geral_org) * 100 if massa_total_geral_org > 0 else 0
        linhas = []
        for _, row in agg_org.iterrows():
            linhas.append({
                "Destino": row[COL_DESTINO],
                "Massa Anual (t)": formatar_numero_br(row["MASSA_FLOAT"], 2),
                "% do tipo": formatar_numero_br(row["% do tipo"], 2),
                "% do total no ano": formatar_numero_br(row["% do total no ano"], 4)
            })
        perc_total_tipo = (total_organicos / massa_total_geral_org) * 100 if massa_total_geral_org > 0 else 0
        linhas.append({
            "Destino": "Total do tipo",
            "Massa Anual (t)": formatar_numero_br(total_organicos, 2),
            "% do tipo": "100,00%",
            "% do total no ano": formatar_numero_br(perc_total_tipo, 4)
        })
        linhas.append({
            "Destino": "Total no ano",
            "Massa Anual (t)": formatar_numero_br(massa_total_geral_org, 2),
            "% do tipo": " - ",
            "% do total no ano": "100,00%"
        })
        df_resumo = pd.DataFrame(linhas)
        st.dataframe(df_resumo, use_container_width=True)
    else:
        st.info("ℹ️ Sem registros de coleta seletiva de orgânicos.")

    st.markdown("---")
    st.subheader(f"🌳 Destinação da coleta de podas e galhadas ({ano_selecionado})")
    df_podas = df_mun_dest[df_mun_dest[COL_TIPO_COLETA].astype(str).str.contains("áreas verdes públicas", case=False, na=False)].copy()
    if not df_podas.empty:
        df_podas["MASSA_FLOAT"] = pd.to_numeric(df_podas[COL_MASSA], errors="coerce").fillna(0)
        ocultar_transbordo_podas = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_podas")
        df_mun_podas = df_mun_dest.copy()
        if ocultar_transbordo_podas:
            df_podas = df_podas[~df_podas[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
            df_mun_podas = df_mun_podas[~df_mun_podas[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
        total_podas = df_podas["MASSA_FLOAT"].sum()
        massa_total_geral_podas = df_mun_podas["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de podas e galhadas coletadas: **{formatar_br(total_podas, auto_precision=False, casas_override=2)} t**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Participação no total", f"{formatar_br((total_podas/massa_total_geral_podas)*100 if massa_total_geral_podas>0 else 0, auto_precision=False, casas_override=2)}%")
        with col2:
            destino_principal = df_podas.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().idxmax() if not df_podas.empty else "N/A"
            st.metric("Destino principal", destino_principal)
        st.markdown("#### 📋 Tabela – Destino da coleta de podas e galhadas")
        agg_podas = df_podas.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_podas = agg_podas.sort_values("MASSA_FLOAT", ascending=False)
        agg_podas["% do tipo"] = (agg_podas["MASSA_FLOAT"] / total_podas) * 100 if total_podas > 0 else 0
        agg_podas["% do total no ano"] = (agg_podas["MASSA_FLOAT"] / massa_total_geral_podas) * 100 if massa_total_geral_podas > 0 else 0
        linhas_podas = []
        for _, row in agg_podas.iterrows():
            linhas_podas.append({
                "Destino": row[COL_DESTINO],
                "Massa Anual (t)": formatar_numero_br(row["MASSA_FLOAT"], 2),
                "% do tipo": formatar_numero_br(row["% do tipo"], 2),
                "% do total no ano": formatar_numero_br(row["% do total no ano"], 4)
            })
        perc_total_tipo_podas = (total_podas / massa_total_geral_podas) * 100 if massa_total_geral_podas > 0 else 0
        linhas_podas.append({
            "Destino": "Total do tipo",
            "Massa Anual (t)": formatar_numero_br(total_podas, 2),
            "% do tipo": "100,00%",
            "% do total no ano": formatar_numero_br(perc_total_tipo_podas, 4)
        })
        linhas_podas.append({
            "Destino": "Total no ano",
            "Massa Anual (t)": formatar_numero_br(massa_total_geral_podas, 2),
            "% do tipo": " - ",
            "% do total no ano": "100,00%"
        })
        df_resumo_podas = pd.DataFrame(linhas_podas)
        st.dataframe(df_resumo_podas, use_container_width=True)
    else:
        st.info("ℹ️ Sem registros de coleta de podas e galhadas.")

    st.markdown("---")
    st.caption(f"""
    Fonte: SINISA (ano {ano_selecionado}) | **Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F)** | IPCC AR5 (GWP-100)
    Baseline (aterro): CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28 | Compostagem: CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265
    DOC/k: ponderados pela caracterização dos resíduos do SINISA (quando disponível) | Cotações em tempo real via Yahoo Finance e APIs de câmbio.
    Fração orgânica de referência: 50% (Pimentel e Capanema, 2025).
    """)


#B9

# =========================================================
# ABA DE IA (com todas as correções de MUNICÍPIO)
# =========================================================
with tab_ia:
    st.header("🧠 Insights com Inteligência Artificial")
    st.markdown("""
    Aqui você pode explorar análises avançadas utilizando técnicas de Inteligência Artificial:
    - **Classificação de destinos** com Processamento de Linguagem Natural (PLN)
    - **Projeção de geração de resíduos per capita** com base no crescimento populacional (município ou Brasil)
    - **Simulação de cenários de compostagem** e potencial de ganhos com créditos de carbono (município ou Brasil)
    - **Clusterização de municípios** por perfil de resíduos (K-Means)
    - **Análise de cobertura** da coleta seletiva de orgânicos e cenários de expansão
    """)

    # --- Classificação PLN ---
    st.subheader("📋 Classificação Inteligente de Destinos (PLN)")
    st.markdown("""
    O SINISA apresenta **diversas variações textuais** para descrever o mesmo destino
    (ex: "Aterro Sanitário", "AS", "Aterro Sani.", "Aterro – Gerenciado").

    O **Composta.IA** utiliza um modelo de **Regressão Logística com TF-IDF** para:
    - ✅ Generalizar padrões textuais com alta acurácia
    - 🔍 Exibir o nível de confiança de cada classificação
    - 🛡️ Recair para regras manuais quando a confiança é baixa (fallback seguro)
    """)
    amostras = df_clean[COL_DESTINO].dropna().sample(min(20, len(df_clean))).tolist()
    dados_comparacao = []
    for texto in amostras:
        classe_regra = classificar_destino_regra(texto)
        classe_ia = classificador_ia.prever(texto, threshold=0.3)
        if classificador_ia.pipeline is not None:
            texto_norm = normalizar_texto(texto)
            probs = classificador_ia.pipeline.predict_proba([texto_norm])[0]
            confianca = max(probs) * 100
        else:
            confianca = 0.0
        dados_comparacao.append({
            "Texto Original": texto[:50] + "..." if len(texto) > 50 else texto,
            "Regra (Manual)": classe_regra,
            "IA (Predição)": classe_ia,
            "Confiança da IA": f"{confianca:.1f}%",
            "Correção?": "✅" if classe_regra != classe_ia else "➖"
        })
    df_comparacao = pd.DataFrame(dados_comparacao)
    st.dataframe(df_comparacao, use_container_width=True, height=400)

    st.subheader("📊 Distribuição Nacional de Destinos (Classificação por IA)")
    @st.cache_data
    def classificar_todos_destinos(df, col_destino):
        return df[col_destino].apply(lambda x: classificador_ia.prever(x, threshold=0.3))
    with st.s
