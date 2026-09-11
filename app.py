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

URLS_POR_ANO = {
    "2023": "https://raw.githubusercontent.com/loopvinyl/composta-ia/main/data/rsuBrasil_2023.xlsx",
    "2024": "https://raw.githubusercontent.com/loopvinyl/composta-ia/main/data/rsuBrasil_2024.xlsx"
}

# =========================================================
# FUNÇÕES DE COTAÇÃO
# =========================================================
def obter_cotacao_carbono():
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
# PARÂMETROS UNFCCC
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

def calcular_doc_k_ponderado(df_municipio):
    doc_map = {
        'GTR1501': 0.15, 'GTR1502': 0.00, 'GTR1503': 0.00, 'GTR1504': 0.00,
        'GTR1505': 0.40, 'GTR1506': 0.24, 'GTR1507': 0.10
    }
    docf_map = {
        'GTR1501': 0.7, 'GTR1502': 0.0, 'GTR1503': 0.0, 'GTR1504': 0.0,
        'GTR1505': 0.5, 'GTR1506': 0.5, 'GTR1507': 0.1
    }
    k_map = {
        'GTR1501': 0.17, 'GTR1502': 0.0, 'GTR1503': 0.0, 'GTR1504': 0.0,
        'GTR1505': 0.07, 'GTR1506': 0.07, 'GTR1507': 0.035
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

def calcular_fracao_organica_nacional(df):
    return 0.50

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

@st.cache_data
def calcular_evitado_por_municipio(df, col_destino, col_massa):
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

def simular_cenarios_compostagem(massa_aterro_ano, co2_evitado_por_tonelada,
                                 preco_carbono_atual, taxa_cambio, anos_projecao=10,
                                 taxa_crescimento_compostagem=0.10, inflacao_carbono=0.02):
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

def projetar_emissao_continua(massa_anual_t, mcf, k, doc, docf, anos=20):
    if massa_anual_t > 0 and massa_anual_t < 1000:
        massa_anual_t = massa_anual_t * 1000
        st.warning(f"⚠️ Massa convertida de kt para t: {massa_anual_t/1000:.0f} kt → {massa_anual_t:,.0f} t")
    if massa_anual_t <= 0 or mcf <= 0:
        return pd.DataFrame(columns=['Ano', 'Emissao_Anual', 'Emissao_Acumulada'])
    ch4_pot_kg = (doc * docf * mcf * F_METHANE_FRACTION * (16/12) *
                  (1 - OX_SOIL_COVER) * PHI_APPLICATION_B)
    fator_tco2_por_ton = ch4_pot_kg * GWP_CH4
    resultados = []
    emissao_acumulada_total = 0.0
    for ano in range(1, anos + 1):
        emissao_ano = 0.0
        for i in range(1, ano + 1):
            anos_decomp = ano - i + 1
            fracao_ano = np.exp(-k * (anos_decomp - 1)) - np.exp(-k * anos_decomp)
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
@st.cache_data
def load_data(ano):
    url = URLS_POR_ANO[ano]
    df_coleta = pd.read_excel(url, sheet_name="Manejo_Coleta_e_Destinação", header=12)
    df_caract = pd.read_excel(url, sheet_name="Manejo_Resíduos_Sólidos_Urbanos", header=12)
    if df_caract.shape[1] >= 10:
        nome_col_j = df_caract.columns[9]
        df_caract = df_caract.rename(columns={nome_col_j: 'POPULACAO_TOTAL'})
        df_caract['POPULACAO_TOTAL'] = pd.to_numeric(df_caract['POPULACAO_TOTAL'], errors='coerce').fillna(0)
    else:
        st.warning("⚠️ A aba 'Manejo_Resíduos_Sólidos_Urbanos' não possui 10 colunas. Coluna J (população) não encontrada.")
        df_caract['POPULACAO_TOTAL'] = 0
    populacao_brasil_snis = df_caract['POPULACAO_TOTAL'].sum()
    total_municipios_caract = df_caract['Cod_IBGE'].nunique() if 'Cod_IBGE' in df_caract.columns else len(df_caract)
    cols_caract = ['Cod_IBGE', 'POPULACAO_TOTAL',
                   'GTR1501', 'GTR1502', 'GTR1503', 'GTR1504',
                   'GTR1505', 'GTR1506', 'GTR1507']
    cols_existentes = [col for col in cols_caract if col in df_caract.columns]
    df_caract_filtrado = df_caract[cols_existentes]
    df = pd.merge(df_coleta, df_caract_filtrado, on='Cod_IBGE', how='left')
    return df, populacao_brasil_snis, total_municipios_caract

df, POPULACAO_BRASIL_SNIS, TOTAL_MUNICIPIOS_CADASTRO = load_data(ano_selecionado)

def encontrar_coluna(df, padroes):
    for col in df.columns:
        for padrao in padroes:
            if padrao.lower() in col.lower():
                return col
    return None

COL_MUNICIPIO = encontrar_coluna(df, ['município', 'municipio', 'nom_mun'])
COL_UF = encontrar_coluna(df, ['uf', 'estado', 'sigla'])
COL_CODIGO_ROTA = encontrar_coluna(df, ['código rota', 'codigo rota', 'rota', 'gtr1000'])
COL_TIPO_COLETA = encontrar_coluna(df, ['tipo de coleta', 'tipo coleta', 'coleta', 'gtr1001'])
COL_MASSA = encontrar_coluna(df, ['massa coletada', 'massa (t)', 'massa total', 'quantidade coletada', 'gtr1008'])
COL_DESTINO = encontrar_coluna(df, ['destino', 'unidade', 'local de destinação', 'gtr1011'])

if None in [COL_MUNICIPIO, COL_UF, COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_MASSA, COL_DESTINO]:
    st.error("❌ Não foi possível identificar todas as colunas necessárias no arquivo. Verifique a estrutura do SINISA.")
    st.stop()

df = df.rename(columns={
    COL_MUNICIPIO: "MUNICÍPIO", COL_TIPO_COLETA: "TIPO_COLETA_EXECUTADA",
    COL_MASSA: "MASSA_COLETADA", COL_UF: "UF", COL_DESTINO: "DESTINO"
})
COL_MUNICIPIO = "MUNICÍPIO"
COL_TIPO_COLETA = "TIPO_COLETA_EXECUTADA"
COL_MASSA = "MASSA_COLETADA"
COL_UF = "UF"
COL_DESTINO = "DESTINO"
df['MASSA_COLETADA'] = pd.to_numeric(df['MASSA_COLETADA'], errors='coerce').fillna(0)

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
with st.spinner("🤖 Inicializando o modelo de Inteligência Artificial..."):
    classificador_ia = ClassificadorDestinoIA()
    try:
        classificador_ia.carregar_ou_treinar(df_clean, col_texto=COL_DESTINO)
        st.success("✅ IA carregada com sucesso!")
    except Exception as e:
        st.warning(f"⚠️ Modelo não encontrado. Treinando com dados atuais... (pode levar alguns segundos)")
        classificador_ia.treinar_com_dados_snis(df_clean, col_texto=COL_DESTINO)
        st.success("✅ IA treinada e salva com sucesso!")

tab_tradicional, tab_ia, tab_diagnostico = st.tabs([
    "📊 Análise Tradicional (SINISA)",
    "🤖 Insights com Inteligência Artificial",
    "🔥 Diagnóstico de Emissões (Baseline)"
])


#B8 — ABA TRADICIONAL (idêntica à original)
with tab_tradicional:
    st.subheader(f"🇧🇷 Brasil — Síntese Nacional de RSU ({ano_selecionado})" if municipio == municipios[0] else f"📍 {municipio} - Ano {ano_selecionado}")

    if municipio == municipios[0]:
        st.markdown("---")
        st.markdown("### 📊 Panorama Nacional de Geração de Resíduos")
        st.markdown(f"**Dados do SINISA – {ano_selecionado}**")
        total_municipios_snis = df_clean['MUNICÍPIO'].nunique()
        df_temp = df_clean.copy()
        df_temp['MCF'] = df_temp[COL_DESTINO].apply(
            lambda x: determinar_mcf_por_destino(x, 'organico') if pd.notna(x) else 0.0
        )
        municipios_com_aterro = df_temp[df_temp['MCF'] > 0]['MUNICÍPIO'].nunique()
        municipios_sem_aterro = total_municipios_snis - municipios_com_aterro
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏙️ Municípios que reportaram coleta", total_municipios_snis,
                    help="Municípios presentes na aba 'Manejo_Coleta_e_Destinação'.")
        col2.metric("🇧🇷 Total de municípios no SINISA", TOTAL_MUNICIPIOS_CADASTRO,
                    help="Municípios cadastrados na aba 'Manejo_Resíduos_Sólidos_Urbanos'.")
        col3.metric("🗑️ Municípios com envio para aterro", municipios_com_aterro,
                    help="Municípios com pelo menos uma rota cujo destino final é aterro/lixão.")
        col4.metric("📭 Sem envio para aterro (ou dados zerados)", municipios_sem_aterro)
        st.caption(f"""
        ℹ️ **Diferença importante:**
        - O SINISA {ano_selecionado} possui **{TOTAL_MUNICIPIOS_CADASTRO} municípios cadastrados**.
        - **{total_municipios_snis}** reportaram efetivamente **rotas de coleta**.
        - A diferença de **{TOTAL_MUNICIPIOS_CADASTRO - total_municipios_snis} municípios** não declararam nenhuma rota de coleta.
        """)
        st.markdown("---")
        ocultar_transbordo_panorama = st.checkbox(
            "Ocultar transbordos no panorama", value=False, key="ocultar_transbordo_panorama",
            help="Exclui rotas cujo destino é 'Transbordo'.")
        with st.spinner("Calculando estatísticas nacionais..."):
            df_panorama = df_clean.copy()
            if ocultar_transbordo_panorama:
                df_panorama = df_panorama[~df_panorama[COL_DESTINO].apply(
                    lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
            df_massa_mun = df_panorama.groupby('MUNICÍPIO').agg({
                'MASSA_COLETADA': 'sum', 'POPULACAO_TOTAL': 'first'
            }).reset_index()
            df_massa_mun = df_massa_mun[(df_massa_mun['POPULACAO_TOTAL'] > 0) & (df_massa_mun['MASSA_COLETADA'] > 0)].copy()
            if not df_massa_mun.empty:
                df_massa_mun['per_capita_kg'] = (df_massa_mun['MASSA_COLETADA'] / df_massa_mun['POPULACAO_TOTAL']) * 1000
                massa_total_brasil_pc = df_massa_mun['MASSA_COLETADA'].sum()
                pop_total_brasil_pc = POPULACAO_BRASIL_SNIS
                per_capita_nacional = (massa_total_brasil_pc / pop_total_brasil_pc) * 1000 if pop_total_brasil_pc > 0 else 0
                per_capita_dia = per_capita_nacional / 365.0 if per_capita_nacional > 0 else 0.0
                fracao_organica_nacional = calcular_fracao_organica_nacional(df_panorama)
                per_capita_organico_ano = per_capita_nacional * fracao_organica_nacional
                per_capita_organico_dia = per_capita_dia * fracao_organica_nacional
                df_ordenado = df_massa_mun.sort_values('MASSA_COLETADA', ascending=False).copy()
                df_ordenado['massa_acumulada'] = df_ordenado['MASSA_COLETADA'].cumsum()
                massa_total = df_ordenado['MASSA_COLETADA'].sum()
                df_ordenado['pct_acumulado'] = (df_ordenado['massa_acumulada'] / massa_total) * 100
                df_ate_80 = df_ordenado[df_ordenado['pct_acumulado'] <= 80]
                pct_municipios_80 = (len(df_ate_80) / len(df_ordenado)) * 100
                st.markdown("##### 📋 Indicadores extraídos diretamente do SINISA")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⚖️ Massa total coletada", f"{formatar_br(massa_total_brasil_pc, auto_precision=False, casas_override=0)} t")
                col2.metric("👥 População total (SINISA)", f"{formatar_br(pop_total_brasil_pc, auto_precision=False, casas_override=0)} hab")
                col3.metric("📊 Per capita anual", f"{formatar_br(per_capita_nacional, auto_precision=False, casas_override=0)} kg/hab/ano")
                col4.metric("📆 Per capita diário", f"{formatar_br(per_capita_dia, auto_precision=False, casas_override=2)} kg/hab/dia")
                st.caption("📌 Valores obtidos de: https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/saneamento/sinisa")
                st.markdown("")
                with st.container(border=True):
                    st.markdown("##### 🔬 Estimativas — fração orgânica de referência")
                    st.caption("Estimativas baseadas na fração orgânica de 50% (Pimentel e Capanema, 2025).")
                    col1, col2 = st.columns(2)
                    col1.metric("🌱 Orgânico anual (estimado)", f"{formatar_br(per_capita_organico_ano, auto_precision=False, casas_override=0)} kg/hab/ano")
                    col2.metric("🌱 Orgânico diário (estimado)", f"{formatar_br(per_capita_organico_dia*1000, auto_precision=False, casas_override=0)} g/hab/dia")
                    st.caption("⚠️ Estimativas baseadas em referência bibliográfica.")
                fig_conc, ax_conc = plt.subplots(figsize=(12, 7))
                df_ordenado['pct_municipios'] = (np.arange(len(df_ordenado)) + 1) / len(df_ordenado) * 100
                ax_conc.plot(df_ordenado['pct_municipios'], df_ordenado['pct_acumulado'], color='#1f77b4', linewidth=3, label='Concentração real da massa')
                ax_conc.axhline(y=80, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='80% da massa total')
                ax_conc.axvline(x=pct_municipios_80, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
                ax_conc.annotate(f'{pct_municipios_80:.1f}% dos municípios\nconcentram 80% da massa',
                                 xy=(pct_municipios_80, 80), xytext=(pct_municipios_80 + 15, 60),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                 fontsize=11, color='red', ha='left',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
                ax_conc.plot([0, 100], [0, 100], color='gray', linestyle=':', alpha=0.5, label='Igualdade perfeita (referência)')
                ax_conc.set_xlabel('Percentual acumulado de municípios (%)', fontsize=12)
                ax_conc.set_ylabel('Percentual acumulado da massa total (%)', fontsize=12)
                ax_conc.set_title(f'Concentração da Massa de RSU – Brasil ({ano_selecionado})', fontsize=14)
                ax_conc.grid(True, linestyle=':', alpha=0.4)
                ax_conc.legend(loc='lower right')
                ax_conc.set_xlim(0, 100); ax_conc.set_ylim(0, 100)
                ax_conc.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
                ax_conc.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
                plt.tight_layout(); st.pyplot(fig_conc); plt.close(fig_conc)
            else:
                st.warning("Dados insuficientes para calcular estatísticas nacionais.")

    st.markdown("---")
    st.subheader(f"🗺️ Para onde o resíduo está indo? (Destinação Final, {ano_selecionado})")
    ocultar_transbordo = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_tab1_principal")
    df_mun_dest = df_mun.copy()
    if ocultar_transbordo:
        df_mun_dest = df_mun_dest[~df_mun_dest[COL_DESTINO].apply(
            lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
    df_mun_dest["MASSA_FLOAT"] = pd.to_numeric(df_mun_dest[COL_MASSA], errors="coerce").fillna(0)
    massa_total_geral = df_mun_dest["MASSA_FLOAT"].sum()
    st.markdown(f"### Total de resíduos coletados: **{formatar_br(massa_total_geral, auto_precision=False, casas_override=0)} t**")
    st.markdown("#### 📊 Distribuição dos principais destinos")
    df_mun_dest['destino_agrupado'] = df_mun_dest[COL_DESTINO].apply(
        lambda x: classificador_ia.prever(x, threshold=0.3) if pd.notna(x) else "Indefinido")
    agg_grafico = df_mun_dest.groupby('destino_agrupado')['MASSA_FLOAT'].sum().reset_index()
    agg_grafico = agg_grafico.sort_values('MASSA_FLOAT', ascending=False).head(8)
    fig_dest, ax_dest = plt.subplots(figsize=(10, 8))
    cores = plt.cm.Set3(np.linspace(0, 1, len(agg_grafico)))
    wedges, texts, autotexts = ax_dest.pie(agg_grafico['MASSA_FLOAT'], labels=None,
                                            autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
                                            startangle=90, colors=cores, textprops={'fontsize': 9}, pctdistance=0.7)
    ax_dest.legend(wedges, agg_grafico['destino_agrupado'], title="Destino",
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax_dest.axis('equal'); plt.tight_layout(); st.pyplot(fig_dest); plt.close(fig_dest)
    st.caption("📌 Classificação dos destinos feita pela IA (PLN).")

    st.markdown("#### 📋 Detalhamento por rota de coleta")
    tabela_destino = df_mun_dest[[COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_DESTINO, "MASSA_FLOAT"]].copy()
    tabela_destino = tabela_destino.rename(columns={COL_CODIGO_ROTA: "Código Rota",
                                                     COL_TIPO_COLETA: "Tipo de Coleta",
                                                     COL_DESTINO: "Tipo de Unidade (SINISA)",
                                                     "MASSA_FLOAT": "Massa (t)"})
    tabela_destino["%"] = (tabela_destino["Massa (t)"] / massa_total_geral) * 100 if massa_total_geral > 0 else 0
    tabela_destino["Massa (t)"] = tabela_destino["Massa (t)"].apply(formatar_numero_br)
    tabela_destino["%"] = tabela_destino["%"].apply(lambda x: formatar_numero_br(x, 1))
    st.dataframe(tabela_destino[["Código Rota", "Tipo de Coleta", "Tipo de Unidade (SINISA)", "Massa (t)", "%"]], use_container_width=True)

    if municipio == municipios[0]:
        st.markdown("---")
        st.subheader(f"📊 Distribuição dos resíduos por tipo de destino ({ano_selecionado})")
        ocultar_transbordo_dist = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_dist")
        df_dist = df_mun_dest.copy()
        if ocultar_transbordo_dist:
            df_dist = df_dist[~df_dist[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
        massa_total_dist = df_dist["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de resíduos coletados: **{formatar_br(massa_total_dist, auto_precision=False, casas_override=0)} t**")
        agg_destino = df_dist.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_destino = agg_destino.sort_values("MASSA_FLOAT", ascending=False)
        agg_destino["Percentual (%)"] = (agg_destino["MASSA_FLOAT"] / massa_total_dist) * 100 if massa_total_dist > 0 else 0
        agg_destino["Massa (t)"] = agg_destino["MASSA_FLOAT"].apply(formatar_numero_br)
        agg_destino["Percentual (%)"] = agg_destino["Percentual (%)"].apply(lambda x: formatar_numero_br(x, 2))
        st.dataframe(agg_destino.rename(columns={COL_DESTINO: "Tipo de Unidade (SINISA)"})[["Tipo de Unidade (SINISA)", "Massa (t)", "Percentual (%)"]], use_container_width=True)
        st.markdown("#### 📊 Principais destinos (gráfico)")
        top_destinos = agg_destino.head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_destinos[COL_DESTINO], top_destinos["MASSA_FLOAT"], color='steelblue')
        ax.set_xlabel('Massa (t)'); ax.set_title('Top 10 destinos de resíduos')
        ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.subheader(f"🏳️ Coleta de RSU pelos estados do Brasil ({ano_selecionado})")
        ocultar_transbordo_est = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_est")
        df_estados = df_mun_dest.copy()
        if ocultar_transbordo_est:
            df_estados = df_estados[~df_estados[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
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
            st.dataframe(agg_estados.rename(columns={"UF": "Estado"})[["Estado", "Massa (t)", "%", "% acumulado"]], use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 8))
            top_estados = agg_estados.head(10)
            ax.barh(top_estados["UF"], top_estados["MASSA_FLOAT"], color='forestgreen')
            ax.set_xlabel('Massa (t)'); ax.set_title('Top 10 estados')
            ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    if municipio == municipios[0]:
        st.markdown("---")
        st.header(f"🏆 Mapeamento de Coleta Seletiva de Orgânicos ({ano_selecionado})")
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
                pct_comp = (massa_compostagem / total_massa_org) * 100 if total_massa_org > 0 else 0
                pct_aterro = (massa_aterro / total_massa_org) * 100 if total_massa_org > 0 else 0
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Municípios com coleta seletiva", num_municipios)
                col_m2.metric("Massa p/ Compostagem", f"{formatar_br(pct_comp, auto_precision=False, casas_override=1)}%")
                col_m3.metric("Massa p/ Aterro", f"{formatar_br(pct_aterro, auto_precision=False, casas_override=1)}%")
                ranking_data = df_org_ranking.groupby([COL_MUNICIPIO, "UF", COL_DESTINO])["MASSA_FLOAT_RANK"].sum().reset_index()
                mapeamento = []
                preco = st.session_state.preco_carbono; cambio = st.session_state.taxa_cambio
                for (mun, uf), grupo in ranking_data.groupby([COL_MUNICIPIO, "UF"]):
                    massa_total_local = grupo["MASSA_FLOAT_RANK"].sum()
                    destinos = ", ".join(sorted(grupo[COL_DESTINO].unique()))
                    grupo["MCF"] = grupo[COL_DESTINO].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
                    grupo_aterro = grupo[grupo["MCF"] > 0]
                    massa_aterro_local = grupo_aterro["MASSA_FLOAT_RANK"].sum()
                    mcf_medio = ((grupo_aterro["MASSA_FLOAT_RANK"] * grupo_aterro["MCF"]).sum() / massa_aterro_local) if massa_aterro_local > 0 else 0.8
                    receita_anual = 0.0
                    if massa_aterro_local > 0:
                        df_mun_caract = df_clean[df_clean[COL_MUNICIPIO] == mun]
                        doc_pond, docf_pond, k_pond = calcular_doc_k_ponderado(df_mun_caract)
                        co2eq_aterro = calcular_co2eq_aterro_20anos(massa_aterro_local, mcf_medio, k_pond, doc_pond, docf_pond)
                        co2eq_compostagem = calcular_co2eq_compostagem_UNFCCC(massa_aterro_local)
                        receita_anual = ((co2eq_aterro - co2eq_compostagem) / ANOS_PROJECAO) * preco * cambio
                    massa_total_municipio = df_clean[df_clean[COL_MUNICIPIO] == mun]['MASSA_COLETADA'].sum()
                    pct_org = (massa_total_local / massa_total_municipio) * 100 if massa_total_municipio > 0 else 0
                    mapeamento.append({"Município": mun, "UF": uf, "Massa Total (t/ano)": massa_total_local,
                                        "Massa para Aterro (t/ano)": massa_aterro_local, "% da massa total": pct_org,
                                        "Tipo(s) de Unidade (SINISA)": destinos, "Receita Potencial (R$/ano)": receita_anual})
                df_mapeamento = pd.DataFrame(mapeamento).sort_values("Massa Total (t/ano)", ascending=False)
                st.dataframe(df_mapeamento.style.format({
                    "Massa Total (t/ano)": lambda x: formatar_numero_br(x, None),
                    "Massa para Aterro (t/ano)": lambda x: formatar_numero_br(x, None),
                    "% da massa total": lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%',
                    "Receita Potencial (R$/ano)": lambda x: f"R$ {formatar_numero_br(x, None)}"
                }), use_container_width=True, height=600)
                st.caption("Baseline (aterro): UNFCCC A6.4-AMT-003. Compostagem: TOOL13 / AMS-III.F.")

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
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
            df_mun_org = df_mun_org[~df_mun_org[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
        total_organicos = df_organicos["MASSA_FLOAT"].sum()
        massa_total_geral_org = df_mun_org["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de orgânicos coletados seletivamente: **{formatar_br(total_organicos, auto_precision=False, casas_override=2)} t**")
        agg_org = df_organicos.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index().sort_values("MASSA_FLOAT", ascending=False)
        fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
        cores_pie = plt.cm.Set3(np.linspace(0, 1, len(agg_org)))
        wedges, texts, autotexts = ax_pie.pie(agg_org["MASSA_FLOAT"], labels=None,
                                               autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
                                               startangle=90, colors=cores_pie, textprops={'fontsize': 9}, pctdistance=0.7)
        ax_pie.legend(wedges, agg_org[COL_DESTINO], title="Destino", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        ax_pie.axis('equal'); plt.tight_layout(); st.pyplot(fig_pie); plt.close(fig_pie)
        st.dataframe(agg_org, use_container_width=True)
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
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
            df_mun_podas = df_mun_podas[~df_mun_podas[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)]
        total_podas = df_podas["MASSA_FLOAT"].sum()
        massa_total_geral_podas = df_mun_podas["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de podas e galhadas coletadas: **{formatar_br(total_podas, auto_precision=False, casas_override=2)} t**")
        st.dataframe(df_podas.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index(), use_container_width=True)
    else:
        st.info("ℹ️ Sem registros de coleta de podas e galhadas.")

    st.markdown("---")
    st.caption(f"Fonte: SINISA (ano {ano_selecionado}) | Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F) | IPCC AR5 (GWP-100)")


#B9 — ABA DE IA
with tab_ia:
    st.header("🧠 Insights com Inteligência Artificial")

    # =========================================================
    # OPÇÃO OCULTAR TRANSBORDOS (NESTA ABA)
    # =========================================================
    ocultar_transbordo_ia = st.checkbox(
        "Ocultar transbordos (nesta aba)",
        value=False,
        key="ocultar_transbordo_ia",
        help="Exclui rotas cujo destino é 'Transbordo' em todas as análises desta aba."
    )
    if ocultar_transbordo_ia:
        df_clean_ia = df_clean[~df_clean[COL_DESTINO].apply(
            lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)].copy()
    else:
        df_clean_ia = df_clean.copy()

    st.markdown("""
    Aqui você pode explorar análises avançadas:
    - **Classificação de destinos** com PLN
    - **Projeção de geração de resíduos per capita**
    - **Simulação de cenários de compostagem**
    - **Clusterização de municípios** (K-Means)
    - **Análise de cobertura** da coleta seletiva
    """)

    st.subheader("📋 Classificação Inteligente de Destinos (PLN)")
    amostras = df_clean_ia[COL_DESTINO].dropna().sample(min(20, len(df_clean_ia))).tolist()
    dados_comparacao = []
    for texto in amostras:
        classe_regra = classificar_destino_regra(texto)
        classe_ia = classificador_ia.prever(texto, threshold=0.3)
        if classificador_ia.pipeline is not None:
            probs = classificador_ia.pipeline.predict_proba([normalizar_texto(texto)])[0]
            confianca = max(probs) * 100
        else:
            confianca = 0.0
        dados_comparacao.append({
            "Texto Original": texto[:50] + "..." if len(texto) > 50 else texto,
            "Regra (Manual)": classe_regra, "IA (Predição)": classe_ia,
            "Confiança da IA": f"{confianca:.1f}%",
            "Correção?": "✅" if classe_regra != classe_ia else "➖"
        })
    st.dataframe(pd.DataFrame(dados_comparacao), use_container_width=True, height=400)

    st.subheader("📊 Distribuição Nacional de Destinos (Classificação por IA)")
    @st.cache_data
    def classificar_todos_destinos(df, col_destino):
        return df[col_destino].apply(lambda x: classificador_ia.prever(x, threshold=0.3))
    with st.spinner("🤖 Classificando todos os destinos com IA..."):
        df_clean_ia['destino_ia'] = classificar_todos_destinos(df_clean_ia, COL_DESTINO)
    contagem_ia = df_clean_ia['destino_ia'].value_counts().reset_index()
    contagem_ia.columns = ['Destino (IA)', 'Quantidade']
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cores = plt.cm.Set3(np.linspace(0, 1, len(contagem_ia)))
    wedges, texts, autotexts = ax1.pie(contagem_ia['Quantidade'], labels=None,
                                        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
                                        startangle=90, colors=cores, textprops={'fontsize': 9}, pctdistance=0.7)
    ax1.legend(wedges, contagem_ia['Destino (IA)'], title="Destino", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax1.axis('equal'); plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    st.markdown("---")
    st.subheader("📈 Clusterização de Municípios por Perfil de Resíduos")
    if st.button("🔍 Executar Clusterização"):
        with st.spinner("Agrupando municípios..."):
            try:
                from utils.ia_clustering import (preparar_dados_clusterizacao, clusterizar_municipios,
                    aplicar_pca, plot_clusters, resumo_clusters, descrever_clusters)
                X, df_cluster = preparar_dados_clusterizacao(df_clean_ia)
                if X.empty:
                    st.warning("Dados insuficientes para clusterização.")
                else:
                    n_clusters = st.slider("Número de clusters:", 2, 6, 4)
                    labels, kmeans, scaler = clusterizar_municipios(X, n_clusters=n_clusters)
                    df_cluster['Cluster'] = labels
                    X_pca, pca = aplicar_pca(X)
                    st.pyplot(plot_clusters(X_pca, labels, df_cluster))
                    st.dataframe(resumo_clusters(df_cluster, labels))
            except Exception as e:
                st.error(f"Erro na clusterização: {e}")

    st.markdown("---")
    st.subheader("📈 Previsão de Geração de Resíduos por Habitante")
    opcoes_proj = ["BRASIL – Todos os municípios"] + sorted(df_clean_ia[COL_MUNICIPIO].unique())
    municipio_proj = st.selectbox("Selecione o município (ou Brasil) para projeção:", opcoes_proj, key="proj_municipio")
    if municipio_proj:
        if municipio_proj == "BRASIL – Todos os municípios":
            df_mun_proj = df_clean_ia.copy()
            massa_atual = df_mun_proj['MASSA_COLETADA'].sum()
            pop_calculada = (df_mun_proj.drop_duplicates(subset=[COL_MUNICIPIO])
                             .loc[lambda d: d['POPULACAO_TOTAL'] > 0, 'POPULACAO_TOTAL'].sum())
            if pop_calculada <= 0:
                pop_calculada = 210000000
            pop_atual = st.number_input("População total do Brasil – ajuste opcional:", min_value=1000,
                                        value=int(pop_calculada), step=1000000, key="pop_brasil_proj")
            titulo_proj = "Brasil"
        else:
            df_mun_proj = df_clean_ia[df_clean_ia[COL_MUNICIPIO] == municipio_proj]
            massa_atual = df_mun_proj['MASSA_COLETADA'].sum()
            pop_serie = df_mun_proj.drop_duplicates(subset=[COL_MUNICIPIO])['POPULACAO_TOTAL']
            pop_calculada = float(pop_serie.iloc[0]) if (not pop_serie.empty and pop_serie.iloc[0] > 0) else 50000
            pop_atual = st.number_input(f"População atual – {municipio_proj}:", min_value=100,
                                        value=int(pop_calculada), step=1000, key=f"pop_{municipio_proj}")
            titulo_proj = municipio_proj
        if massa_atual > 0:
            col1, col2 = st.columns(2)
            with col1:
                taxa_pop = st.slider("Taxa de crescimento populacional anual (%)", 0.0, 5.0, 1.0, 0.1) / 100
            with col2:
                anos_proj = st.slider("Anos de projeção", 5, 30, 10)
            if st.button("📊 Projetar Resíduos per Capita"):
                try:
                    df_proj = projetar_residuos_per_capita(pop_atual, massa_atual, taxa_pop, anos_proj)
                    st.pyplot(plot_projecao_residuos(df_proj))
                    st.dataframe(df_proj)
                except Exception as e:
                    st.error(f"Erro na projeção: {e}")

    st.markdown("---")
    st.subheader("💰 Simulador: Créditos de Carbono com Compostagem")
    opcoes_sim = ["BRASIL – Todos os municípios"] + sorted(df_clean_ia[COL_MUNICIPIO].unique())
    municipio_sim = st.selectbox("Selecione o município (ou Brasil) para a simulação:", opcoes_sim, key="sim_municipio")
    if municipio_sim:
        df_mun_sim = df_clean_ia.copy() if municipio_sim == "BRASIL – Todos os municípios" else df_clean_ia[df_clean_ia[COL_MUNICIPIO] == municipio_sim]
        df_mun_sim['MCF'] = df_mun_sim[COL_DESTINO].apply(determinar_mcf_por_destino)
        df_org_aterro = df_mun_sim[df_mun_sim['MCF'] > 0]
        massa_aterro_atual = df_org_aterro['MASSA_COLETADA'].sum()
        if massa_aterro_atual > 0:
            col1, col2 = st.columns(2)
            with col1:
                taxa_crescimento = st.slider("Taxa anual de aumento da compostagem (%)", 5, 30, 15, 1) / 100
                anos_sim = st.slider("Anos de projeção", 5, 20, 10)
            with col2:
                inflacao_carbono = st.slider("Inflação anual do preço do carbono (%)", 0, 5, 2, 1) / 100
            if st.button("🚀 Executar Simulação"):
                try:
                    df_evitado_mun_sim = calcular_evitado_por_municipio(df_mun_sim, COL_DESTINO, COL_MASSA)
                    total_massa_org_sim = df_evitado_mun_sim['Massa_Org_Seletiva'].sum()
                    total_evitado_sim = df_evitado_mun_sim['Evitado_Total'].sum()
                    co2_evitado_por_t = total_evitado_sim / total_massa_org_sim if total_massa_org_sim > 0 else 0.5
                    if co2_evitado_por_t > 0:
                        df_sim = simular_cenarios_compostagem(massa_aterro_atual, co2_evitado_por_t,
                            st.session_state.preco_carbono, st.session_state.taxa_cambio,
                            anos_projecao=anos_sim, taxa_crescimento_compostagem=taxa_crescimento,
                            inflacao_carbono=inflacao_carbono)
                        st.pyplot(plot_simulacao_compostagem(df_sim))
                        st.dataframe(df_sim)
                except Exception as e:
                    st.error(f"Erro na simulação: {e}")

    st.markdown("---")
    st.subheader("🌍 Cenários de Expansão da Compostagem no Brasil")
    st.info("Análise de expansão baseada nos municípios com e sem coleta seletiva de orgânicos.")
    mask_organicos = df_clean_ia[COL_TIPO_COLETA].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)
    df_org = df_clean_ia[mask_organicos].copy()
    if not df_org.empty:
        with st.spinner("Calculando evitados reais por município..."):
            df_evitado_mun = calcular_evitado_por_municipio(df_clean_ia, COL_DESTINO, COL_MASSA)
            total_massa_org_real = df_evitado_mun['Massa_Org_Seletiva'].sum()
            total_evitado_real = df_evitado_mun['Evitado_Total'].sum()
            evitado_medio_por_t_real = total_evitado_real / total_massa_org_real if total_massa_org_real > 0 else 0
        df_org['MCF'] = df_org[COL_DESTINO].apply(determinar_mcf_por_destino)
        total_aterro = df_org[df_org['MCF'] > 0]['MASSA_COLETADA'].sum()
        total_compost = df_org[df_org['MCF'] == 0]['MASSA_COLETADA'].sum()
        total_massa_org = total_aterro + total_compost
        pct_compost_real = (total_compost / total_massa_org) * 100 if total_massa_org > 0 else 0
        st.metric("Percentual destinado à compostagem", f"{formatar_br(pct_compost_real, auto_precision=False, casas_override=2)}%")
        st.info("Cenário atual consolidado calculado a partir dos dados do SINISA.")


#B10 — ABA DE DIAGNÓSTICO (RESTAURADA INTEGRALMENTE + OCULTAR TRANSBORDOS)
with tab_diagnostico:
    st.header("🔥 Diagnóstico de Emissões de Metano (Baseline)")

    # =========================================================
    # OPÇÃO OCULTAR TRANSBORDOS (NESTA ABA)
    # =========================================================
    ocultar_transbordo_diag = st.checkbox(
        "Ocultar transbordos (nesta aba)",
        value=False,
        key="ocultar_transbordo_diag",
        help="Exclui rotas cujo destino é 'Transbordo' em todas as análises desta aba."
    )
    if ocultar_transbordo_diag:
        df_clean_diag = df_clean[~df_clean[COL_DESTINO].apply(
            lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False)].copy()
    else:
        df_clean_diag = df_clean.copy()

    st.markdown("""
    Esta análise revela **quanto cada município emite com base nos dados mais recentes do SINISA**,
    considerando **três fatores determinantes**:
    
    1. **Quantidade de resíduos** enviada a aterros (massa real declarada);
    2. **Mix de resíduos** (composição orgânica, representada pelo DOC e taxa de decaimento k);
    3. **Destino final e gestão** (MCF – diferencia aterros sanitários, controlados e lixões).
    
    O cálculo segue a **metodologia UNFCCC A6.4-AMT-003 (modelo anual, Equação 1)**, projetando a geração de metano ao longo de **20 anos** a partir da massa de resíduos depositada no ano de referência.
    """)

    @st.cache_data
    def calcular_emissoes_brutas_por_municipio(df):
        resultados = []
        municipios = df['MUNICÍPIO'].unique()
        with st.spinner(f"🔄 Calculando emissões para {len(municipios)} municípios..."):
            for mun in municipios:
                df_mun = df[df['MUNICÍPIO'] == mun].copy()
                doc_pond, docf_pond, k_pond = calcular_doc_k_ponderado(df_mun)
                df_mun['MCF'] = df_mun[COL_DESTINO].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
                df_aterro = df_mun[df_mun['MCF'] > 0].copy()
                if df_aterro.empty:
                    continue
                df_aterro['MASSA_FLOAT'] = pd.to_numeric(df_aterro['MASSA_COLETADA'], errors='coerce').fillna(0)
                df_aterro = df_aterro[df_aterro['MASSA_FLOAT'] > 0]
                if df_aterro.empty:
                    continue
                massa_total_aterro = df_aterro['MASSA_FLOAT'].sum()
                mcf_medio = (df_aterro['MASSA_FLOAT'] * df_aterro['MCF']).sum() / massa_total_aterro
                co2eq_20anos = calcular_co2eq_aterro_20anos(massa_total_aterro, mcf_medio, k_pond, doc_pond, docf_pond)
                emissao_anual = co2eq_20anos / 20.0
                pop = pd.to_numeric(df_mun['POPULACAO_TOTAL'].iloc[0], errors='coerce') if 'POPULACAO_TOTAL' in df_mun.columns else 0
                if pd.isna(pop) or pop <= 0:
                    pop = 0
                intensidade = emissao_anual / massa_total_aterro if massa_total_aterro > 0 else 0
                uf = df_mun['UF'].iloc[0] if 'UF' in df_mun.columns else 'N/A'
                if mcf_medio >= 0.8:
                    gestao_cat = "Sanitário"
                elif mcf_medio >= 0.4:
                    gestao_cat = "Controlado"
                else:
                    gestao_cat = "Lixão/Precário"
                resultados.append({
                    'MUNICÍPIO': mun, 'UF': uf,
                    'Massa_Aterro_Anual_t': massa_total_aterro,
                    'MCF_Medio': mcf_medio, 'DOC_Medio': doc_pond, 'DOCF_Medio': docf_pond, 'k_Medio': k_pond,
                    'Emissao_Bruta_tCO2e_ano': emissao_anual,
                    'Intensidade_tCO2e_por_t': intensidade,
                    'Emissao_per_capita_kgCO2e': (emissao_anual * 1000) / pop if pop > 0 else 0,
                    'Gestao_Predominante': gestao_cat
                })
        return pd.DataFrame(resultados)

    with st.spinner("⏳ Processando dados de todos os municípios..."):
        df_emissoes = calcular_emissoes_brutas_por_municipio(df_clean_diag)

    total_municipios_snis = df_clean_diag['MUNICÍPIO'].nunique()
    total_municipios_aterro = len(df_emissoes)
    total_com_emissao_zero = total_municipios_snis - total_municipios_aterro

    st.markdown("---")
    st.subheader("📊 Estatísticas Gerais da Base de Dados")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏙️ Total de municípios no SINISA", total_municipios_snis)
    col2.metric("🗑️ Municípios que enviaram resíduos para aterro", total_municipios_aterro)
    col3.metric("📭 Municípios sem envio para aterro (ou dados zerados)", total_com_emissao_zero)

    if df_emissoes.empty:
        st.warning("Nenhum município com resíduos enviados para aterro foi encontrado.")
    else:
        # =========================================================
        # SEÇÃO 1: LIMIARES DO SBCE (10k e 25k) - CENÁRIO ATUAL
        # =========================================================
        st.markdown("---")
        st.subheader("⚖️ Municípios acima dos Limiares do SBCE (10.000 e 25.000 tCO₂e/ano)")
        st.markdown("""
        A Lei do SBCE (15.042/2024):
        - **> 10.000 tCO₂e/ano**: Obrigação de MRV.
        - **> 25.000 tCO₂e/ano**: Obrigação plena (MRV + CBEs).
        """)
        df_limiares = df_emissoes.copy()
        df_limiares['Emissao_Bruta_tCO2e_ano'] = pd.to_numeric(df_limiares['Emissao_Bruta_tCO2e_ano'], errors='coerce').fillna(0)
        df_acima_10k = df_limiares[df_limiares['Emissao_Bruta_tCO2e_ano'] > 10000].copy()
        df_acima_25k = df_limiares[df_limiares['Emissao_Bruta_tCO2e_ano'] > 25000].copy()
        col1, col2 = st.columns(2)
        col1.metric("🔹 Acima de 10.000 tCO₂e (MRV)", f"{len(df_acima_10k)} municípios")
        col2.metric("🔺 Acima de 25.000 tCO₂e (Obrigação Plena)", f"{len(df_acima_25k)} municípios")
        st.markdown("#### 📋 Todos os municípios com emissão > 10.000 tCO₂e/ano")
        if not df_acima_10k.empty:
            df_exibicao_10k = df_acima_10k[['MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Emissao_Bruta_tCO2e_ano', 'Massa_Aterro_Anual_t']]
            df_exibicao_10k = df_exibicao_10k.sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
            st.dataframe(df_exibicao_10k.style.format({
                'Emissao_Bruta_tCO2e_ano': lambda x: f"{x:,.0f}".replace(",", "."),
                'Massa_Aterro_Anual_t': lambda x: f"{x:,.0f}".replace(",", ".")
            }), use_container_width=True, height=400)
        else:
            st.info("ℹ️ Nenhum município ultrapassa 10.000 tCO₂e/ano.")
        st.markdown("#### 🔺 Destaque: municípios acima de 25.000 tCO₂e/ano")
        if not df_acima_25k.empty:
            df_exibicao_25k = df_acima_25k[['MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Emissao_Bruta_tCO2e_ano', 'Massa_Aterro_Anual_t']].sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
            st.dataframe(df_exibicao_25k.style.format({
                'Emissao_Bruta_tCO2e_ano': lambda x: f"{x:,.0f}".replace(",", "."),
                'Massa_Aterro_Anual_t': lambda x: f"{x:,.0f}".replace(",", ".")
            }), use_container_width=True, height=300)
        else:
            st.info("ℹ️ Nenhum município ultrapassa 25.000 tCO₂e/ano.")

        st.markdown("---")
        estados = sorted(df_emissoes['UF'].unique())
        estado_selecionado = st.selectbox("Filtrar por Estado:", ["Todos"] + estados)
        df_filtrado = df_emissoes if estado_selecionado == "Todos" else df_emissoes[df_emissoes['UF'] == estado_selecionado]

        total_emissoes = df_filtrado['Emissao_Bruta_tCO2e_ano'].sum()
        total_massa = df_filtrado['Massa_Aterro_Anual_t'].sum()
        media_intensidade = df_filtrado['Intensidade_tCO2e_por_t'].mean()
        num_lixoes = df_filtrado[df_filtrado['Gestao_Predominante'] == 'Lixão/Precário'].shape[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌍 Emissão Média Anual (20 anos)", f"{formatar_br(total_emissoes, auto_precision=False, casas_override=0)} tCO₂e")
        col2.metric("⚖️ Massa em Aterro", f"{formatar_br(total_massa, auto_precision=False, casas_override=0)} t")
        col3.metric("📊 Intensidade Média", f"{formatar_br(media_intensidade, auto_precision=False, casas_override=2)} tCO₂e/t")
        col4.metric("⚠️ Municípios com Lixão", num_lixoes)

        st.markdown("---")
        st.markdown("#### 📉 Curva de Concentração das Emissões de Metano (Pareto)")
        df_emissoes_ordenado = df_filtrado.sort_values('Emissao_Bruta_tCO2e_ano', ascending=False).copy()
        df_emissoes_ordenado['emissao_acumulada'] = df_emissoes_ordenado['Emissao_Bruta_tCO2e_ano'].cumsum()
        total_emissoes = df_emissoes_ordenado['Emissao_Bruta_tCO2e_ano'].sum()
        df_emissoes_ordenado['pct_acumulado_emissao'] = (df_emissoes_ordenado['emissao_acumulada'] / total_emissoes) * 100
        df_ate_80_emissoes = df_emissoes_ordenado[df_emissoes_ordenado['pct_acumulado_emissao'] <= 80]
        pct_municipios_80_emissoes = (len(df_ate_80_emissoes) / len(df_emissoes_ordenado)) * 100
        df_ate_50_emissoes = df_emissoes_ordenado[df_emissoes_ordenado['pct_acumulado_emissao'] <= 50]
        pct_municipios_50_emissoes = (len(df_ate_50_emissoes) / len(df_emissoes_ordenado)) * 100
        fig_emissoes, ax_emissoes = plt.subplots(figsize=(12, 7))
        df_emissoes_ordenado['pct_municipios_emissoes'] = (np.arange(len(df_emissoes_ordenado)) + 1) / len(df_emissoes_ordenado) * 100
        ax_emissoes.plot(df_emissoes_ordenado['pct_municipios_emissoes'], df_emissoes_ordenado['pct_acumulado_emissao'], color='#1f77b4', linewidth=3, label='Concentração real das emissões')
        ax_emissoes.axhline(y=80, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='80% das emissões totais')
        ax_emissoes.axvline(x=pct_municipios_80_emissoes, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        ax_emissoes.annotate(f'{pct_municipios_80_emissoes:.1f}% dos municípios\nconcentram 80% das emissões',
                              xy=(pct_municipios_80_emissoes, 80), xytext=(pct_municipios_80_emissoes + 15, 60),
                              arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=11, color='red', ha='left',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
        ax_emissoes.plot([0, 100], [0, 100], color='gray', linestyle=':', alpha=0.5, label='Igualdade perfeita')
        ax_emissoes.set_xlabel('Percentual acumulado de municípios (%)')
        ax_emissoes.set_ylabel('Percentual acumulado das emissões (%)')
        ax_emissoes.set_title(f'Concentração das Emissões de Metano – Brasil ({ano_selecionado})', fontsize=14)
        ax_emissoes.grid(True, linestyle=':', alpha=0.4)
        ax_emissoes.legend(loc='lower right')
        ax_emissoes.set_xlim(0, 100); ax_emissoes.set_ylim(0, 100)
        ax_emissoes.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        ax_emissoes.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        plt.tight_layout(); st.pyplot(fig_emissoes); plt.close(fig_emissoes)

        st.markdown("---")
        st.subheader("🏆 Top 20 Municípios que mais Emitem Metano (emissão absoluta)")
        top20 = df_filtrado.nlargest(20, 'Emissao_Bruta_tCO2e_ano').sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
        cor_map = {'Sanitário': '#2ecc71', 'Controlado': '#f39c12', 'Lixão/Precário': '#e74c3c'}
        top20['Cor'] = top20['Gestao_Predominante'].map(cor_map)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(top20['MUNICÍPIO'] + " (" + top20['UF'] + ")", top20['Emissao_Bruta_tCO2e_ano'], color=top20['Cor'])
        ax.set_xlabel('Emissão Média Anual (tCO₂e / ano)')
        ax.set_title('Ranking de Emissões de Metano por Município')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatar_br(x, auto_precision=False, casas_override=2)))
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='Aterro Sanitário'),
                            Patch(facecolor='#f39c12', label='Aterro Controlado'),
                            Patch(facecolor='#e74c3c', label='Lixão/Precário')]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.invert_yaxis(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.subheader("🏆 Top 20 Municípios com Maior Emissão de Metano por Habitante")
        df_percapita = df_filtrado[df_filtrado['Emissao_per_capita_kgCO2e'] > 0].copy()
        if not df_percapita.empty:
            top20_percapita = df_percapita.nlargest(20, 'Emissao_per_capita_kgCO2e').sort_values('Emissao_per_capita_kgCO2e', ascending=False)
            top20_percapita['Cor'] = top20_percapita['Gestao_Predominante'].map(cor_map)
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            ax2.barh(top20_percapita['MUNICÍPIO'] + " (" + top20_percapita['UF'] + ")", top20_percapita['Emissao_per_capita_kgCO2e'], color=top20_percapita['Cor'])
            ax2.set_xlabel('Emissão per capita (kgCO₂e / hab / ano)')
            ax2.set_title('Ranking de Emissões de Metano por Habitante')
            ax2.legend(handles=legend_elements, loc='lower right')
            ax2.invert_yaxis(); plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        st.markdown("---")
        st.subheader("📊 Matriz de Decisão: Massa x Intensidade")
        med_massa = df_filtrado['Massa_Aterro_Anual_t'].median()
        med_intensidade = df_filtrado['Intensidade_tCO2e_por_t'].median()
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        def categorizar(row):
            if row['Massa_Aterro_Anual_t'] >= med_massa and row['Intensidade_tCO2e_por_t'] >= med_intensidade:
                return 'Crítico'
            elif row['Massa_Aterro_Anual_t'] < med_massa and row['Intensidade_tCO2e_por_t'] >= med_intensidade:
                return 'Ineficiente'
            elif row['Massa_Aterro_Anual_t'] >= med_massa and row['Intensidade_tCO2e_por_t'] < med_intensidade:
                return 'Referência'
            else:
                return 'Baixa Prioridade'
        df_filtrado = df_filtrado.copy()
        df_filtrado['Categoria'] = df_filtrado.apply(categorizar, axis=1)
        cores_cat = {'Crítico': '#e74c3c', 'Ineficiente': '#f39c12', 'Referência': '#2ecc71', 'Baixa Prioridade': '#3498db'}
        for cat in df_filtrado['Categoria'].unique():
            subset = df_filtrado[df_filtrado['Categoria'] == cat]
            ax3.scatter(subset['Massa_Aterro_Anual_t'], subset['Intensidade_tCO2e_por_t'], label=cat, color=cores_cat[cat], alpha=0.7, s=50)
        ax3.axvline(x=med_massa, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=med_intensidade, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Massa enviada ao Aterro (t/ano)')
        ax3.set_ylabel('Intensidade de Emissão (tCO₂e / t)')
        ax3.set_title('Matriz de Priorização de Municípios')
        ax3.legend(); ax3.grid(True, linestyle=':', alpha=0.3)
        ax3.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        st.markdown("---")
        st.subheader("📋 Detalhamento por Município")
        tabela_diagnostico = df_filtrado.copy()
        tabela_diagnostico['Emissao_Bruta_tCO2e_ano'] = tabela_diagnostico['Emissao_Bruta_tCO2e_ano'].apply(lambda x: formatar_numero_br(x, 0))
        tabela_diagnostico['Massa_Aterro_Anual_t'] = tabela_diagnostico['Massa_Aterro_Anual_t'].apply(lambda x: formatar_numero_br(x, 0))
        tabela_diagnostico['Intensidade_tCO2e_por_t'] = tabela_diagnostico['Intensidade_tCO2e_por_t'].apply(lambda x: formatar_numero_br(x, 2))
        tabela_diagnostico['Emissao_per_capita_kgCO2e'] = tabela_diagnostico['Emissao_per_capita_kgCO2e'].apply(lambda x: formatar_numero_br(x, 2))
        tabela_diagnostico['MCF_Medio'] = tabela_diagnostico['MCF_Medio'].apply(lambda x: formatar_numero_br(x, 2))
        tabela_diagnostico['DOC_Medio'] = tabela_diagnostico['DOC_Medio'].apply(lambda x: formatar_numero_br(x, 3))
        tabela_diagnostico = tabela_diagnostico[[
            'MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Massa_Aterro_Anual_t',
            'MCF_Medio', 'DOC_Medio', 'Intensidade_tCO2e_por_t',
            'Emissao_Bruta_tCO2e_ano', 'Emissao_per_capita_kgCO2e'
        ]]
        tabela_diagnostico = tabela_diagnostico.rename(columns={
            'MUNICÍPIO': 'Município', 'UF': 'UF', 'Gestao_Predominante': 'Gestão',
            'Massa_Aterro_Anual_t': 'Massa (t/ano)', 'MCF_Medio': 'MCF médio',
            'DOC_Medio': 'DOC médio', 'Intensidade_tCO2e_por_t': 'Intensidade (tCO₂e/t)',
            'Emissao_Bruta_tCO2e_ano': 'Emissão Média Anual (tCO₂e/ano)',
            'Emissao_per_capita_kgCO2e': 'Emissão per capita (kgCO₂e)'
        })
        st.dataframe(tabela_diagnostico, use_container_width=True, height=500)

        # =========================================================
        # SEÇÃO 2: PROJEÇÃO CONTÍNUA DE 20 ANOS
        # =========================================================
        st.markdown("---")
        st.subheader("📈 Projeção Contínua de Emissões (20 anos com depósitos anuais)")
        st.markdown("""
        **Cenário:** E se o município continuar depositando a mesma quantidade de resíduos no aterro todos os anos?
        """)
        opcoes_proj_cont = ["BRASIL – Todos os municípios"] + sorted(df_emissoes['MUNICÍPIO'].unique())
        municipio_proj_cont = st.selectbox("Selecione o município (ou Brasil) para a projeção contínua:",
                                            opcoes_proj_cont, key="proj_cont_municipio")
        if municipio_proj_cont:
            if municipio_proj_cont == "BRASIL – Todos os municípios":
                df_mun_proj_cont = df_emissoes.copy()
                massa_anual = df_mun_proj_cont['Massa_Aterro_Anual_t'].sum()
                mcf_medio = (df_mun_proj_cont['Massa_Aterro_Anual_t'] * df_mun_proj_cont['MCF_Medio']).sum() / massa_anual if massa_anual > 0 else 0.8
                doc_medio = (df_mun_proj_cont['Massa_Aterro_Anual_t'] * df_mun_proj_cont['DOC_Medio']).sum() / massa_anual if massa_anual > 0 else 0.15
                docf_medio = (df_mun_proj_cont['Massa_Aterro_Anual_t'] * df_mun_proj_cont['DOCF_Medio']).sum() / massa_anual if massa_anual > 0 else 0.5
                k_medio = (df_mun_proj_cont['Massa_Aterro_Anual_t'] * df_mun_proj_cont['k_Medio']).sum() / massa_anual if massa_anual > 0 else 0.07
                titulo_proj_cont = "Brasil"
            else:
                df_mun_proj_cont = df_emissoes[df_emissoes['MUNICÍPIO'] == municipio_proj_cont]
                massa_anual = df_mun_proj_cont['Massa_Aterro_Anual_t'].sum()
                mcf_medio = df_mun_proj_cont['MCF_Medio'].iloc[0]
                doc_medio = df_mun_proj_cont['DOC_Medio'].iloc[0]
                docf_medio = df_mun_proj_cont['DOCF_Medio'].iloc[0]
                k_medio = df_mun_proj_cont['k_Medio'].iloc[0]
                titulo_proj_cont = municipio_proj_cont
            st.info(f"📌 **Massa anual:** {formatar_br(massa_anual, auto_precision=False, casas_override=0)} t — MCF médio: {mcf_medio:.2f}, k médio: {k_medio:.4f}")
            if massa_anual > 0 and mcf_medio > 0:
                df_proj_cont = projetar_emissao_continua(massa_anual, mcf_medio, k_medio, doc_medio, docf_medio)
                fig_cont, ax_cont = plt.subplots(figsize=(12, 6))
                ax_cont.plot(df_proj_cont['Ano'], df_proj_cont['Emissao_Acumulada'], 'o-', color='blue', linewidth=2, label='Emissão Acumulada (tCO₂e)')
                ax_cont.plot(df_proj_cont['Ano'], df_proj_cont['Emissao_Anual'], 's-', color='orange', linewidth=2, label='Emissão Anual (tCO₂e)')
                ax_cont.set_xlabel('Ano'); ax_cont.set_ylabel('Emissões (tCO₂e)')
                ax_cont.set_title(f'Projeção Contínua de 20 anos – {titulo_proj_cont}')
                ax_cont.legend(); ax_cont.grid(True, linestyle='--', alpha=0.3)
                ax_cont.yaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
                plt.tight_layout(); st.pyplot(fig_cont); plt.close(fig_cont)
                with st.expander("📋 Ver dados anuais da projeção contínua"):
                    st.dataframe(df_proj_cont)

        # =========================================================
        # SEÇÃO 3: LIMIARES DO SBCE – CENÁRIO CONTÍNUO
        # =========================================================
        st.markdown("---")
        st.subheader("⚖️ Municípios acima dos Limiares do SBCE – **Cenário Contínuo**")
        st.markdown("""
        Este cenário considera depósitos anuais repetidos, calculando a emissão no 20º ano de operação contínua.
        """)
        df_continua = df_emissoes.copy()
        def calcular_emissao_continua_ano_n(massa_anual, mcf, doc, docf, k, n=20):
            if massa_anual <= 0 or mcf <= 0:
                return 0.0
            ch4_pot_kg = (doc * docf * mcf * F_METHANE_FRACTION * (16/12) *
                          (1 - OX_SOIL_COVER) * PHI_APPLICATION_B)
            fator_tco2_por_ton = ch4_pot_kg * GWP_CH4
            return massa_anual * fator_tco2_por_ton * (1 - np.exp(-k * n))
        df_continua['Emissao_Continua_Ano20'] = df_continua.apply(
            lambda row: calcular_emissao_continua_ano_n(
                row['Massa_Aterro_Anual_t'], row['MCF_Medio'], row['DOC_Medio'],
                row['DOCF_Medio'], row['k_Medio'], n=20), axis=1)
        df_acima_10k_cont = df_continua[df_continua['Emissao_Continua_Ano20'] > 10000].copy()
        df_acima_25k_cont = df_continua[df_continua['Emissao_Continua_Ano20'] > 25000].copy()
        col1, col2 = st.columns(2)
        col1.metric("🔹 Acima de 10.000 tCO₂e (MRV) – Cenário Contínuo", f"{len(df_acima_10k_cont)} municípios")
        col2.metric("🔺 Acima de 25.000 tCO₂e (Obrigação Plena) – Cenário Contínuo", f"{len(df_acima_25k_cont)} municípios")
        st.markdown("#### 📋 Todos os municípios com emissão > 10.000 tCO₂e/ano (Cenário Contínuo – ano 20)")
        if not df_acima_10k_cont.empty:
            df_exibicao_10k_cont = df_acima_10k_cont[['MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Emissao_Continua_Ano20', 'Massa_Aterro_Anual_t']].sort_values('Emissao_Continua_Ano20', ascending=False)
            st.dataframe(df_exibicao_10k_cont.style.format({
                'Emissao_Continua_Ano20': lambda x: f"{x:,.0f}".replace(",", "."),
                'Massa_Aterro_Anual_t': lambda x: f"{x:,.0f}".replace(",", ".")
            }), use_container_width=True, height=400)
        else:
            st.info("ℹ️ Nenhum município ultrapassa 10.000 tCO₂e/ano no cenário contínuo.")

        st.markdown("#### 📊 Comparação entre cenários")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cenário atual (1 depósito)", f"{len(df_acima_10k)} municípios > 10k")
        with col2:
            st.metric("Cenário contínuo (ano 20)", f"{len(df_acima_10k_cont)} municípios > 10k")
        with col3:
            diferenca = len(df_acima_10k_cont) - len(df_acima_10k)
            st.metric("Diferença", f"+{diferenca}" if diferenca > 0 else f"{diferenca}")

        st.markdown("---")
        st.caption("""
        **Metodologia:** UNFCCC A6.4-AMT-003 (Application B) – Baseline de aterro.
        - **Emissão Média Anual**: média aritmética do total de emissões de CH₄ projetado para os 20 anos seguintes.
        - **MCF**: 1,0 (Sanitário), 0,4-0,8 (Controlado), <0,4 (Lixão/Precário).
        - **Cenário Contínuo**: considera depósitos anuais repetidos, calculando a emissão no 20º ano de operação.
        """)


#B11 — AUTORIA
st.markdown("---")
st.subheader("📬 Autoria e uso")
st.markdown("""
Este aplicativo foi desenvolvido para apoiar a gestão de resíduos sólidos, 
mapear oportunidades de compostagem e auxiliar municípios a se prepararem para o mercado de créditos de carbono.

**Potencial de uso:**  
- Mapeamento de municípios com coleta seletiva de orgânicos.  
- Estimativa de emissões evitadas com compostagem.  
- Projeção de receitas com créditos de carbono (metodologia UNFCCC).  
- Identificação de prioridades para expansão da coleta seletiva.
""")

st.markdown("---")
st.caption("""
**Composta.IA** | Ferramenta de apoio à gestão de resíduos sólidos e créditos de carbono  
Dados: SINISA (2023/2024) | Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F) | IPCC AR5 (GWP-100)
Fração orgânica de referência: 50% (Pimentel e Capanema, 2025).
""")
