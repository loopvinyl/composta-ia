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
Este aplicativo interpreta os **tipos de coleta executada** informados pelos municípios no SNIS
e avalia o **potencial técnico para compostagem** de resíduos sólidos urbanos,
utilizando **Inteligência Artificial** para padronizar os dados e a **metodologia UNFCCC A6.4-AMT-003** para o cálculo de emissões.

**Ferramenta de apoio à gestão pública** – desenvolvida para subsidiar o SINISA e políticas de resíduos sólidos.
""")

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
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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
# FUNÇÃO PARA CALCULAR DOC e k PONDERADOS (VIA SNIS)
# =========================================================
def calcular_doc_k_ponderado(df_municipio):
    colunas_caract = {
        'Alimentos_Verdes': 'GTR1501',
        'Vidros': 'GTR1502',
        'Metais': 'GTR1503',
        'Plasticos': 'GTR1504',
        'Papeis': 'GTR1505',
        'Têxteis': 'GTR1506',
        'Outros': 'GTR1507'
    }
    colunas_presentes = [col for col in colunas_caract.values() if col in df_municipio.columns]
    if not colunas_presentes:
        return DOC_PADRAO, K_PADRAO
    df_caract = df_municipio[colunas_presentes].copy()
    for col in df_caract.columns:
        df_caract[col] = pd.to_numeric(df_caract[col], errors='coerce').fillna(0)
    pct = {}
    for nome, col in colunas_caract.items():
        if col in df_caract.columns:
            val = df_caract[col].mean()
            pct[nome] = val if val > 0 else 0
        else:
            pct[nome] = 0
    if sum(pct.values()) == 0:
        return DOC_PADRAO, K_PADRAO
    doc_pond = (pct['Alimentos_Verdes'] * 0.7 +
                pct['Papeis'] * 0.5 +
                pct['Têxteis'] * 0.24 +
                pct['Outros'] * 0.1) / 100.0
    k_pond = (pct['Alimentos_Verdes'] * 0.17 +
              pct['Papeis'] * 0.07 +
              pct['Têxteis'] * 0.07 +
              pct['Outros'] * 0.035) / 100.0
    doc_pond = max(doc_pond, DOC_PADRAO) if doc_pond > 0 else DOC_PADRAO
    k_pond = max(k_pond, K_PADRAO) if k_pond > 0 else K_PADRAO
    return doc_pond, k_pond

# =========================================================
# FUNÇÕES DE CÁLCULO – ATERRO (BASELINE UNFCCC)
# =========================================================
def construir_lotes_diarios(massa_total_ano_kg, dias_entrada=365, dias_projecao=DIAS_PROJECAO):
    entrada = np.zeros(dias_projecao, dtype=float)
    if dias_entrada > 0:
        massa_diaria = massa_total_ano_kg / dias_entrada
        entrada[:dias_entrada] = massa_diaria
    return entrada

def calcular_emissoes_aterro_diario(massa_total_ano_kg, mcf, k_ano, temp_C, doc,
                                    phi=PHI_APPLICATION_B, ox=OX_SOIL_COVER,
                                    dias_projecao=DIAS_PROJECAO, dias_entrada=365):
    if massa_total_ano_kg <= 0 or mcf <= 0:
        return np.zeros(dias_projecao), np.zeros(dias_projecao)
    docf = 0.0147 * temp_C + 0.28
    ch4_pot_por_kg = (doc * docf * mcf * F_METHANE_FRACTION * (16/12) *
                      (1 - ox) * phi)
    entrada = construir_lotes_diarios(massa_total_ano_kg, dias_entrada, dias_projecao)
    t = np.arange(1, dias_projecao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    kernel_ch4 = np.maximum(kernel_ch4, 0)
    ch4_diario_kg = np.convolve(entrada, kernel_ch4, mode='full')[:dias_projecao] * ch4_pot_por_kg
    co2eq_diario_t = (ch4_diario_kg * GWP_CH4) / 1000.0
    return ch4_diario_kg, co2eq_diario_t

def calcular_co2eq_aterro_20anos(massa_t_ano, mcf, k_ano, doc):
    if massa_t_ano <= 0 or mcf <= 0:
        return 0.0
    massa_kg = massa_t_ano * 1000
    _, co2eq_dia = calcular_emissoes_aterro_diario(massa_kg, mcf, k_ano, T_ORGANICO, doc)
    return co2eq_dia.sum()

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
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =========================================================
@st.cache_data
def load_data(ano):
    url = URLS_POR_ANO[ano]
    df = pd.read_excel(url, sheet_name="Manejo_Coleta_e_Destinação", header=12)
    df = df.dropna(how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df

df = load_data(ano_selecionado)

COL_CODIGO_ROTA = df.columns[16]
COL_MUNICIPIO = df.columns[2]
COL_TIPO_COLETA = df.columns[17]
COL_MASSA = df.columns[24]
COL_DESTINO = df.columns[28]
COL_UF = df.columns[3]

df = df.rename(columns={
    COL_MUNICIPIO: "MUNICÍPIO",
    COL_TIPO_COLETA: "TIPO_COLETA_EXECUTADA",
    COL_MASSA: "MASSA_COLETADA"
})

COL_MUNICIPIO = "MUNICÍPIO"
COL_TIPO_COLETA = "TIPO_COLETA_EXECUTADA"
COL_MASSA = "MASSA_COLETADA"

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
tab_tradicional, tab_ia = st.tabs(["📊 Análise Tradicional (SNIS)", "🤖 Insights com Inteligência Artificial"])

# ======================== ABA TRADICIONAL ========================
with tab_tradicional:
    st.subheader(f"🇧🇷 Brasil — Síntese Nacional de RSU ({ano_selecionado})" if municipio == municipios[0] else f"📍 {municipio} - Ano {ano_selecionado}")

    # =========================================================
    # 1. 🗺️ Destinação Final
    # =========================================================
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
    st.caption("📌 Classificação dos destinos feita pela IA (PLN) para padronizar as variações textuais do SNIS.")

    st.markdown("#### 📋 Detalhamento por rota de coleta")
    tabela_destino = df_mun_dest[[COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_DESTINO, "MASSA_FLOAT"]].copy()
    tabela_destino = tabela_destino.rename(columns={
        COL_CODIGO_ROTA: "Código Rota",
        COL_TIPO_COLETA: "Tipo de Coleta",
        COL_DESTINO: "Tipo de Unidade (SNIS)",
        "MASSA_FLOAT": "Massa (t)"
    })
    tabela_destino["%"] = (tabela_destino["Massa (t)"] / massa_total_geral) * 100 if massa_total_geral > 0 else 0
    tabela_destino["Massa (t)"] = tabela_destino["Massa (t)"].apply(formatar_numero_br)
    tabela_destino["%"] = tabela_destino["%"].apply(lambda x: formatar_numero_br(x, 1))

    st.dataframe(
        tabela_destino[["Código Rota", "Tipo de Coleta", "Tipo de Unidade (SNIS)", "Massa (t)", "%"]],
        use_container_width=True
    )
    st.caption("📌 Os dados refletem fielmente os registros do SNIS. A classificação dos destinos é feita pela IA.")

    # =========================================================
    # 2. 📊 Distribuição por tipo de destino (Brasil)
    # =========================================================
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
            agg_destino.rename(columns={COL_DESTINO: "Tipo de Unidade (SNIS)"})[["Tipo de Unidade (SNIS)", "Massa (t)", "Percentual (%)"]],
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

        # =========================================================
        # 3. 🏳️ Coleta de RSU pelos estados
        # =========================================================
        st.markdown("---")
        st.subheader(f"🏳️ Coleta de RSU pelos estados do Brasil ({ano_selecionado})")

        ocultar_transbordo_est = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_est")

        df_estados = df_mun_dest.copy()
        if ocultar_transbordo_est:
            df_estados = df_estados[~df_estados[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]

        massa_total_est = df_estados["MASSA_FLOAT"].sum()
        agg_estados = df_estados.groupby(COL_UF)["MASSA_FLOAT"].sum().reset_index()
        agg_estados = agg_estados.sort_values("MASSA_FLOAT", ascending=False)
        agg_estados["%"] = (agg_estados["MASSA_FLOAT"] / massa_total_est) * 100 if massa_total_est > 0 else 0
        agg_estados["% acumulado"] = agg_estados["%"].cumsum()

        agg_estados["Massa (t)"] = agg_estados["MASSA_FLOAT"].apply(formatar_numero_br)
        agg_estados["%"] = agg_estados["%"].apply(lambda x: formatar_numero_br(x, 2))
        agg_estados["% acumulado"] = agg_estados["% acumulado"].apply(lambda x: formatar_numero_br(x, 2))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                agg_estados.rename(columns={COL_UF: "Estado"})[["Estado", "Massa (t)", "%", "% acumulado"]],
                use_container_width=True
            )
        with col2:
            fig, ax = plt.subplots(figsize=(6, 8))
            top_estados = agg_estados.head(10)
            ax.barh(top_estados[COL_UF], top_estados["MASSA_FLOAT"], color='forestgreen')
            ax.set_xlabel('Massa (t)')
            ax.set_title('Top 10 estados')
            ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # =========================================================
    # 4. 🏆 RANKING MUNICIPAL
    # =========================================================
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

                ranking_data = df_org_ranking.groupby([COL_MUNICIPIO, COL_UF, COL_DESTINO])["MASSA_FLOAT_RANK"].sum().reset_index()

                mapeamento = []
                preco = st.session_state.preco_carbono
                cambio = st.session_state.taxa_cambio

                for (mun, uf), grupo in ranking_data.groupby([COL_MUNICIPIO, COL_UF]):
                    massa_total_local = grupo["MASSA_FLOAT_RANK"].sum()
                    destinos = ", ".join(sorted(grupo[COL_DESTINO].unique()))
                    
                    grupo["MCF"] = grupo[COL_DESTINO].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
                    massa_aterro_local = grupo[grupo["MCF"] > 0]["MASSA_FLOAT_RANK"].sum()
                    
                    receita_anual = 0.0
                    if massa_aterro_local > 0:
                        df_mun_caract = df_clean[df_clean[COL_MUNICIPIO] == mun]
                        doc_pond, k_pond = calcular_doc_k_ponderado(df_mun_caract)
                        
                        co2eq_aterro = calcular_co2eq_aterro_20anos(massa_aterro_local, 0.8, k_pond, doc_pond)
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
                        "Tipo(s) de Unidade (SNIS)": destinos,
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
                - **DOC e k**: calculados dinamicamente a partir da caracterização dos resíduos do SNIS (quando disponível).
                - **% da massa total**: percentual da massa total de RSU do município que é composta por orgânicos da coleta seletiva.
                - Receita potencial anual considerando o preço atual do carbono.
                """)

    # =========================================================
    # 5. ♻️ ORGÂNICOS
    # =========================================================
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

    # =========================================================
    # 6. 🌳 PODAS E GALHADAS
    # =========================================================
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

    # =========================================================
    # Rodapé da aba tradicional
    # =========================================================
    st.markdown("---")
    st.caption(f"""
    Fonte: SNIS (ano {ano_selecionado}) | **Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F)** | IPCC AR5 (GWP-100)
    Baseline (aterro): CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28 | Compostagem: CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265
    DOC/k: ponderados pela caracterização dos resíduos do SNIS (quando disponível) | Cotações em tempo real via Yahoo Finance e APIs de câmbio.
    """)

# ======================== ABA DE IA ========================
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
    
    # =========================================================
    # CLASSIFICAÇÃO DE DESTINOS (PLN)
    # =========================================================
    st.subheader("📋 Classificação Inteligente de Destinos (PLN)")
    
    st.markdown("""
    O SNIS apresenta **diversas variações textuais** para descrever o mesmo destino 
    (ex: "Aterro Sanitário", "AS", "Aterro Sani.", "Aterro – Gerenciado"). 

    O **Composta.IA** utiliza um modelo de **Regressão Logística com TF-IDF** para:
    - ✅ Generalizar padrões textuais com alta acurácia
    - 🔍 Exibir o nível de confiança de cada classificação
    - 🛡️ Recair para regras manuais quando a confiança é baixa (fallback seguro)
    """)
    
    # Comparação: Regra vs IA
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
    
    # Distribuição dos destinos pela IA
    st.subheader("📊 Distribuição Nacional de Destinos (Classificação por IA)")
    
    @st.cache_data
    def classificar_todos_destinos(df, col_destino):
        return df[col_destino].apply(lambda x: classificador_ia.prever(x, threshold=0.3))
    
    with st.spinner("🤖 Classificando todos os destinos com IA..."):
        df_clean['destino_ia'] = classificar_todos_destinos(df_clean, COL_DESTINO)
    
    contagem_ia = df_clean['destino_ia'].value_counts().reset_index()
    contagem_ia.columns = ['Destino (IA)', 'Quantidade']
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cores = plt.cm.Set3(np.linspace(0, 1, len(contagem_ia)))
    wedges, texts, autotexts = ax1.pie(
        contagem_ia['Quantidade'],
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        colors=cores,
        textprops={'fontsize': 9},
        pctdistance=0.7,
    )
    ax1.legend(wedges, contagem_ia['Destino (IA)'],
               title="Destino",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1),
               fontsize=9)
    ax1.axis('equal')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    st.dataframe(
        contagem_ia.style.format({
            "Quantidade": lambda x: formatar_numero_br(x, 0)
        }),
        use_container_width=True
    )
    
    # =========================================================
    # CLUSTERIZAÇÃO DE MUNICÍPIOS
    # =========================================================
    st.markdown("---")
    st.subheader("📈 Clusterização de Municípios por Perfil de Resíduos")
    
    st.markdown("""
    Agrupamos municípios com perfis semelhantes de geração e destinação de resíduos usando **K-Means**.
    Isso ajuda a identificar quais municípios são prioritários para políticas de compostagem.
    """)
    
    if st.button("🔍 Executar Clusterização"):
        with st.spinner("Agrupando municípios por similaridade..."):
            try:
                from utils.ia_clustering import (
                    preparar_dados_clusterizacao,
                    clusterizar_municipios,
                    aplicar_pca,
                    plot_clusters,
                    resumo_clusters,
                    descrever_clusters
                )
                
                X, df_cluster = preparar_dados_clusterizacao(df_clean)
                if X.empty:
                    st.warning("Dados insuficientes para clusterização.")
                else:
                    n_clusters = st.slider("Número de clusters:", 2, 6, 4)
                    labels, kmeans, scaler = clusterizar_municipios(X, n_clusters=n_clusters)
                    df_cluster['Cluster'] = labels
                    
                    X_pca, pca = aplicar_pca(X)
                    fig = plot_clusters(X_pca, labels, df_cluster)
                    st.pyplot(fig)
                    
                    st.subheader("📊 Resumo dos Clusters")
                    resumo = resumo_clusters(df_cluster, labels)
                    st.dataframe(resumo.style.format({
                        'Massa_Media': '{:.0f}',
                        'Massa_Mediana': '{:.0f}',
                        'Massa_Total_Cluster': '{:.0f}',
                        'Rotas_Media': '{:.1f}',
                        'Pct_Aterro_Media': '{:.1f}',
                        'Pct_Compostagem_Media': '{:.1f}'
                    }))
                    
                    st.subheader("📝 Perfil de cada Cluster")
                    descricoes = descrever_clusters(df_cluster, labels)
                    for cluster in sorted(descricoes.keys()):
                        with st.expander(f"Cluster {cluster+1} – Clique para ver detalhes"):
                            st.markdown(descricoes[cluster])
                    
                    st.subheader("📍 Municípios por Cluster")
                    for cluster in sorted(df_cluster['Cluster'].unique()):
                        with st.expander(f"Cluster {cluster+1}"):
                            municipios_cluster = df_cluster[df_cluster['Cluster'] == cluster][['MUNICÍPIO', 'UF', 'Massa_Total']]
                            municipios_cluster = municipios_cluster.sort_values('Massa_Total', ascending=False)
                            st.dataframe(municipios_cluster.style.format({
                                'Massa_Total': '{:.0f}'
                            }), use_container_width=True)
            except Exception as e:
                st.error(f"Erro na clusterização: {e}")
                st.info("ℹ️ Verifique se o arquivo `utils/ia_clustering.py` está atualizado.")
    
    # =========================================================
    # SEÇÃO 1: PREVISÃO DE GERAÇÃO PER CAPITA
    # =========================================================
    st.markdown("---")
    st.subheader("📈 Previsão de Geração de Resíduos por Habitante")
    
    st.markdown("""
    Projeta a quantidade de resíduos que o município (ou o Brasil inteiro) precisará gerenciar 
    com base no crescimento populacional. A geração per capita é mantida constante a partir dos dados atuais do SNIS.
    """)
    
    opcoes_proj = ["BRASIL – Todos os municípios"] + sorted(df_clean[COL_MUNICIPIO].unique())
    municipio_proj = st.selectbox(
        "Selecione o município (ou Brasil) para projeção:",
        opcoes_proj,
        key="proj_municipio"
    )
    
    if municipio_proj:
        if municipio_proj == "BRASIL – Todos os municípios":
            df_mun_proj = df_clean.copy()
            massa_atual = df_mun_proj['MASSA_COLETADA'].sum()
            pop_atual = st.number_input(
                "População total do Brasil (habitantes) – IBGE 2024:", 
                min_value=1000, value=210000000, step=1000000
            )
            titulo_proj = "Brasil"
        else:
            df_mun_proj = df_clean[df_clean[COL_MUNICIPIO] == municipio_proj]
            massa_atual = df_mun_proj['MASSA_COLETADA'].sum()
            pop_atual = st.number_input(
                f"População atual do município (habitantes) – {municipio_proj}:", 
                min_value=100, value=50000, step=1000
            )
            titulo_proj = municipio_proj
        
        if massa_atual <= 0:
            st.warning("Não há dados de massa coletada para a seleção.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                taxa_pop = st.slider("Taxa de crescimento populacional anual (%)", 0.0, 5.0, 1.0, 0.1) / 100
            with col2:
                anos_proj = st.slider("Anos de projeção", 5, 30, 10)
            
            if st.button("📊 Projetar Resíduos per Capita"):
                with st.spinner("Calculando projeções..."):
                    try:
                        df_proj = projetar_residuos_per_capita(pop_atual, massa_atual, taxa_pop, anos_proj)
                        fig = plot_projecao_residuos(df_proj)
                        st.pyplot(fig)
                        
                        st.dataframe(df_proj.style.format({
                            'Populacao_Projetada': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                            'Massa_Projetada_ton': lambda x: formatar_br(x, auto_precision=False, casas_override=0)
                        }))
                        
                        ultimo = df_proj.iloc[-1]
                        if titulo_proj == "Brasil":
                            st.success(f"📌 **Em {ultimo['Ano']:.0f}, o Brasil precisará gerenciar aproximadamente {formatar_br(ultimo['Massa_Projetada_ton'], auto_precision=False, casas_override=0)} toneladas de resíduos.**")
                        else:
                            st.success(f"📌 **Em {ultimo['Ano']:.0f}, o município {titulo_proj} precisará gerenciar aproximadamente {formatar_br(ultimo['Massa_Projetada_ton'], auto_precision=False, casas_override=0)} toneladas de resíduos.**")
                    except Exception as e:
                        st.error(f"Erro na projeção: {e}")
    
    # =========================================================
    # SEÇÃO 2: SIMULAÇÃO DE CENÁRIOS DE COMPOSTAGEM
    # =========================================================
    st.markdown("---")
    st.subheader("💰 Simulador: Quanto o município (ou o Brasil) pode ganhar com créditos de carbono?")
    
    st.markdown("""
    Simule o impacto financeiro de **aumentar gradualmente** a quantidade de orgânicos desviada do aterro para a compostagem.
    """)
    
    opcoes_sim = ["BRASIL – Todos os municípios"] + sorted(df_clean[COL_MUNICIPIO].unique())
    municipio_sim = st.selectbox(
        "Selecione o município (ou Brasil) para a simulação:",
        opcoes_sim,
        key="sim_municipio"
    )
    
    if municipio_sim:
        if municipio_sim == "BRASIL – Todos os municípios":
            df_mun_sim = df_clean.copy()
            titulo_sim = "Brasil"
        else:
            df_mun_sim = df_clean[df_clean[COL_MUNICIPIO] == municipio_sim]
            titulo_sim = municipio_sim
        
        df_mun_sim['MCF'] = df_mun_sim[COL_DESTINO].apply(determinar_mcf_por_destino)
        df_org_aterro = df_mun_sim[df_mun_sim['MCF'] > 0]
        massa_aterro_atual = df_org_aterro['MASSA_COLETADA'].sum()
        
        if massa_aterro_atual <= 0:
            st.warning("Esta seleção não envia resíduos orgânicos para aterro (já utiliza compostagem ou reciclagem total).")
        else:
            col1, col2 = st.columns(2)
            with col1:
                taxa_crescimento = st.slider("Taxa anual de aumento da compostagem (%)", 5, 30, 15, 1) / 100
                anos_sim = st.slider("Anos de projeção", 5, 20, 10)
            with col2:
                inflacao_carbono = st.slider("Inflação anual do preço do carbono (%)", 0, 5, 2, 1) / 100
            
            if st.button("🚀 Executar Simulação de Cenários"):
                with st.spinner("Calculando projeções..."):
                    try:
                        doc_pond, k_pond = calcular_doc_k_ponderado(df_mun_sim)
                        co2_aterro = calcular_co2eq_aterro_20anos(massa_aterro_atual, 0.8, k_pond, doc_pond)
                        co2_compostagem = calcular_co2eq_compostagem_UNFCCC(massa_aterro_atual)
                        co2_evitado_por_t = (co2_aterro - co2_compostagem) / massa_aterro_atual if massa_aterro_atual > 0 else 0
                        
                        if co2_evitado_por_t <= 0:
                            st.warning("O coeficiente de emissões evitadas é zero ou negativo. Verifique os cálculos.")
                        else:
                            df_sim = simular_cenarios_compostagem(
                                massa_aterro_atual,
                                co2_evitado_por_t,
                                st.session_state.preco_carbono,
                                st.session_state.taxa_cambio,
                                anos_projecao=anos_sim,
                                taxa_crescimento_compostagem=taxa_crescimento,
                                inflacao_carbono=inflacao_carbono
                            )
                            
                            fig = plot_simulacao_compostagem(df_sim)
                            st.pyplot(fig)
                            
                            st.subheader("📈 Detalhamento Anual")
                            st.dataframe(df_sim.style.format({
                                'Massa_Desviada_Acumulada(t)': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                                'Receita_Anual_BRL': lambda x: f"R$ {formatar_br(x, auto_precision=False, casas_override=2)}",
                                'Ganho_Adicional_BRL': lambda x: f"R$ {formatar_br(x, auto_precision=False, casas_override=2)}",
                                'Receita_Acumulada_BRL': lambda x: f"R$ {formatar_br(x, auto_precision=False, casas_override=2)}"
                            }))
                            
                            valor_final = df_sim['Receita_Acumulada_BRL'].iloc[-1]
                            valor_final_fmt = formatar_br(valor_final, auto_precision=False, casas_override=2)
                            st.success(f"💰 **Potencial total em {anos_sim} anos para {titulo_sim}: R$ {valor_final_fmt}**")
                            
                            with st.expander("📊 Ver detalhamento dos cálculos (baseline e compostagem)"):
                                st.markdown("""
                                ### 🔍 Metodologia utilizada
                                - **Baseline (aterro)**: UNFCCC A6.4-AMT-003 – CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28.
                                - **Cenário de compostagem**: UNFCCC TOOL13 / AMS-III.F – CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265.
                                - **Emissões evitadas** = emissões do aterro – emissões da compostagem.
                                """)
                                
                                st.markdown(f"""
                                **📌 Dados de entrada:**
                                - Massa de orgânicos que vai para aterro atualmente: **{formatar_br(massa_aterro_atual, auto_precision=False, casas_override=0)} t/ano**
                                - **Taxa de decaimento (k) utilizada:** {formatar_br(k_pond, auto_precision=False, casas_override=3)} ano⁻¹
                                - **DOC utilizado:** {formatar_br(doc_pond, auto_precision=False, casas_override=3)}
                                - Coeficiente de emissões do aterro por tonelada: **{formatar_br(co2_aterro / massa_aterro_atual if massa_aterro_atual > 0 else 0, auto_precision=False, casas_override=2)} tCO₂e/t**
                                - Coeficiente de emissões da compostagem por tonelada: **{formatar_br(co2_compostagem / massa_aterro_atual if massa_aterro_atual > 0 else 0, auto_precision=False, casas_override=2)} tCO₂e/t**
                                - Emissões evitadas por tonelada desviada: **{formatar_br(co2_evitado_por_t, auto_precision=False, casas_override=2)} tCO₂e/t**
                                """)
                                
                                st.info("""
                                💡 **Interpretação:**  
                                A cada ano, a quantidade de resíduos desviada para compostagem aumenta, gerando mais emissões evitadas e, consequentemente, mais receita com créditos de carbono.  
                                O valor acumulado mostra o potencial total de ganhos ao longo do período.
                                """)
                            
                            st.info("ℹ️ Esta simulação considera o aumento gradual da compostagem ano a ano, com base nos dados atuais do SNIS. O valor é acumulado.")
                    except Exception as e:
                        st.error(f"Erro na simulação: {e}")

    # =========================================================
    # SEÇÃO 3: CENÁRIOS DE EXPANSÃO DA COMPOSTAGEM (COM TEXTO REFINADO)
    # =========================================================
    st.markdown("---")
    st.subheader("🌍 Cenários de Expansão da Compostagem no Brasil")

    st.markdown("""
    Esta seção analisa o cenário atual da compostagem de resíduos orgânicos no Brasil e projeta cenários futuros,
    priorizando os municípios que já possuem coleta seletiva de orgânicos.
    """)

    # -----------------------------------------------------------------
    # 1. Identificar municípios com coleta seletiva de orgânicos
    # -----------------------------------------------------------------
    mask_organicos = df_clean[COL_TIPO_COLETA].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True
    )
    df_org = df_clean[mask_organicos].copy()

    if df_org.empty:
        st.info("Nenhum município registrou coleta seletiva de resíduos orgânicos no SNIS para este ano.")
    else:
        # -----------------------------------------------------------------
        # 2. Calcular massas de aterro e compostagem por município
        # -----------------------------------------------------------------
        df_org['MCF'] = df_org[COL_DESTINO].apply(determinar_mcf_por_destino)
        df_aterro = df_org[df_org['MCF'] > 0].groupby(COL_MUNICIPIO).agg({COL_MASSA: 'sum'}).reset_index()
        df_aterro.rename(columns={COL_MASSA: 'Massa_Aterro'}, inplace=True)
        df_compost = df_org[df_org['MCF'] == 0].groupby(COL_MUNICIPIO).agg({COL_MASSA: 'sum'}).reset_index()
        df_compost.rename(columns={COL_MASSA: 'Massa_Compostagem'}, inplace=True)

        df_mun_cenario = pd.merge(df_aterro, df_compost, on=COL_MUNICIPIO, how='outer').fillna(0)
        df_mun_cenario['Massa_Total'] = df_mun_cenario['Massa_Aterro'] + df_mun_cenario['Massa_Compostagem']

        df_uf_mun = df_clean[[COL_MUNICIPIO, COL_UF]].drop_duplicates(subset=[COL_MUNICIPIO])
        df_uf_mun.rename(columns={COL_UF: 'UF'}, inplace=True)
        df_mun_cenario = pd.merge(df_mun_cenario, df_uf_mun, on=COL_MUNICIPIO, how='left')

        # Percentual de compostagem sobre a massa orgânica coletada seletivamente
        df_mun_cenario['Pct_Compostagem'] = (df_mun_cenario['Massa_Compostagem'] / df_mun_cenario['Massa_Total']) * 100
        df_mun_cenario['Pct_Compostagem'] = df_mun_cenario['Pct_Compostagem'].fillna(0).round(2)

        # -----------------------------------------------------------------
        # 2.1 Calcular o percentual de coleta seletiva dos municípios que já têm coleta
        # -----------------------------------------------------------------
        df_total_mun = df_clean.groupby(COL_MUNICIPIO).agg({COL_MASSA: 'sum'}).reset_index()
        df_total_mun.rename(columns={COL_MASSA: 'Massa_Total_RSU'}, inplace=True)

        df_org_sum = df_org.groupby(COL_MUNICIPIO).agg({COL_MASSA: 'sum'}).reset_index()
        df_org_sum.rename(columns={COL_MASSA: 'Massa_Seletiva_Organicos'}, inplace=True)

        df_pct_seletiva = pd.merge(df_total_mun, df_org_sum, on=COL_MUNICIPIO, how='inner')
        df_pct_seletiva['Pct_Seletiva'] = (df_pct_seletiva['Massa_Seletiva_Organicos'] / df_pct_seletiva['Massa_Total_RSU']) * 100
        df_pct_seletiva['Pct_Seletiva'] = df_pct_seletiva['Pct_Seletiva'].fillna(0).round(2)

        df_uf_mun2 = df_clean[[COL_MUNICIPIO, COL_UF]].drop_duplicates(subset=[COL_MUNICIPIO])
        df_uf_mun2 = df_uf_mun2.rename(columns={COL_UF: 'UF'})
        df_pct_seletiva = pd.merge(df_pct_seletiva, df_uf_mun2, on=COL_MUNICIPIO, how='left')

        if 'UF' not in df_pct_seletiva.columns:
            df_pct_seletiva['UF'] = ''

        # -----------------------------------------------------------------
        # 3. Exibir resumo atual
        # -----------------------------------------------------------------
        total_aterro = df_mun_cenario['Massa_Aterro'].sum()
        total_compost = df_mun_cenario['Massa_Compostagem'].sum()
        total_massa_org = total_aterro + total_compost
        pct_compost_real = (total_compost / total_massa_org) * 100 if total_massa_org > 0 else 0

        st.markdown("### 📊 Cenário Atual")
        col1, col2, col3 = st.columns(3)
        col1.metric("Municípios com coleta seletiva", len(df_mun_cenario))
        col2.metric("Total de orgânicos coletados", f"{formatar_br(total_massa_org, auto_precision=False, casas_override=0)} t")
        col3.metric("Percentual destinado à compostagem", f"{formatar_br(pct_compost_real, auto_precision=False, casas_override=2)}%")

        st.markdown("#### 📋 Municípios com coleta seletiva – detalhamento")
        df_mun_detalhe = df_mun_cenario[['MUNICÍPIO', 'UF', 'Massa_Total', 'Massa_Compostagem', 'Pct_Compostagem']].copy()
        df_mun_detalhe = df_mun_detalhe.sort_values('Pct_Compostagem', ascending=False)
        st.dataframe(
            df_mun_detalhe.style.format({
                'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                'Massa_Compostagem': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                'Pct_Compostagem': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%'
            }),
            use_container_width=True
        )

        with st.expander("📋 Percentual de coleta seletiva dos municípios com coleta seletiva"):
            cols_to_show = ['MUNICÍPIO', 'Massa_Total_RSU', 'Massa_Seletiva_Organicos', 'Pct_Seletiva']
            if 'UF' in df_pct_seletiva.columns:
                cols_to_show.insert(1, 'UF')
            st.dataframe(
                df_pct_seletiva[cols_to_show].style.format({
                    'Massa_Total_RSU': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Massa_Seletiva_Organicos': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Pct_Seletiva': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%'
                }),
                use_container_width=True
            )

        # -----------------------------------------------------------------
        # 4. Cenário 1 – Atual (com percentual real de compostagem)
        # -----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📌 Cenário 1 – Situação Atual (com percentual real de compostagem)")

        doc_medio, k_medio = DOC_PADRAO, K_PADRAO
        co2_aterro_por_t = calcular_co2eq_aterro_20anos(1, 0.8, k_medio, doc_medio)
        co2_compost_por_t = calcular_co2eq_compostagem_UNFCCC(1)
        co2_evitado_por_t = co2_aterro_por_t - co2_compost_por_t

        massa_compost_real = total_compost
        evitado_atual = massa_compost_real * co2_evitado_por_t
        receita_atual = evitado_atual * st.session_state.preco_carbono * st.session_state.taxa_cambio

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Massa compostada atualmente",
                f"{formatar_br(massa_compost_real, auto_precision=False, casas_override=0)} t"
            )
        with col2:
            st.metric(
                "Emissões evitadas (atual)",
                f"{formatar_br(evitado_atual, auto_precision=False, casas_override=2)} tCO₂e"
            )
        st.metric(
            "Receita anual estimada (atual)",
            f"R$ {formatar_br(receita_atual, auto_precision=False, casas_override=2)}"
        )
        st.caption(f"Percentual de compostagem sobre o total de orgânicos: {formatar_br(pct_compost_real, auto_precision=False, casas_override=2)}%")

        # -----------------------------------------------------------------
        # 5. Cenário 2 – Expansão da coleta seletiva para novos municípios
        # -----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📌 Cenário 2 – Expansão da coleta seletiva para novos municípios")

        # Identificar municípios sem coleta seletiva para obter os números
        df_total_geral = df_clean.groupby(COL_MUNICIPIO).agg({
            COL_MASSA: 'sum',
            COL_UF: 'first'
        }).reset_index()
        df_total_geral.rename(columns={COL_MASSA: 'Massa_Total_Geral', COL_UF: 'UF'}, inplace=True)

        municipios_com_seletiva = df_mun_cenario['MUNICÍPIO'].unique()
        df_sem_seletiva = df_total_geral[~df_total_geral[COL_MUNICIPIO].isin(municipios_com_seletiva)]
        massa_sem_seletiva = df_sem_seletiva['Massa_Total_Geral'].sum()
        num_sem_seletiva = len(df_sem_seletiva)
        num_com_seletiva = len(municipios_com_seletiva)

        # Texto descritivo com os números
        st.markdown(f"""
        Este cenário considera a implementação da coleta seletiva de orgânicos em municípios que atualmente não a possuem.

        **Contexto atual:**
        - **{num_com_seletiva} municípios** já possuem coleta seletiva de orgânicos.
        - **{num_sem_seletiva} municípios** ainda não possuem essa coleta.
        - Esses {num_sem_seletiva} municípios geram **{formatar_br(massa_sem_seletiva, auto_precision=False, casas_override=0)} t/ano** de resíduos sólidos urbanos.

        A meta de cobertura (percentual da massa total que será coletada seletivamente) é definida com base nos percentuais dos municípios que já possuem coleta seletiva:
        - **Realista**: utiliza o **1º quartil (25%)** dos percentuais de coleta seletiva.
        - **Otimista**: utiliza a **média** dos percentuais de coleta seletiva.

        A **massa adicional desviada** é calculada aplicando a meta de cobertura à massa total dos municípios sem coleta seletiva. Dessa massa adicional coletada seletivamente, uma fração é efetivamente compostada (usando o percentual real de compostagem **{formatar_br(pct_compost_real, auto_precision=False, casas_override=2)}%** observado nos municípios que já possuem coleta seletiva).
        """)

        if massa_sem_seletiva == 0:
            st.info("✅ Todos os municípios já possuem coleta seletiva de orgânicos. Não há expansão possível.")
        else:
            # Calcular percentis de coleta seletiva dos municípios que já têm
            pct_25 = df_pct_seletiva['Pct_Seletiva'].quantile(0.25)
            pct_media = df_pct_seletiva['Pct_Seletiva'].mean()

            # Fallback para evitar valores muito baixos ou inconsistentes
            if pct_25 <= 0.01 or np.isnan(pct_25):
                pct_25 = 0.5
                st.warning("⚠️ O 1º quartil calculado foi muito baixo ou nulo. Usando valor de fallback de 0,5% para o cenário realista.")
            if pct_media <= 0.01 or np.isnan(pct_media):
                pct_media = 2.0
                st.warning("⚠️ A média calculada foi muito baixa ou nula. Usando valor de fallback de 2,0% para o cenário otimista.")

            tipo_cenario = st.radio(
                "Escolha a meta de cobertura para os novos municípios:",
                options=["Realista (1º quartil)", "Otimista (média)"],
                index=0
            )

            if tipo_cenario == "Realista (1º quartil)":
                meta_cobertura = pct_25
                rotulo = f"1º quartil ({formatar_br(pct_25, auto_precision=False, casas_override=2)}%)"
            else:
                meta_cobertura = pct_media
                rotulo = f"média ({formatar_br(pct_media, auto_precision=False, casas_override=2)}%)"

            st.info(f"**Meta de cobertura escolhida:** {rotulo}")

            # Massa adicional que será coletada seletivamente
            massa_adicional_coletada = massa_sem_seletiva * (meta_cobertura / 100)

            # Dessa massa adicional coletada, qual percentual será efetivamente compostado?
            massa_adicional_compost = massa_adicional_coletada * (pct_compost_real / 100)

            evitado_adicional = massa_adicional_compost * co2_evitado_por_t
            receita_adicional = evitado_adicional * st.session_state.preco_carbono * st.session_state.taxa_cambio

            massa_total_compost_futuro = massa_compost_real + massa_adicional_compost
            evitado_total = evitado_atual + evitado_adicional
            receita_total = receita_atual + receita_adicional

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Massa adicional coletada seletivamente",
                    f"{formatar_br(massa_adicional_coletada, auto_precision=False, casas_override=0)} t"
                )
                st.caption(f"Meta de cobertura: {formatar_br(meta_cobertura, auto_precision=False, casas_override=2)}%")
            with col2:
                st.metric(
                    "Massa adicional compostada",
                    f"{formatar_br(massa_adicional_compost, auto_precision=False, casas_override=0)} t"
                )
                st.caption(f"Taxa de compostagem: {formatar_br(pct_compost_real, auto_precision=False, casas_override=2)}%")
            with col3:
                st.metric(
                    "Receita adicional",
                    f"R$ {formatar_br(receita_adicional, auto_precision=False, casas_override=2)}"
                )

            st.markdown("---")
            st.subheader("📈 Resultado consolidado após expansão")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Massa total compostada (futuro)",
                    f"{formatar_br(massa_total_compost_futuro, auto_precision=False, casas_override=0)} t"
                )
                st.caption(f"Acréscimo de {formatar_br((massa_adicional_compost/massa_compost_real)*100 if massa_compost_real>0 else 0, auto_precision=False, casas_override=1)}%")
            with col2:
                st.metric(
                    "Receita total anual (futuro)",
                    f"R$ {formatar_br(receita_total, auto_precision=False, casas_override=2)}"
                )
                st.caption(f"Acréscimo de {formatar_br((receita_adicional/receita_atual)*100 if receita_atual>0 else 0, auto_precision=False, casas_override=1)}%")

            with st.expander("📋 Principais municípios sem coleta seletiva (prioritários para expansão)"):
                df_top_sem = df_sem_seletiva.nlargest(10, 'Massa_Total_Geral')[['MUNICÍPIO', 'UF', 'Massa_Total_Geral']]
                st.dataframe(
                    df_top_sem.style.format({
                        'Massa_Total_Geral': lambda x: formatar_br(x, auto_precision=False, casas_override=0)
                    }),
                    use_container_width=True
                )
                st.caption(f"Total de {len(df_sem_seletiva)} municípios sem coleta seletiva, representando {formatar_br(massa_sem_seletiva, auto_precision=False, casas_override=0)} t de resíduos.")

            st.info("""
            💡 **Interpretação:**  
            - O **Cenário 1** mostra a situação atual, com o percentual real de compostagem sobre os orgânicos coletados seletivamente.  
            - O **Cenário 2** projeta o ganho ao implementar coleta seletiva em novos municípios, usando uma meta realista (1º quartil) ou otimista (média) de cobertura.  
            - A taxa de compostagem aplicada sobre a nova coleta é a mesma observada atualmente (percentual real).
            """)

    # =========================================================
    # SEÇÃO 4: ANÁLISE DE COBERTURA DA COLETA SELETIVA DE ORGÂNICOS
    # =========================================================
    st.markdown("---")
    st.subheader("📊 Análise de Cobertura da Coleta Seletiva de Orgânicos")

    st.markdown("""
    Esta seção analisa o percentual da massa total de resíduos que é coberta pela coleta seletiva de orgânicos em cada município,
    e projeta o impacto de uma expansão da cobertura para todos os municípios, com três cenários.
    """)

    # -----------------------------------------------------------------
    # 1. Calcular massa total e massa de coleta seletiva por município
    # -----------------------------------------------------------------
    df_total = df_clean.groupby(COL_MUNICIPIO).agg({
        COL_MASSA: 'sum',
        COL_UF: 'first'
    }).reset_index()
    df_total.rename(columns={
        COL_MASSA: 'Massa_Total',
        COL_UF: 'UF'
    }, inplace=True)

    mask_organicos = df_clean[COL_TIPO_COLETA].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True
    )
    df_seletiva = df_clean[mask_organicos].groupby(COL_MUNICIPIO).agg({COL_MASSA: 'sum'}).reset_index()
    df_seletiva.rename(columns={COL_MASSA: 'Massa_Seletiva_Organicos'}, inplace=True)

    df_cobertura = pd.merge(df_total, df_seletiva, on=COL_MUNICIPIO, how='left').fillna(0)
    df_cobertura['Pct_Seletiva'] = (df_cobertura['Massa_Seletiva_Organicos'] / df_cobertura['Massa_Total']) * 100
    df_cobertura['Pct_Seletiva'] = df_cobertura['Pct_Seletiva'].round(2)
    df_cobertura['Possui_Seletiva'] = df_cobertura['Massa_Seletiva_Organicos'] > 0

    # -----------------------------------------------------------------
    # 2. Resumo nacional
    # -----------------------------------------------------------------
    total_municipios = len(df_cobertura)
    municipios_com_seletiva = df_cobertura[df_cobertura['Possui_Seletiva']].shape[0]
    municipios_sem_seletiva = total_municipios - municipios_com_seletiva

    massa_total_brasil = df_cobertura['Massa_Total'].sum()
    massa_seletiva_brasil = df_cobertura['Massa_Seletiva_Organicos'].sum()
    pct_seletiva_brasil = (massa_seletiva_brasil / massa_total_brasil) * 100 if massa_total_brasil > 0 else 0
    media_pct_municipios = df_cobertura[df_cobertura['Massa_Total'] > 0]['Pct_Seletiva'].mean()

    st.markdown("### 📊 Resumo Nacional")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Municípios totais", total_municipios)
    col2.metric("Com coleta seletiva de orgânicos", municipios_com_seletiva)
    col3.metric("Sem coleta seletiva de orgânicos", municipios_sem_seletiva)
    col4.metric("Massa total de RSU", f"{formatar_br(massa_total_brasil, auto_precision=False, casas_override=0)} t")

    col1, col2, col3 = st.columns(3)
    col1.metric("Massa em coleta seletiva orgânica", f"{formatar_br(massa_seletiva_brasil, auto_precision=False, casas_override=0)} t")
    col2.metric("Percentual nacional (massa)", f"{formatar_br(pct_seletiva_brasil, auto_precision=False, casas_override=2)}%")
    col3.metric("Média municipal (não ponderada)", f"{formatar_br(media_pct_municipios, auto_precision=False, casas_override=2)}%")

    # =========================================================
    # 🏆 TOP 10 MUNICÍPIOS
    # =========================================================
    st.markdown("### 🏆 Destaques da Coleta Seletiva de Orgânicos")

    df_com_seletiva = df_cobertura[df_cobertura['Possui_Seletiva']].copy()
    if not df_com_seletiva.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Top 10 – Maior Percentual de Cobertura")
            top_pct = df_com_seletiva.nlargest(10, 'Pct_Seletiva')
            top_pct = top_pct[['MUNICÍPIO', 'UF', 'Massa_Seletiva_Organicos', 'Massa_Total', 'Pct_Seletiva']]
            st.dataframe(
                top_pct.style.format({
                    'Massa_Seletiva_Organicos': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Pct_Seletiva': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%'
                }),
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.markdown("#### 📊 Top 10 – Maior Massa Destinada à Compostagem")
            top_massa = df_com_seletiva.nlargest(10, 'Massa_Seletiva_Organicos')
            top_massa = top_massa[['MUNICÍPIO', 'UF', 'Massa_Seletiva_Organicos', 'Massa_Total', 'Pct_Seletiva']]
            st.dataframe(
                top_massa.style.format({
                    'Massa_Seletiva_Organicos': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                    'Pct_Seletiva': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Nenhum município com coleta seletiva de orgânicos para listar.")

    with st.expander("📋 Detalhamento por município (clique para expandir)"):
        st.dataframe(
            df_cobertura.style.format({
                'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                'Massa_Seletiva_Organicos': lambda x: formatar_br(x, auto_precision=False, casas_override=0),
                'Pct_Seletiva': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%'
            }),
            use_container_width=True
        )

    st.subheader("📊 Distribuição dos percentuais de cobertura")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df_cobertura[df_cobertura['Massa_Total'] > 0]
    bins = np.linspace(0, 100, 21)
    ax.hist(df_plot['Pct_Seletiva'], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(pct_seletiva_brasil, color='red', linestyle='--', label=f'Média nacional (massa): {pct_seletiva_brasil:.2f}%')
    ax.axvline(media_pct_municipios, color='green', linestyle='--', label=f'Média municipal: {media_pct_municipios:.2f}%')
    ax.set_xlabel('Percentual de coleta seletiva de orgânicos (%)')
    ax.set_ylabel('Número de municípios')
    ax.set_title('Distribuição da cobertura da coleta seletiva de orgânicos por município')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3. Cenários de expansão (com massa compostada em todos os cards)
    # -----------------------------------------------------------------
    st.markdown("### 🚀 Cenários de Expansão da Cobertura")

    st.markdown("""
    Três cenários são apresentados para a universalização da coleta seletiva de orgânicos:
    - **Cenário Atual (Pessimista)**: mantém a situação atual (sem expansão).
    - **Cenário Realista**: municípios sem coleta seletiva alcançam o **1º quartil (25%)** dos percentuais dos municípios que já possuem coleta seletiva.
    - **Cenário Otimista**: municípios sem coleta seletiva alcançam a **média** dos percentuais dos municípios que já possuem coleta seletiva.
    """)

    df_com_seletiva = df_cobertura[df_cobertura['Possui_Seletiva']]
    if not df_com_seletiva.empty and len(df_com_seletiva) >= 4:
        pct_25 = np.percentile(df_com_seletiva['Pct_Seletiva'], 25)
        pct_media = df_com_seletiva['Pct_Seletiva'].mean()
    elif not df_com_seletiva.empty:
        pct_25 = df_com_seletiva['Pct_Seletiva'].min()
        pct_media = df_com_seletiva['Pct_Seletiva'].mean()
    else:
        pct_25 = 0
        pct_media = 0

    # Fallback para evitar valores inconsistentes
    if pct_25 <= 0.01 or np.isnan(pct_25):
        pct_25 = 0.5
        st.warning("⚠️ O 1º quartil calculado foi muito baixo ou nulo. Usando valor de fallback de 0,5% para o cenário realista.")
    if pct_media <= 0.01 or np.isnan(pct_media):
        pct_media = 2.0
        st.warning("⚠️ A média calculada foi muito baixa ou nula. Usando valor de fallback de 2,0% para o cenário otimista.")

    df_sem_seletiva = df_cobertura[~df_cobertura['Possui_Seletiva']]
    massa_sem_seletiva = df_sem_seletiva['Massa_Total'].sum()

    massa_adicional_realista = massa_sem_seletiva * (pct_25 / 100) if pct_25 > 0 else 0
    massa_adicional_otimista = massa_sem_seletiva * (pct_media / 100) if pct_media > 0 else 0

    doc_medio, k_medio = DOC_PADRAO, K_PADRAO
    co2_aterro_por_t = calcular_co2eq_aterro_20anos(1, 0.8, k_medio, doc_medio)
    co2_compost_por_t = calcular_co2eq_compostagem_UNFCCC(1)
    co2_evitado_por_t = co2_aterro_por_t - co2_compost_por_t

    massa_compostada_atual = massa_seletiva_brasil

    evitado_atual = massa_compostada_atual * co2_evitado_por_t
    receita_atual = evitado_atual * st.session_state.preco_carbono * st.session_state.taxa_cambio

    evitado_adicional_realista = massa_adicional_realista * co2_evitado_por_t
    massa_compostada_realista = massa_compostada_atual + massa_adicional_realista
    evitado_total_realista = evitado_atual + evitado_adicional_realista
    receita_total_realista = receita_atual + (evitado_adicional_realista * st.session_state.preco_carbono * st.session_state.taxa_cambio)

    evitado_adicional_otimista = massa_adicional_otimista * co2_evitado_por_t
    massa_compostada_otimista = massa_compostada_atual + massa_adicional_otimista
    evitado_total_otimista = evitado_atual + evitado_adicional_otimista
    receita_total_otimista = receita_atual + (evitado_adicional_otimista * st.session_state.preco_carbono * st.session_state.taxa_cambio)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📉 Cenário Atual (Pessimista)")
        st.metric("Massa compostada", f"{formatar_br(massa_compostada_atual, auto_precision=False, casas_override=0)} t")
        st.metric("Emissões evitadas", f"{formatar_br(evitado_atual, auto_precision=False, casas_override=2)} tCO₂e")
        st.metric("Receita", f"R$ {formatar_br(receita_atual, auto_precision=False, casas_override=2)}")
        st.caption(f"Cobertura nacional: {formatar_br(pct_seletiva_brasil, auto_precision=False, casas_override=2)}%")
    with col2:
        st.markdown("#### 📊 Cenário Realista (1º Quartil)")
        st.metric("Massa compostada", f"{formatar_br(massa_compostada_realista, auto_precision=False, casas_override=0)} t")
        st.metric("Massa adicional", f"{formatar_br(massa_adicional_realista, auto_precision=False, casas_override=0)} t")
        st.metric("Emissões evitadas totais", f"{formatar_br(evitado_total_realista, auto_precision=False, casas_override=2)} tCO₂e")
        st.metric("Receita total", f"R$ {formatar_br(receita_total_realista, auto_precision=False, casas_override=2)}")
        st.caption(f"Meta de cobertura: {formatar_br(pct_25, auto_precision=False, casas_override=2)}% (1º quartil)")
    with col3:
        st.markdown("#### 📈 Cenário Otimista (Média)")
        st.metric("Massa compostada", f"{formatar_br(massa_compostada_otimista, auto_precision=False, casas_override=0)} t")
        st.metric("Massa adicional", f"{formatar_br(massa_adicional_otimista, auto_precision=False, casas_override=0)} t")
        st.metric("Emissões evitadas totais", f"{formatar_br(evitado_total_otimista, auto_precision=False, casas_override=2)} tCO₂e")
        st.metric("Receita total", f"R$ {formatar_br(receita_total_otimista, auto_precision=False, casas_override=2)}")
        st.caption(f"Meta de cobertura: {formatar_br(pct_media, auto_precision=False, casas_override=2)}% (média)")

    with st.expander("📋 Municípios com menores percentuais de cobertura (referência para os cenários)"):
        st.markdown("#### 📊 Menores percentuais **positivos** (> 0%)")
        if not df_com_seletiva.empty:
            df_referencia = df_com_seletiva.nsmallest(10, 'Pct_Seletiva')[['MUNICÍPIO', 'UF', 'Pct_Seletiva', 'Massa_Total']]
            st.dataframe(
                df_referencia.style.format({
                    'Pct_Seletiva': lambda x: formatar_br(x, auto_precision=False, casas_override=2) + '%',
                    'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0)
                }),
                use_container_width=True
            )
            st.caption(f"📌 O cenário realista usa o 1º quartil ({formatar_br(pct_25, auto_precision=False, casas_override=2)}%) como meta, baseado nos 25% menores percentuais entre os que já possuem coleta seletiva.")
        else:
            st.info("Nenhum município com coleta seletiva para referência.")

        st.markdown("---")
        st.markdown("#### 🚫 Municípios com **0% de cobertura** (sem coleta seletiva de orgânicos)")
        if not df_sem_seletiva.empty:
            df_zero = df_sem_seletiva.nlargest(10, 'Massa_Total')[['MUNICÍPIO', 'UF', 'Massa_Total']]
            st.dataframe(
                df_zero.style.format({
                    'Massa_Total': lambda x: formatar_br(x, auto_precision=False, casas_override=0)
                }),
                use_container_width=True
            )
            total_zero = len(df_sem_seletiva)
            massa_zero = df_sem_seletiva['Massa_Total'].sum()
            st.caption(f"📌 Total de {total_zero} municípios sem coleta seletiva, que representam {formatar_br(massa_zero, auto_precision=False, casas_override=0)} t de resíduos (potencial de expansão).")
        else:
            st.success("✅ Todos os municípios já possuem coleta seletiva de orgânicos!")

    st.info("""
    💡 **Interpretação:**  
    - O cenário atual mostra as emissões evitadas com a infraestrutura existente.  
    - O cenário realista é uma meta factível, baseada no que os municípios com menores índices já conseguem alcançar.  
    - O cenário otimista representa uma meta mais ambiciosa, baseada na média dos municípios que já possuem coleta seletiva.  
    - A receita total considera o preço do carbono e o câmbio atuais.
    """)

# =========================================================
# AUTORIA E USO
# =========================================================
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

# =========================================================
# RODAPÉ GERAL DO APP
# =========================================================
st.markdown("---")
st.caption("""
**Composta.IA** | Ferramenta de apoio à gestão de resíduos sólidos e créditos de carbono  
Dados: SNIS (2023/2024) | Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F) | IPCC AR5 (GWP-100)
""")
