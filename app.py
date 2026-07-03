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

# =========================================================
# PARÂMETROS UNFCCC A6.4-AMT-003-v01.0 (2025) – Application B (Tropical Wet)
# =========================================================
GWP_CH4 = 28.0      # Tabela 9 da norma (IPCC AR5)
GWP_N2O = 265.0     # IPCC AR5 (para compostagem)
PHI_APPLICATION_B = 0.85   # Tabela 5 – Tropical úmido
OX_SOIL_COVER = 0.383      # Tabela 6 – Aterro com cobertura de solo
F_METHANE_FRACTION = 0.5   # Tabela 3 – Fração de metano no gás
MCF_DEFAULT_BULK = 0.8     # Fallback para destinos não classificados

# Parâmetros fixos do modelo FOD (base UNFCCC)
ANOS_PROJECAO = 20
DIAS_PROJECAO = ANOS_PROJECAO * 365
T_ORGANICO = 25.0          # Temperatura média (°C) – Brasil tropical
DOC_PADRAO = 0.15          # Bulk waste (quando não há caracterização)
K_PADRAO = 0.07            # Bulk waste – Tropical wet (Tabela 10)

# =========================================================
# FUNÇÃO PARA CALCULAR DOC e k PONDERADOS (VIA SNIS)
# =========================================================
def calcular_doc_k_ponderado(df_municipio):
    """
    Calcula DOC e k ponderados com base na caracterização dos resíduos do SNIS.
    Usa as Tabelas 7 e 10 da UNFCCC A6.4-AMT-003 (Tropical Wet).
    """
    colunas_caract = {
        'Alimentos_Verdes': 'GTR1501',
        'Vidros': 'GTR1502',
        'Metais': 'GTR1503',
        'Plasticos': 'GTR1504',
        'Papeis': 'GTR1505',
        'Têxteis': 'GTR1506',
        'Outros': 'GTR1507'
    }
    
    df_caract = df_municipio[[col for col in colunas_caract.values() if col in df_municipio.columns]].copy()
    if df_caract.empty or df_caract.isna().all().all():
        return DOC_PADRAO, K_PADRAO
    
    for col in df_caract.columns:
        df_caract[col] = pd.to_numeric(df_caract[col], errors='coerce').fillna(0)
    
    pct = {}
    for nome, col in colunas_caract.items():
        if col in df_caract.columns:
            val = df_caract[col].mean()
            pct[nome] = val if val > 0 else 0
        else:
            pct[nome] = 0
    
    # Valores da Tabela 7 (DOC) e Tabela 10 (k) – Tropical Wet
    doc_pond = (pct['Alimentos_Verdes'] * 0.7 +
                pct['Papeis'] * 0.5 +
                pct['Têxteis'] * 0.24 +
                pct['Outros'] * 0.1) / 100.0
    
    k_pond = (pct['Alimentos_Verdes'] * 0.17 +
              pct['Papeis'] * 0.07 +
              pct['Têxteis'] * 0.07 +
              pct['Outros'] * 0.035) / 100.0
    
    doc_pond = max(doc_pond, DOC_PADRAO)
    k_pond = max(k_pond, K_PADRAO)
    
    return doc_pond, k_pond

# =========================================================
# FUNÇÕES DE CÁLCULO – ATERRO (BASELINE UNFCCC - APENAS CH4)
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
    """
    Calcula APENAS as emissões de METANO (CH4) do aterro.
    Totalmente alinhado com a UNFCCC A6.4-AMT-003 (Application B - Tropical Wet).
    Retorna: (ch4_diario_kg, co2eq_diario_t)
    """
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

# =========================================================
# FUNÇÃO DA COMPOSTAGEM (UNFCCC TOOL13)
# =========================================================
def calcular_co2eq_compostagem_UNFCCC(massa_t_ano):
    """
    Emissões da compostagem usando fatores padrão UNFCCC (AMS-III.F / TOOL13).
    CH4 = 0,002 kg CH4 / kg resíduo úmido
    N2O = 0,0002 kg N2O / kg resíduo úmido
    GWP: CH4=28, N2O=265 (IPCC AR5)
    """
    if massa_t_ano <= 0:
        return 0.0
    massa_kg = massa_t_ano * 1000
    ch4_kg = massa_kg * 0.002
    n2o_kg = massa_kg * 0.0002
    co2eq_t = (ch4_kg * GWP_CH4 + n2o_kg * GWP_N2O) / 1000.0
    return co2eq_t

# =========================================================
# MCF POR DESTINO
# =========================================================
def determinar_mcf_por_destino(destino, tipo_residuo='organico'):
    if pd.isna(destino):
        return 0.0
    destino_norm = normalizar_texto(destino)
    if "ATERRO SANITARIO" in destino_norm:
        mcf_base = 1.0 if "GERENCIADO" in destino_norm or "COLETA" in destino_norm else 0.8
    elif "ATERRO CONTROLADO" in destino_norm:
        mcf_base = 0.4
    elif "LIXAO" in destino_norm or "VAZADOURO" in destino_norm:
        mcf_base = 0.4
    else:
        mcf_base = 0.0
    return mcf_base

# =========================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =========================================================
@st.cache_data
def load_data(ano):
    url = URLS_POR_ANO[ano]
    df = pd.read_excel(url, sheet_name="Manejo_Coleta_e_Destinação", header=13)
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
# INICIALIZAÇÃO DA INTELIGÊNCIA ARTIFICIAL (PLN)
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
    # 🗺️ Destinação Final
    # =========================================================
    st.markdown("---")
    st.subheader(f"🗺️ Para onde o resíduo está indo? (Destinação Final, {ano_selecionado})")

    ocultar_transbordo = st.checkbox("Ocultar transbordos", value=False)

    if ocultar_transbordo:
        df_mun = df_mun[~df_mun[COL_DESTINO].apply(
            lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
        )]

    df_mun["MASSA_FLOAT"] = pd.to_numeric(df_mun[COL_MASSA], errors="coerce").fillna(0)

    massa_total = df_mun["MASSA_FLOAT"].sum()
    st.markdown(f"### Total de resíduos coletados: **{formatar_numero_br(massa_total)} t**")
    st.markdown("""
    A tabela abaixo exibe **cada rota de coleta** e seu respectivo tipo de unidade, exatamente como declarado no SNIS.
    Nenhuma agregação ou filtro foi aplicado – os valores correspondem à massa anual coletada para cada rota e destino.
    """)

    tabela_destino = df_mun[[COL_CODIGO_ROTA, COL_TIPO_COLETA, COL_DESTINO, "MASSA_FLOAT"]].copy()
    tabela_destino = tabela_destino.rename(columns={
        COL_CODIGO_ROTA: "Código Rota",
        COL_TIPO_COLETA: "Tipo de Coleta",
        COL_DESTINO: "Tipo de Unidade (SNIS)",
        "MASSA_FLOAT": "Massa (t)"
    })

    tabela_destino["%"] = (tabela_destino["Massa (t)"] / massa_total) * 100 if massa_total > 0 else 0
    tabela_destino["Massa (t)"] = tabela_destino["Massa (t)"].apply(formatar_numero_br)
    tabela_destino["%"] = tabela_destino["%"].apply(lambda x: formatar_numero_br(x, 1))

    st.dataframe(tabela_destino[["Código Rota", "Tipo de Coleta", "Tipo de Unidade (SNIS)", "Massa (t)", "%"]], use_container_width=True)
    st.caption("📌 Os dados refletem fielmente os registros do SNIS.")

    # =========================================================
    # 📊 Distribuição por tipo de destino (Brasil)
    # =========================================================
    if municipio == municipios[0]:
        st.markdown("---")
        st.subheader(f"📊 Distribuição dos resíduos por tipo de destino ({ano_selecionado})")

        ocultar_transbordo_dist = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_dist")

        df_dist = df_mun.copy()
        if ocultar_transbordo_dist:
            df_dist = df_dist[~df_dist[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]

        massa_total_dist = df_dist["MASSA_FLOAT"].sum()
        st.markdown(f"### Total de resíduos coletados: **{formatar_numero_br(massa_total_dist)} t**")

        agg_destino = df_dist.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_destino = agg_destino.sort_values("MASSA_FLOAT", ascending=False)
        agg_destino["Percentual (%)"] = (agg_destino["MASSA_FLOAT"] / massa_total_dist) * 100 if massa_total_dist > 0 else 0
        agg_destino["Massa (t)"] = agg_destino["MASSA_FLOAT"].apply(formatar_numero_br)
        agg_destino["Percentual (%)"] = agg_destino["Percentual (%)"].apply(lambda x: formatar_numero_br(x, 2))
        st.dataframe(
            agg_destino.rename(columns={COL_DESTINO: "Tipo de Unidade (SNIS)"})[["Tipo de Unidade (SNIS)", "Massa (t)", "Percentual (%)"]],
            use_container_width=True
        )
        st.caption("Nota: a soma das massas pode exceder o total coletado devido a duplicidades nas rotas (ex.: transbordo e destino final).")

        # =========================================================
        # 🏳️ Coleta de RSU pelos estados do Brasil
        # =========================================================
        st.markdown("---")
        st.subheader(f"🏳️ Coleta de RSU pelos estados do Brasil ({ano_selecionado})")

        ocultar_transbordo_est = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_est")

        df_estados = df_mun.copy()
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

        st.dataframe(
            agg_estados.rename(columns={COL_UF: "Estado"})[["Estado", "Massa (t)", "%", "% acumulado"]],
            use_container_width=True
        )

    # =========================================================
    # 🏆 RANKING MUNICIPAL (COM DOC/k DINÂMICO)
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
                col_m2.metric("Massa p/ Compostagem", f"{formatar_numero_br(pct_comp, 1)}%")
                col_m3.metric("Massa p/ Aterro", f"{formatar_numero_br(pct_aterro, 1)}%")

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

                    mapeamento.append({
                        "Município": mun,
                        "UF": uf,
                        "Massa Total (t/ano)": massa_total_local,
                        "Massa para Aterro (t/ano)": massa_aterro_local,
                        "Tipo(s) de Unidade (SNIS)": destinos,
                        "Receita Potencial (R$/ano)": receita_anual
                    })

                df_mapeamento = pd.DataFrame(mapeamento).sort_values("Massa Total (t/ano)", ascending=False)

                st.dataframe(df_mapeamento.style.format({
                    "Massa Total (t/ano)": lambda x: formatar_numero_br(x, None),
                    "Massa para Aterro (t/ano)": lambda x: formatar_numero_br(x, None),
                    "Receita Potencial (R$/ano)": lambda x: f"R$ {formatar_numero_br(x, None)}"
                }), use_container_width=True, height=600)

                st.caption("""
                - **Baseline (aterro)**: alinhado à UNFCCC A6.4-AMT-003 (Application B) – CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28.
                - **Cenário de compostagem**: UNFCCC TOOL13 (AMS-III.F) – CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265.
                - **DOC e k**: calculados dinamicamente a partir da caracterização dos resíduos do SNIS (quando disponível).
                - Receita potencial anual considerando o preço atual do carbono.
                """)

    # =========================================================
    # ♻️ ORGÂNICOS (com DOC/k dinâmico)
    # =========================================================
    st.markdown("---")
    st.subheader(f"♻️ Destinação da Coleta Seletiva de Resíduos Orgânicos ({ano_selecionado})")
    df_organicos = df_mun[df_mun[COL_TIPO_COLETA].astype(str).str.contains(
        "seletiva.*orgânico|orgânico.*seletiva", case=False, na=False, regex=True)].copy()

    if not df_organicos.empty:
        df_organicos["MASSA_FLOAT"] = pd.to_numeric(df_organicos[COL_MASSA], errors="coerce").fillna(0)

        ocultar_transbordo_org = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_org")

        df_mun_org = df_mun.copy()
        if ocultar_transbordo_org:
            df_organicos = df_organicos[~df_organicos[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
            df_mun_org = df_mun_org[~df_mun_org[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]

        total_organicos = df_organicos["MASSA_FLOAT"].sum()
        massa_total_geral = df_mun_org["MASSA_FLOAT"].sum()

        st.markdown(f"### Total de orgânicos coletados seletivamente: **{formatar_numero_br(total_organicos)} t**")

        st.markdown("#### Tabela – Destino da coleta de recicláveis orgânicos")
        agg_org = df_organicos.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        agg_org = agg_org.sort_values("MASSA_FLOAT", ascending=False)
        agg_org["% do tipo"] = (agg_org["MASSA_FLOAT"] / total_organicos) * 100 if total_organicos > 0 else 0
        agg_org["% do total no ano"] = (agg_org["MASSA_FLOAT"] / massa_total_geral) * 100 if massa_total_geral > 0 else 0

        linhas = []
        for _, row in agg_org.iterrows():
            linhas.append({
                "Destino": row[COL_DESTINO],
                "Massa Anual (t)": formatar_numero_br(row["MASSA_FLOAT"], 2),
                "% do tipo": formatar_numero_br(row["% do tipo"], 2),
                "% do total no ano": formatar_numero_br(row["% do total no ano"], 4)
            })

        perc_total_tipo = (total_organicos / massa_total_geral) * 100 if massa_total_geral > 0 else 0
        linhas.append({
            "Destino": "Total do tipo",
            "Massa Anual (t)": formatar_numero_br(total_organicos, 2),
            "% do tipo": "100,00%",
            "% do total no ano": formatar_numero_br(perc_total_tipo, 4)
        })

        linhas.append({
            "Destino": "Total no ano",
            "Massa Anual (t)": formatar_numero_br(massa_total_geral, 2),
            "% do tipo": " - ",
            "% do total no ano": "100,00%"
        })

        df_resumo = pd.DataFrame(linhas)
        st.dataframe(df_resumo, use_container_width=True)

        st.markdown("#### Detalhamento por destino")
        df_org_dest = df_organicos.groupby(COL_DESTINO)["MASSA_FLOAT"].sum().reset_index()
        df_org_dest["%"] = (df_org_dest["MASSA_FLOAT"] / total_organicos) * 100 if total_organicos > 0 else 0
        df_org_dest = df_org_dest.sort_values("%", ascending=False)
        df_org_dest_view = df_org_dest.copy()
        df_org_dest_view["Massa (t)"] = df_org_dest_view["MASSA_FLOAT"].apply(formatar_numero_br)
        df_org_dest_view["%"] = df_org_dest_view["%"].apply(lambda x: formatar_numero_br(x, 1))
        st.dataframe(
            df_org_dest_view.rename(columns={COL_DESTINO: "Tipo de Unidade (SNIS)"})[["Tipo de Unidade (SNIS)", "Massa (t)", "%"]],
            use_container_width=True
        )

        st.subheader("🔥 Emissões detalhadas (Orgânicos) – Metodologia UNFCCC")
        df_org_dest["MCF"] = df_org_dest[COL_DESTINO].apply(lambda x: determinar_mcf_por_destino(x, 'organico'))
        resultados = []
        co2eq_aterro_total = 0.0
        massa_aterro_total = 0.0

        doc_pond, k_pond = calcular_doc_k_ponderado(df_mun)

        for _, row in df_org_dest.iterrows():
            massa_t, mcf = row["MASSA_FLOAT"], row["MCF"]
            if mcf > 0 and massa_t > 0:
                co2eq_aterro = calcular_co2eq_aterro_20anos(massa_t, mcf, k_pond, doc_pond)
                co2eq_aterro_total += co2eq_aterro
                massa_aterro_total += massa_t
                resultados.append({
                    "Tipo de Unidade (SNIS)": row[COL_DESTINO],
                    "Massa (t)": formatar_numero_br(massa_t),
                    "MCF": formatar_numero_br(mcf),
                    "CO₂e aterro (20 anos)": formatar_numero_br(co2eq_aterro)
                })

        if resultados:
            st.dataframe(pd.DataFrame(resultados), use_container_width=True)

            co2eq_compostagem = calcular_co2eq_compostagem_UNFCCC(massa_aterro_total)
            evitado = co2eq_aterro_total - co2eq_compostagem

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Massa em aterros", formatar_massa_br(massa_aterro_total))
            col2.metric("CO₂e aterro (20 anos)", f"{formatar_numero_br(co2eq_aterro_total)} tCO₂e")
            col3.metric("CO₂e compostagem (20 anos)", f"{formatar_numero_br(co2eq_compostagem)} tCO₂e")
            col4.metric("Emissões Evitadas", f"{formatar_numero_br(evitado)} tCO₂e")

            # =========================================================
            # 💰 POTENCIAL DE CRÉDITOS DE CARBONO
            # =========================================================
            st.markdown("---")
            st.subheader("💰 Potencial de Créditos de Carbono (Compostagem - UNFCCC)")

            with st.container():
                st.markdown("### 🌍 Cotações de Mercado")
                col_cot1, col_cot2, col_cot3 = st.columns(3)
                with col_cot1:
                    if st.button("🔄 Atualizar Cotações"):
                        preco, moeda, _, _, _ = obter_cotacao_carbono()
                        cambio, moeda_r, _, _ = obter_cotacao_euro_real()
                        st.session_state.preco_carbono = preco
                        st.session_state.moeda_carbono = moeda
                        st.session_state.taxa_cambio = cambio
                        st.session_state.moeda_real = moeda_r
                        st.rerun()
                preco = st.session_state.preco_carbono
                moeda = st.session_state.moeda_carbono
                cambio = st.session_state.taxa_cambio
                with col_cot2:
                    st.metric("Carbono", f"{moeda} {formatar_br(preco)}/tCO₂e")
                with col_cot3:
                    st.metric("Câmbio EUR/BRL", f"R$ {formatar_br(cambio)}")
                st.metric("Preço em R$", f"R$ {formatar_br(preco * cambio)}/tCO₂e")

            valor_total_eur = calcular_valor_creditos(evitado, preco, "€")
            valor_total_brl = calcular_valor_creditos(evitado, preco, "R$", cambio)

            st.metric("Valor total em Reais (R$)", f"R$ {formatar_br(valor_total_brl)}")
            st.caption(f"Equivalente a {moeda} {formatar_br(valor_total_eur)}")

            st.info("ℹ️ **Metodologia:**\n\n"
                    "- **Baseline (aterro)**: UNFCCC A6.4-AMT-003 – CH₄ apenas, φ=0.85, OX=0.383, GWP_CH4=28.\n"
                    "- **Cenário de compostagem**: UNFCCC TOOL13 (AMS-III.F) – CH₄=0.002, N₂O=0.0002, GWP_CH4=28, GWP_N2O=265.\n"
                    "- **DOC e k**: ponderados pela caracterização dos resíduos (quando disponível).\n"
                    "- As quantidades exibidas na tela são arredondadas para duas casas decimais.")

        else:
            st.success("✅ Nenhum orgânico destinado a aterro.")
    else:
        st.info("ℹ️ Sem registros de coleta seletiva de orgânicos.")

    # =========================================================
    # 🌳 DESTINO DA COLETA DE PODAS E GALHADAS
    # =========================================================
    st.markdown("---")
    st.subheader(f"🌳 Destinação da coleta de podas e galhadas ({ano_selecionado})")
    df_podas = df_mun[df_mun[COL_TIPO_COLETA].astype(str).str.contains("áreas verdes públicas", case=False, na=False)].copy()

    if not df_podas.empty:
        df_podas["MASSA_FLOAT"] = pd.to_numeric(df_podas[COL_MASSA], errors="coerce").fillna(0)

        ocultar_transbordo_podas = st.checkbox("Ocultar transbordos", value=False, key="ocultar_transbordo_podas")

        df_mun_podas = df_mun.copy()
        if ocultar_transbordo_podas:
            df_podas = df_podas[~df_podas[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]
            df_mun_podas = df_mun_podas[~df_mun_podas[COL_DESTINO].apply(
                lambda x: "TRANSBORDO" in normalizar_texto(x) if pd.notna(x) else False
            )]

        total_podas = df_podas["MASSA_FLOAT"].sum()
        massa_total_geral_podas = df_mun_podas["MASSA_FLOAT"].sum()

        st.markdown(f"### Total de podas e galhadas coletadas: **{formatar_numero_br(total_podas)} t**")

        st.markdown("#### Tabela – Destino da coleta de podas e galhadas")
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
    st.header("🧠 Classificação Inteligente de Destinos (Processamento de Linguagem Natural)")
    
    st.markdown("""
    O SNIS possui **mais de 100 variações textuais** para descrever o mesmo destino 
    (ex: "Aterro Sanitário", "AS", "Aterro Sani.", "Aterro – Gerenciado"). 
    
    O **Composta.IA** utiliza um modelo de **Regressão Logística com TF-IDF** para:
    - ✅ Generalizar padrões textuais com alta acurácia (>95%)
    - 🔍 Exibir o nível de confiança de cada classificação
    - 🛡️ Recair para regras manuais quando a confiança é baixa (fallback seguro)
    """)
    
    # =========================================================
    # COMPARAÇÃO: Regra vs IA
    # =========================================================
    st.subheader("📋 Comparação: Regra Manual vs. Inteligência Artificial")
    
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
    
    # =========================================================
    # DISTRIBUIÇÃO DOS DESTINOS PELA IA
    # =========================================================
    st.subheader("📊 Distribuição Nacional de Destinos (Classificação por IA)")
    
    @st.cache_data
    def classificar_todos_destinos(df, col_destino):
        return df[col_destino].apply(lambda x: classificador_ia.prever(x, threshold=0.3))
    
    with st.spinner("🤖 Classificando todos os destinos com IA..."):
        df_clean['destino_ia'] = classificar_todos_destinos(df_clean, COL_DESTINO)
    
    contagem_ia = df_clean['destino_ia'].value_counts().reset_index()
    contagem_ia.columns = ['Destino (IA)', 'Quantidade']
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cores = plt.cm.Set3(np.linspace(0, 1, len(contagem_ia)))
    ax1.pie(contagem_ia['Quantidade'], 
            labels=contagem_ia['Destino (IA)'], 
            autopct='%1.1f%%', 
            startangle=90,
            colors=cores,
            textprops={'fontsize': 9})
    ax1.axis('equal')
    st.pyplot(fig1)
    
    st.dataframe(
        contagem_ia.style.format({
            "Quantidade": lambda x: formatar_numero_br(x, 0)
        }),
        use_container_width=True
    )
    
    # =========================================================
    # PRÓXIMOS PASSOS
    # =========================================================
    st.markdown("---")
    st.subheader("🚀 Em breve: Novos módulos de IA")
    
    col_prox1, col_prox2 = st.columns(2)
    
    with col_prox1:
        st.markdown("""
        ### 📈 Clusterização de Municípios
        Agruparemos municípios com perfis semelhantes de geração e destinação de resíduos usando **K-Means**.
        
        *Benefício:* Identificar quais municípios são prioritários para políticas de compostagem.
        """)
    
    with col_prox2:
        st.markdown("""
        ### 🔮 Previsão de Geração de Resíduos
        Utilizaremos **Random Forest** para projetar a geração de RSU para os próximos 5 anos.
        
        *Benefício:* Auxiliar no planejamento de aterros, usinas e metas de reciclagem.
        """)
    
    st.info("💡 Esses módulos serão adicionados na próxima etapa. Fique ligado!")

# =========================================================
# RODAPÉ GERAL DO APP
# =========================================================
st.markdown("---")
st.caption("""
**Composta.IA** | 30º Concurso Inovação no Setor Público - Categoria IV (Inteligência Artificial para o Bem Público) | 
Dados: SNIS (2023/2024) | Metodologia: UNFCCC A6.4-AMT-003 (2025) + TOOL13 (AMS-III.F) | IPCC AR5 (GWP-100)
""")
