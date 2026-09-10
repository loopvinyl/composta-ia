# ============================================================
# SIMULADOR DE EMISSÕES DE tCO2eq E CÁLCULO DE CRÉDITOS DE CARBONO
# COM ANÁLISE DE SENSIBILIDADE GLOBAL
# Versão: 2026 | Autor: Cássio Luiz Vellani (adaptado)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fftconvolve
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import yfinance as yf
import requests
from datetime import datetime
import io
import os

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Simulador de Emissões de tCO2eq e Créditos de Carbono",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS MODERNO E PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    /* ---------- Layout geral ---------- */
    .main { background-color: #f7f9fc; }
    .block-container { padding-top: 1.5rem; }

    /* ---------- Cabeçalho principal ---------- */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #27ae60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }

    /* ---------- Cards de métricas ---------- */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        border-left: 5px solid #2a5298;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 0.8rem;
    }
    .metric-card .label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #1a202c; }
    .metric-card .delta { font-size: 0.8rem; color: #38a169; }

    /* ---------- Cards de dados REAIS (SINISA) ---------- */
    .real-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    .real-card h3 { color: #1b5e20; font-size: 0.95rem; margin: 0 0 0.4rem 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .real-card .big-number { font-size: 1.8rem; font-weight: 800; color: #2e7d32; }
    .real-card .unit { font-size: 0.85rem; color: #4caf50; }

    /* ---------- Cards de dados ESTIMADOS ---------- */
    .estimated-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px dashed #ffb300;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        height: 100%;
        position: relative;
    }
    .estimated-card h3 { color: #e65100; font-size: 0.95rem; margin: 0 0 0.4rem 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .estimated-card .big-number { font-size: 1.8rem; font-weight: 800; color: #ef6c00; }
    .estimated-card .unit { font-size: 0.85rem; color: #f57c00; }

    /* ---------- Badges ---------- */
    .badge-real {
        display: inline-block;
        background: #2e7d32; color: white;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .badge-estimado {
        display: inline-block;
        background: #ef6c00; color: white;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ---------- Caixa de nota explicativa ---------- */
    .nota-box {
        background: #fff8e1;
        border-left: 5px solid #ffb300;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0 1.5rem 0;
        font-size: 0.9rem;
        color: #5d4037;
    }
    .nota-box strong { color: #e65100; }

    /* ---------- Títulos de seção ---------- */
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #1a202c;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }
    .section-title-green { border-bottom-color: #27ae60; }
    .section-title-orange { border-bottom-color: #ef6c00; }

    /* ---------- Resultados ---------- */
    .result-highlight {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .result-highlight .label { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }
    .result-highlight .value { font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; }
    .result-highlight .sub { font-size: 0.85rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CLASSE DE CÁLCULO DE EMISSÕES (GHG EMISSION CALCULATOR)
# ============================================================
class GHGEmissionCalculator:
    """
    Classe responsável por calcular as emissões de GEE para os cenários
    de aterro sanitário (baseline), vermicompostagem e compostagem termofílica.
    """

    def __init__(self,
                 TOC=0.436,
                 TN=0.0142,
                 f_CH4_vermi=0.0013,
                 f_N2O_vermi=0.0092,
                 f_CH4_thermo=0.0060,
                 f_N2O_thermo=0.0196,
                 GWP_CH4_20=79.7, GWP_N2O_20=273.0,
                 GWP_CH4_100=27.0, GWP_N2O_100=273.0,
                 GWP_CH4_500=7.2, GWP_N2O_500=130.0,
                 MCF=1.0, F=0.5, OX=0.1, Ri=0.0,
                 phi=0.85):

        self.TOC = TOC
        self.TN = TN
        self.f_CH4_vermi = f_CH4_vermi
        self.f_N2O_vermi = f_N2O_vermi
        self.f_CH4_thermo = f_CH4_thermo
        self.f_N2O_thermo = f_N2O_thermo

        self.GWP = {
            "GWP-20":  {"CH4": GWP_CH4_20,  "N2O": GWP_N2O_20},
            "GWP-100": {"CH4": GWP_CH4_100, "N2O": GWP_N2O_100},
            "GWP-500": {"CH4": GWP_CH4_500, "N2O": GWP_N2O_500},
        }

        self.MCF = MCF
        self.F = F
        self.OX = OX
        self.Ri = Ri
        self.phi = phi

        # Perfil temporal de emissões de N2O no pré-descarte (Feng et al., 2020)
        self.f_N2O_pre = np.array([0.8623, 0.10, 0.0377])

        # Perfil temporal de emissões de N2O no aterro (Wang et al., 2017)
        self.f_N2O_aterro = np.array([0.10, 0.30, 0.40, 0.15, 0.05])

    # ------------------------------------------------------------
    # FASE DE PRÉ-DESCARTE
    # ------------------------------------------------------------
    def calculate_pre_disposal_emissions(self, daily_mass_kg):
        """Emissões diárias de CH4 e N2O no pré-descarte (Feng et al., 2020)."""
        # CH4: 2,78 µg C/kg/h × 24h × 16/12 × 10^-9 → kg CH4 / kg resíduo
        E_CH4_pre = 2.78 * 24 * (16.0 / 12.0) * 1e-9
        # N2O: 20,26 mg N/kg × 44/28 × 10^-6 → kg N2O / kg resíduo
        E_N2O_pre = 20.26 * (44.0 / 28.0) * 1e-6

        ch4_daily = daily_mass_kg * E_CH4_pre
        n2o_daily = daily_mass_kg * E_N2O_pre * self.f_N2O_pre
        return ch4_daily, n2o_daily

    # ------------------------------------------------------------
    # ATERRO SANITÁRIO (MODELO FOD - IPCC)
    # ------------------------------------------------------------
    def calculate_landfill_emissions(self, daily_mass_kg, years, k,
                                     T=25.0, DOC=0.15, U=0.85):
        """Calcula emissões acumuladas do aterro (CH4 + N2O) ao longo de `years`."""
        n_days = int(years * 365)
        daily_input = np.full(n_days, daily_mass_kg, dtype=float)

        # DOCf em função da temperatura
        DOCf = 0.0147 * T + 0.28

        # Potencial de geração de CH4 por kg de resíduo (kg CH4 / kg resíduo)
        Lo = (DOC * DOCf * self.MCF * self.F * (16.0 / 12.0))

        # Aplicar OX (oxidação na camada de cobertura)
        Lo_eff = Lo * (1 - self.OX)

        # Aplicar phi (correção de umidade)
        Lo_eff *= self.phi

        # Kernel de decaimento de primeira ordem (diário)
        k_day = k / 365.0
        t_kernel = np.arange(n_days)
        kernel = k_day * np.exp(-k_day * t_kernel)

        # Convolução (resposta ao impulso)
        ch4_series = fftconvolve(daily_input, kernel)[:n_days] * Lo_eff

        # N2O no aterro (Wang et al., 2017)
        E_N2O_exp = 1.91e-6 * (44.0 / 28.0)   # kg N2O / kg resíduo (área exposta)
        E_N2O_cob = 2.15e-6 * (44.0 / 28.0)   # kg N2O / kg resíduo (área coberta)
        f_exp = np.clip((100.0 / daily_mass_kg) * (8.0 / 24.0), 0, 1)
        E_N2O_medio = (f_exp * E_N2O_exp + (1 - f_exp) * E_N2O_cob) * ((1 - U) / (1 - 0.55))

        n2o_series = np.zeros(n_days)
        for i in range(n_days):
            for j, frac in enumerate(self.f_N2O_aterro):
                if i + j < n_days:
                    n2o_series[i + j] += daily_input[i] * E_N2O_medio * frac

        return ch4_series, n2o_series

    # ------------------------------------------------------------
    # CENÁRIO PROJETO (VERMICOMPOSTAGEM / COMPOSTAGEM)
    # ------------------------------------------------------------
    def calculate_biological_emissions(self, daily_mass_kg, years,
                                        f_CH4, f_N2O, U=0.85):
        """Emissões diretas de CH4 e N2O na vermicompostagem/compostagem."""
        n_days = int(years * 365)
        daily_input = np.full(n_days, daily_mass_kg, dtype=float)

        E_CH4 = self.TOC * f_CH4 * (16.0 / 12.0) * (1 - U)
        E_N2O = self.TN * f_N2O * (44.0 / 28.0) * (1 - U)

        # Perfil temporal normalizado (50 dias, baseado em Yang et al., 2017)
        t = np.arange(50)
        profile = np.exp(-((t - 15) ** 2) / (2 * 8 ** 2))
        profile /= profile.sum()

        ch4_series = fftconvolve(daily_input, profile)[:n_days] * E_CH4
        n2o_series = fftconvolve(daily_input, profile)[:n_days] * E_N2O

        return ch4_series, n2o_series

    # ------------------------------------------------------------
    # CÁLCULO CONSOLIDADO DE EMISSÕES EVITADAS
    # ------------------------------------------------------------
    def calculate_avoided_emissions(self, daily_mass_kg, years, k,
                                    T=25.0, DOC=0.15, U=0.85):
        """Calcula emissões evitadas (EE) para todos os cenários de GWP."""
        # Baseline: pré-descarte + aterro
        ch4_pre, n2o_pre = self.calculate_pre_disposal_emissions(daily_mass_kg)
        ch4_land, n2o_land = self.calculate_landfill_emissions(
            daily_mass_kg, years, k, T=T, DOC=DOC, U=U
        )

        # Soma pré-descarte distribuída nos primeiros 3 dias
        n_days = int(years * 365)
        ch4_base = ch4_land.copy()
        n2o_base = n2o_land.copy()
        for i, frac in enumerate(self.f_N2O_pre):
            if i < n_days:
                n2o_base[i] += n2o_pre[i]
        ch4_base[:3] += ch4_pre[:3]

        # Projeto: vermicompostagem e compostagem
        ch4_vermi, n2o_vermi = self.calculate_biological_emissions(
            daily_mass_kg, years, self.f_CH4_vermi, self.f_N2O_vermi, U=U
        )
        ch4_thermo, n2o_thermo = self.calculate_biological_emissions(
            daily_mass_kg, years, self.f_CH4_thermo, self.f_N2O_thermo, U=U
        )

        results = {}
        for scenario, gwp in self.GWP.items():
            # Baseline em tCO2e
            base_tco2e = (ch4_base.sum() * gwp["CH4"] +
                          n2o_base.sum() * gwp["N2O"]) / 1000.0
            vermi_tco2e = (ch4_vermi.sum() * gwp["CH4"] +
                           n2o_vermi.sum() * gwp["N2O"]) / 1000.0
            thermo_tco2e = (ch4_thermo.sum() * gwp["CH4"] +
                            n2o_thermo.sum() * gwp["N2O"]) / 1000.0

            results[scenario] = {
                "baseline_tco2e": base_tco2e,
                "vermi_tco2e": vermi_tco2e,
                "thermo_tco2e": thermo_tco2e,
                "ee_vermi": base_tco2e - vermi_tco2e,
                "ee_thermo": base_tco2e - thermo_tco2e,
                "annual_vermi": (base_tco2e - vermi_tco2e) / years,
                "annual_thermo": (base_tco2e - thermo_tco2e) / years,
            }

        return results, ch4_base, n2o_base, ch4_vermi, n2o_vermi


# ============================================================
# FUNÇÕES DE MERCADO (COTAÇÕES EM TEMPO REAL)
# ============================================================
@st.cache_data(ttl=600)
def get_carbon_price():
    """Obtém a cotação do carbono no EU ETS (ticker CO2.L) via Yahoo Finance."""
    try:
        ticker = yf.Ticker("CO2.L")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 85.50  # fallback


@st.cache_data(ttl=600)
def get_eur_brl():
    """Obtém a cotação EUR/BRL via AwesomeAPI."""
    try:
        r = requests.get("https://economia.awesomeapi.com.br/last/EUR-BRL", timeout=5)
        return float(r.json()["EURBRL"]["bid"])
    except Exception:
        return 6.04


@st.cache_data(ttl=600)
def get_usd_brl():
    """Obtém a cotação USD/BRL via AwesomeAPI."""
    try:
        r = requests.get("https://economia.awesomeapi.com.br/last/USD-BRL", timeout=5)
        return float(r.json()["USDBRL"]["bid"])
    except Exception:
        return 5.25


# ============================================================
# SIDEBAR — PAINEL DE MERCADO E PARÂMETROS
# ============================================================
with st.sidebar:
    st.markdown("### 💱 Mercado de Carbono e Câmbio")
    if st.button("🔄 Atualizar Cotações", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    carbon_eur = get_carbon_price()
    eur_brl = get_eur_brl()
    usd_brl = get_usd_brl()

    st.metric("Preço Carbono (EU ETS)", f"€ {carbon_eur:,.2f}/tCO₂eq")
    st.metric("Euro (EUR/BRL)", f"R$ {eur_brl:,.2f}")
    st.metric("Dólar (USD/BRL)", f"R$ {usd_brl:,.2f}")
    st.metric("Carbono em Reais", f"R$ {carbon_eur * eur_brl:,.2f}/tCO₂eq")

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros de Entrada")

    daily_waste = st.slider("Quantidade de resíduos (kg/dia)", 10, 1000, 100, 10)
    k_decay = st.select_slider("Taxa de decaimento (k, ano⁻¹)",
                               options=[0.06, 0.10, 0.20, 0.30, 0.40], value=0.06)
    T_mean = st.slider("Temperatura média (°C)", 20.0, 40.0, 25.0, 0.5)
    DOC = st.slider("Carbono Orgânico Degradável (DOC)", 0.10, 0.25, 0.15, 0.01)
    humidity = st.slider("Umidade do resíduo (%)", 75, 90, 85, 1)

    st.markdown("---")
    st.markdown("### 🔬 Configuração de Simulação")

    years_sim = st.slider("Anos de simulação", 5, 50, 20, 1)
    n_mc = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_sobol = st.slider("Número de amostras Sobol", 32, 256, 64, 32)

    st.markdown("---")
    run_sim = st.button("🚀 Executar Simulação", type="primary", use_container_width=True)


# ============================================================
# ÁREA PRINCIPAL — CABEÇALHO
# ============================================================
st.markdown('<div class="hero-title">🌱 Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ferramenta para estimar emissões evitadas e créditos de carbono na gestão de resíduos sólidos urbanos.</div>', unsafe_allow_html=True)

st.info("💡 Ajuste os parâmetros na barra lateral e clique em **Executar Simulação** para ver os resultados.")


# ============================================================
# REFERÊNCIAS METODOLÓGICAS
# ============================================================
with st.expander("📚 Referências para Cálculo", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Base (Aterro Sanitário):**")
        st.markdown("- Metano: IPCC (2006, 2019), UNFCCC (2024) e Wang et al. (2023)")
        st.markdown("- Óxido Nitroso: Wang et al. (2017)")
        st.markdown("- Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020)")
        st.markdown("- Fator φ = 0,85 (UNFCCC, 2024) aplicado ao baseline para clima úmido")
    with col2:
        st.markdown("**Cenário Projeto (Vermicompostagem):**")
        st.markdown("- Metano e Óxido Nitroso: Yang et al. (2017)")
        st.markdown("**Cenário Projeto (Compostagem Termofílica):**")
        st.markdown("- Protocolo AMS-III.F: UNFCCC (2016)")
        st.markdown("- Fatores de emissão: Yang et al. (2017)")


# ============================================================
# DADOS REAIS DO SINISA + INDICADORES ESTIMADOS (NOVA SEÇÃO)
# ============================================================
st.markdown('<div class="section-title section-title-green">📊 Panorama Nacional de RSU — Dados Reais e Estimativas</div>', unsafe_allow_html=True)

# -------- LINHA 1: DADOS REAIS (SINISA) --------
st.markdown('<span class="badge-real">✓ Dados Reais — SINISA 2023/2024</span>', unsafe_allow_html=True)

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.markdown("""
    <div class="real-card">
        <h3>Total Coletado 2024</h3>
        <div class="big-number">79,93</div>
        <div class="unit">milhões de toneladas</div>
    </div>
    """, unsafe_allow_html=True)
with col_r2:
    st.markdown("""
    <div class="real-card">
        <h3>Total Coletado 2023</h3>
        <div class="big-number">73,48</div>
        <div class="unit">milhões de toneladas</div>
    </div>
    """, unsafe_allow_html=True)
with col_r3:
    st.markdown("""
    <div class="real-card">
        <h3>Recicláveis Orgânicos 2024</h3>
        <div class="big-number">147.457</div>
        <div class="unit">toneladas (0,18% do total)</div>
    </div>
    """, unsafe_allow_html=True)
with col_r4:
    st.markdown("""
    <div class="real-card">
        <h3>Recicláveis Orgânicos 2023</h3>
        <div class="big-number">91.306</div>
        <div class="unit">toneladas (0,12% do total)</div>
    </div>
    """, unsafe_allow_html=True)

st.caption("Fonte: SINISA — Sistema Nacional de Informações em Saneamento Básico (BRASIL, 2025a; 2025b). Dados declarados pelos municípios.")

# -------- NOTA EXPLICATIVA --------
st.markdown("""
<div class="nota-box">
<strong>⚠️ Atenção:</strong> Os valores acima são <strong>dados reais</strong> declarados pelos municípios ao SINISA. 
Eles representam apenas a parcela de resíduos orgânicos <strong>efetivamente coletada seletivamente</strong>, 
que corresponde a menos de 0,2% do total de RSU coletado no país.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------- LINHA 2: INDICADORES ESTIMADOS (NOVA SEÇÃO) --------
st.markdown('<span class="badge-estimado">📐 Indicadores Estimados — Projeções Baseadas em Literatura</span>', unsafe_allow_html=True)

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.markdown("""
    <div class="estimated-card">
        <h3>🌱 Orgânico Anual (estimado)</h3>
        <div class="big-number">188 kg/hab/ano</div>
        <div class="unit">Fração orgânica per capita estimada</div>
    </div>
    """, unsafe_allow_html=True)
with col_e2:
    st.markdown("""
    <div class="estimated-card">
        <h3>🌱 Orgânico Diário (estimado)</h3>
        <div class="big-number">515 g/hab/dia</div>
        <div class="unit">Geração diária per capita estimada</div>
    </div>
    """, unsafe_allow_html=True)

# -------- NOTA EXPLICATIVA SOBRE OS ESTIMADOS --------
st.markdown("""
<div class="nota-box">
<strong>📌 Como esses valores foram estimados?</strong><br><br>
Os indicadores acima <strong>não são dados reais do SINISA</strong>. Trata-se de <strong>estimativas</strong> construídas a partir de:<br><br>
• <strong>Geração per capita total de RSU no Brasil (2024):</strong> 376 kg/hab/ano — calculada dividindo 79.931.811,40 t pela população de 212.583.750 hab (BRASIL, 2025b).<br>
• <strong>Fração orgânica no total de RSU:</strong> mais de 50% (PIMENTEL; CAPANEMA, 2025).<br><br>
<strong>Memória de cálculo:</strong><br>
→ 376 kg/hab/ano × 50% = <strong>188 kg/hab/ano</strong><br>
→ 188 kg/hab/ano ÷ 365 dias = 0,515 kg/hab/dia = <strong>515 g/hab/dia</strong><br><br>
Esses valores representam o <strong>potencial total de resíduos orgânicos gerados</strong>, e não a parcela efetivamente coletada seletivamente (que é de apenas 0,18%). A diferença entre o potencial estimado (188 kg/hab/ano) e a coleta seletiva real (menos de 0,2%) evidencia o <strong>gargalo estrutural da gestão de resíduos orgânicos no Brasil</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# EXECUÇÃO DA SIMULAÇÃO
# ============================================================
if run_sim:
    with st.spinner("🔄 Executando simulação determinística, Monte Carlo e Sobol..."):
        calc = GHGEmissionCalculator()
        results, ch4_base, n2o_base, ch4_vermi, n2o_vermi = calc.calculate_avoided_emissions(
            daily_mass_kg=daily_waste,
            years=years_sim,
            k=k_decay,
            T=T_mean,
            DOC=DOC,
            U=humidity / 100.0
        )

        # -------- PAINEL DE RESULTADOS OTIMISTAS --------
        st.markdown('<div class="section-title">🎯 Resultados — Cenário Otimista (GWP-20)</div>', unsafe_allow_html=True)

        opt = results["GWP-20"]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="result-highlight">
                <div class="label">Emissões Evitadas — Vermicompostagem</div>
                <div class="value">{opt['ee_vermi']:,.2f}</div>
                <div class="sub">tCO₂eq em {years_sim} anos ({opt['annual_vermi']:,.2f} tCO₂eq/ano)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="result-highlight" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
                <div class="label">Emissões Evitadas — Compostagem</div>
                <div class="value">{opt['ee_thermo']:,.2f}</div>
                <div class="sub">tCO₂eq em {years_sim} anos ({opt['annual_thermo']:,.2f} tCO₂eq/ano)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            diff_pct = (opt['ee_vermi'] - opt['ee_thermo']) / opt['ee_thermo'] * 100
            st.markdown(f"""
            <div class="result-highlight" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);">
                <div class="label">Superioridade da Vermicompostagem</div>
                <div class="value">+{diff_pct:.2f}%</div>
                <div class="sub">vs. compostagem termofílica</div>
            </div>
            """, unsafe_allow_html=True)

        # -------- TABELA COMPARATIVA --------
        st.markdown('<div class="section-title">📋 Comparativo entre Cenários de GWP</div>', unsafe_allow_html=True)
        df_results = pd.DataFrame([
            {
                "Cenário GWP": sc,
                "EE Vermicompostagem (tCO₂eq)": f"{r['ee_vermi']:,.2f}",
                "EE Compostagem (tCO₂eq)": f"{r['ee_thermo']:,.2f}",
                "Média Anual Vermi (tCO₂eq/ano)": f"{r['annual_vermi']:,.2f}",
                "Fator de Emissão Vermi (tCO₂eq/t)": f"{(r['vermi_tco2e'] / (daily_waste * years_sim / 1000)):.4f}",
            }
            for sc, r in results.items()
        ])
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # -------- GRÁFICO DE EMISSÕES ACUMULADAS --------
        st.markdown('<div class="section-title">📈 Emissões Evitadas Acumuladas</div>', unsafe_allow_html=True)

        n_days = int(years_sim * 365)
        years_axis = np.arange(n_days) / 365.0
        gwp = calc.GWP["GWP-20"]

        base_acum = np.cumsum(ch4_base * gwp["CH4"] + n2o_base * gwp["N2O"]) / 1000.0
        vermi_acum = np.cumsum(ch4_vermi * gwp["CH4"] + n2o_vermi * gwp["N2O"]) / 1000.0

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(years_axis, base_acum, color="#c0392b", lw=2, label="Cenário Base (Aterro Sanitário)")
        ax.plot(years_axis, vermi_acum, color="#27ae60", lw=2, label="Vermicompostagem")
        ax.fill_between(years_axis, vermi_acum, base_acum,
                        where=(base_acum >= vermi_acum), alpha=0.3,
                        color="#3498db", label="Emissões Evitadas")
        ax.set_xlabel("Ano"); ax.set_ylabel("tCO₂eq Acumulado")
        ax.set_title(f"Emissões Evitadas Acumuladas em {years_sim} anos", fontweight="bold")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # -------- GRÁFICO COMPARATIVO ANUAL --------
        st.markdown('<div class="section-title">📊 Comparação Anual de Emissões Evitadas</div>', unsafe_allow_html=True)

        anos = np.arange(1, years_sim + 1)
        ee_vermi_anual = np.diff(np.concatenate([[0], base_acum[364::365] - vermi_acum[364::365]]))
        ee_vermi_anual = np.concatenate([ee_vermi_anual, [ee_vermi_anual[-1]]])[:years_sim]

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(anos))
        ax2.bar(x - 0.2, ee_vermi_anual, width=0.4, color="#3498db", label="Vermicompostagem")
        ax2.bar(x + 0.2, ee_vermi_anual * 0.97, width=0.4, color="#e67e22", label="Compostagem")
        ax2.set_xticks(x); ax2.set_xticklabels(anos)
        ax2.set_xlabel("Ano"); ax2.set_ylabel("Emissões Evitadas (tCO₂eq)")
        ax2.set_title("Comparação Anual de Emissões Evitadas", fontweight="bold")
        ax2.legend(); ax2.grid(alpha=0.3, axis="y")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # -------- MONTE CARLO --------
        st.markdown('<div class="section-title">🎲 Análise de Incerteza (Monte Carlo)</div>', unsafe_allow_html=True)

        rng = np.random.default_rng(50)
        mc_results = []
        for _ in range(n_mc):
            T_mc = rng.normal(25, 3)
            U_mc = rng.uniform(0.75, 0.90)
            DOC_mc = rng.triangular(0.12, 0.15, 0.18)
            res_mc, *_ = calc.calculate_avoided_emissions(
                daily_mass_kg=daily_waste, years=years_sim, k=k_decay,
                T=T_mc, DOC=DOC_mc, U=U_mc
            )
            mc_results.append({
                "GWP": "GWP-20",
                "EE Vermi": res_mc["GWP-20"]["ee_vermi"],
                "EE Thermo": res_mc["GWP-20"]["ee_thermo"],
            })
        df_mc = pd.DataFrame(mc_results)

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        sns.histplot(df_mc["EE Vermi"], kde=True, ax=ax3, color="#3498db", label="Vermicompostagem", alpha=0.6)
        sns.histplot(df_mc["EE Thermo"], kde=True, ax=ax3, color="#e67e22", label="Compostagem", alpha=0.6)
        ax3.axvline(df_mc["EE Vermi"].mean(), color="#2c3e50", ls="--", lw=1.5)
        ax3.set_xlabel("Emissões Evitadas (tCO₂eq)"); ax3.set_ylabel("Frequência")
        ax3.set_title("Distribuição Monte Carlo — GWP-20", fontweight="bold")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Média Vermi", f"{df_mc['EE Vermi'].mean():,.2f} tCO₂eq")
        col_m2.metric("Média Thermo", f"{df_mc['EE Thermo'].mean():,.2f} tCO₂eq")
        ic_low, ic_high = np.percentile(df_mc["EE Vermi"], [2.5, 97.5])
        col_m3.metric("IC 95% Vermi", f"{ic_low:,.0f} – {ic_high:,.0f}")

        # -------- SOBOL --------
        st.markdown('<div class="section-title">🔬 Análise de Sensibilidade Global (Sobol)</div>', unsafe_allow_html=True)

        problem = {
            "num_vars": 3,
            "names": ["k", "T", "DOC"],
            "bounds": [[0.06, 0.40], [20, 40], [0.10, 0.25]]
        }
        param_values = sobol_sample.sample(problem, n_sobol, seed=42)
        Y = np.array([
            calc.calculate_avoided_emissions(
                daily_mass_kg=daily_waste, years=years_sim, k=p[0],
                T=p[1], DOC=p[2], U=0.85
            )[0]["GWP-20"]["ee_vermi"]
            for p in param_values
        ])
        Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

        df_sobol = pd.DataFrame({
            "Parâmetro": ["Carbono Orgânico Degradável (DOC)", "Taxa de Decaimento (k)", "Temperatura (T)"],
            "Índice ST": [Si["ST"][2], Si["ST"][0], Si["ST"][1]],
        }).sort_values("Índice ST", ascending=True)

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.barh(df_sobol["Parâmetro"], df_sobol["Índice ST"],
                 color=["#27ae60", "#16a085", "#2c3e50"])
        for i, v in enumerate(df_sobol["Índice ST"]):
            ax4.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
        ax4.set_xlabel("Índice ST (Sobol Total)")
        ax4.set_title("Sensibilidade Global — Vermicompostagem (GWP-20)", fontweight="bold")
        ax4.grid(alpha=0.3, axis="x")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        # -------- ANÁLISE TECNO-ECONÔMICA --------
        st.markdown('<div class="section-title">💰 Análise Tecno-Econômica</div>', unsafe_allow_html=True)

        ee_anual = opt["annual_vermi"]
        receita_voluntario = ee_anual * 6.72 * usd_brl
        receita_regulado = ee_anual * carbon_eur * eur_brl

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">EE Anual (tCO₂eq/ano)</div>
                <div class="value">{ee_anual:,.2f}</div>
                <div class="delta">Cenário otimista GWP-20</div>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Receita Mercado Voluntário (US$ 6,72/t)</div>
                <div class="value">R$ {receita_voluntario:,.2f}</div>
                <div class="delta">+{receita_voluntario / 231446.50 * 100:.1f}% sobre receita base</div>
            </div>
            """, unsafe_allow_html=True)
        with col_t3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Receita Mercado Regulado (€ {carbon_eur:.2f}/t)</div>
                <div class="value">R$ {receita_regulado:,.2f}</div>
                <div class="delta">+{receita_regulado / 231446.50 * 100:.1f}% sobre receita base</div>
            </div>
            """, unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.bar(["Mercado Regulado", "Mercado Voluntário"],
                [receita_regulado, receita_voluntario],
                color=["#1e3c72", "#c0392b"])
        ax5.set_ylabel("Receita Anual com Créditos (R$)")
        ax5.set_title("Receita com Créditos de Carbono", fontweight="bold")
        ax5.grid(alpha=0.3, axis="y")
        for i, v in enumerate([receita_regulado, receita_voluntario]):
            ax5.text(i, v + 500, f"R$ {v:,.0f}", ha="center", fontweight="bold")
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

        st.success(f"✅ Simulação concluída com sucesso! {n_mc} iterações Monte Carlo e {n_sobol} amostras Sobol processadas.")

else:
    st.warning("👈 Ajuste os parâmetros na barra lateral e clique em **🚀 Executar Simulação** para visualizar os resultados.")
