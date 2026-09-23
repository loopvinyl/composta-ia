"""
Composta.IA — Ponto de entrada.
================================
Este arquivo controla se o app está EM BREVE ou LIBERADO.

Para liberar o aplicativo completo, faça UMA das opções:
  1) Altere MOSTRAR_EM_BREVE_LOCAL para False (abaixo);
  2) Ou defina nos Secrets do Streamlit Cloud:
         MOSTRAR_EM_BREVE = false
  3) Ou crie .streamlit/secrets.toml com:
         MOSTRAR_EM_BREVE = false
"""
import textwrap
import streamlit as st

# =========================================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Composta.IA — Em breve de volta ao ar",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 🔒 CHAVE MESTRA
# =========================================================
MOSTRAR_EM_BREVE_LOCAL = True  # <-- troque para False para LIBERAR o app

try:
    MOSTRAR_EM_BREVE = bool(st.secrets.get("MOSTRAR_EM_BREVE", MOSTRAR_EM_BREVE_LOCAL))
except Exception:
    MOSTRAR_EM_BREVE = MOSTRAR_EM_BREVE_LOCAL

# =========================================================
# 🚧 HTML + CSS DA PÁGINA "EM BREVE"
# =========================================================
PAGINA_EM_BREVE = textwrap.dedent("""
<style>
.stApp { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 55%, #f0fdf4 100%); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.pagina-container {
    max-width: 980px; margin: 2.5rem auto 2rem auto; padding: 2.75rem 3rem 2.25rem 3rem;
    background-color: #ffffff; border-radius: 24px;
    box-shadow: 0 15px 50px rgba(22, 101, 52, 0.14);
    border: 1px solid #bbf7d0;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #1f2937;
}
.pagina-header { text-align: center; margin-bottom: 1.75rem; }
.pagina-emoji { font-size: 4rem; line-height: 1; display: inline-block;
    animation: pulse 2.6s ease-in-out infinite; }
@keyframes pulse { 0%,100% { transform: scale(1); opacity:1; }
                   50% { transform: scale(1.10); opacity:0.88; } }
.pagina-titulo { color:#14532d; font-size:2.3rem; font-weight:800;
    margin:0.25rem 0 0.4rem 0; letter-spacing:-0.6px; }
.pagina-subtitulo { color:#15803d; font-size:1.15rem; font-weight:600; margin:0; }
.status-faixa {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 1px solid #86efac; border-radius: 14px;
    padding: 1rem 1.5rem; margin: 1.5rem 0 1.75rem 0;
    display: flex; align-items: center; gap: 1rem; text-align: left;
}
.status-icone { font-size: 2rem; line-height: 1; flex-shrink: 0; }
.status-texto { color:#14532d; font-size:1rem; line-height:1.55; margin:0; }
.status-texto strong { color:#166534; }
.secao-titulo {
    color:#14532d; font-size:1.3rem; font-weight:700;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #bbf7d0;
    display: flex; align-items: center; gap: 0.6rem;
}
.cards-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem; margin-bottom: 1.5rem;
}
.card {
    background:#f9fafb; border:1px solid #e5e7eb; border-left:5px solid #22c55e;
    border-radius: 12px; padding: 1.1rem 1.2rem; text-align: left;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.card:hover { transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(22, 101, 52, 0.10); }
.card-icone { font-size: 1.6rem; margin-bottom: 0.4rem; display: block; }
.card-titulo { color:#14532d; font-size:1.02rem; font-weight:700; margin:0 0 0.4rem 0; }
.card-texto { color:#4b5563; font-size:0.92rem; line-height:1.55; margin:0; }
.timeline { list-style: none; padding: 0; margin: 1rem 0 1.5rem 0; }
.timeline li { position: relative; padding: 0.75rem 0 0.75rem 2.25rem;
    border-left: 3px solid #bbf7d0; margin-left: 0.5rem; }
.timeline li:last-child { border-left-color: transparent; }
.timeline li .ponto { position: absolute; left:-9px; top:1rem;
    width:14px; height:14px; border-radius:50%; background:#22c55e;
    border:3px solid #ffffff; box-shadow: 0 0 0 2px #86efac; }
.timeline li.passado .ponto { background:#16a34a; }
.timeline li.atual   .ponto { background:#f59e0b; box-shadow: 0 0 0 2px #fde68a; }
.timeline li.futuro  .ponto { background:#e5e7eb; box-shadow: 0 0 0 2px #d1d5db; }
.timeline-titulo { color:#14532d; font-weight:700; font-size:1rem; margin:0 0 0.2rem 0; }
.timeline-texto { color:#4b5563; font-size:0.92rem; line-height:1.55; margin:0; }
.badges { text-align: center; margin: 1.5rem 0 0.5rem 0; }
.badge { display:inline-block; background-color:#dcfce7; color:#166534;
    padding:0.4rem 0.9rem; border-radius:999px; font-size:0.82rem;
    font-weight:600; margin:0.2rem; border:1px solid #86efac; }
.badge.destaque { background-color:#fef3c7; color:#92400e; border-color:#fde68a; }
.rodape { text-align:center; color:#6b7280; font-size:0.88rem; line-height:1.7;
    margin-top:1.5rem; padding-top:1.25rem; border-top: 1px dashed #bbf7d0; }
.rodape strong { color:#166534; }
.rodape em { color:#15803d; font-style: normal; font-weight: 600; }
</style>

<div class="pagina-container">
    <div class="pagina-header">
        <div class="pagina-emoji">🌱</div>
        <h1 class="pagina-titulo">Composta.IA</h1>
        <p class="pagina-subtitulo">Potencial de Compostagem e Créditos de Carbono para os municípios brasileiros</p>
    </div>

    <div class="status-faixa">
        <div class="status-icone">🚀</div>
        <p class="status-texto">
            O <strong>Composta.IA</strong> esteve no ar em <strong>fase de testes</strong>.
            Agora avança para a <strong>próxima etapa do projeto</strong> —
            <strong>voltaremos ao ar em breve com todas as funcionalidades liberadas.</strong>
        </p>
    </div>

    <h2 class="secao-titulo">🔎 O que o Composta.IA faz</h2>

    <div class="cards-grid">
        <div class="card">
            <span class="card-icone">📊</span>
            <p class="card-titulo">Leitura dos dados do SINISA</p>
            <p class="card-texto">Interpreta automaticamente os tipos de coleta e destinação declarados pelos municípios no SINISA (2023 e 2024).</p>
        </div>
        <div class="card">
            <span class="card-icone">🤖</span>
            <p class="card-titulo">IA que padroniza os dados</p>
            <p class="card-texto">Usa Processamento de Linguagem Natural (PLN) para reconhecer e padronizar as variações textuais dos destinos finais.</p>
        </div>
        <div class="card">
            <span class="card-icone">🌱</span>
            <p class="card-titulo">Potencial de compostagem</p>
            <p class="card-texto">Estima quanto do resíduo orgânico coletado pode ser desviado do aterro e transformado em composto, por município.</p>
        </div>
        <div class="card">
            <span class="card-icone">🔥</span>
            <p class="card-titulo">Diagnóstico de emissões (baseline)</p>
            <p class="card-texto">Calcula as emissões de metano (CH₄) dos aterros, lixões e aterros controlados, seguindo a norma <strong>UNFCCC A6.4-AMT-003</strong>.</p>
        </div>
        <div class="card">
            <span class="card-icone">💰</span>
            <p class="card-titulo">Créditos de carbono</p>
            <p class="card-texto">Estima o valor financeiro das emissões evitadas com a compostagem, usando cotação real de carbono e câmbio EUR/BRL.</p>
        </div>
        <div class="card">
            <span class="card-icone">📈</span>
            <p class="card-titulo">Cenários e projeções</p>
            <p class="card-texto">Projeta a expansão da coleta seletiva, simula receitas futuras e identifica municípios prioritários para políticas públicas.</p>
        </div>
        <div class="card">
            <span class="card-icone">🎯</span>
            <p class="card-titulo">Apoio ao SBCE e ao INPI</p>
            <p class="card-texto">Identifica municípios acima dos limiares de 10.000 e 25.000 tCO₂e/ano, subsidiando o Sistema Brasileiro de Comércio de Emissões e a Etapa 2 (Resíduos).</p>
        </div>
        <div class="card">
            <span class="card-icone">🇧🇷</span>
            <p class="card-titulo">Cobertura nacional</p>
            <p class="card-texto">Abrange os <strong>5.570 municípios brasileiros</strong>, com painéis comparativos por estado e por porte populacional.</p>
        </div>
    </div>

    <h2 class="secao-titulo">🗓️ Onde estamos</h2>

    <ul class="timeline">
        <li class="passado">
            <span class="ponto"></span>
            <p class="timeline-titulo">✅ Fase 1 — Desenvolvimento e testes</p>
            <p class="timeline-texto">O aplicativo foi publicado e permaneceu no ar em ambiente de testes, com validação dos cálculos, das bases do SINISA e da metodologia UNFCCC aplicada.</p>
        </li>
        <li class="atual">
            <span class="ponto"></span>
            <p class="timeline-titulo">⏳ Fase 2 — Próxima etapa do projeto (em andamento)</p>
            <p class="timeline-texto">Consolidação metodológica, ajustes finais e encaminhamentos legais necessários para a liberação definitiva da plataforma.</p>
        </li>
        <li class="futuro">
            <span class="ponto"></span>
            <p class="timeline-titulo">🔜 Fase 3 — De volta ao ar</p>
            <p class="timeline-texto">Retorno do Composta.IA com <strong>todas as funcionalidades liberadas</strong>: análise completa dos municípios, diagnósticos, projeções, simulações de créditos de carbono e painéis comparativos nacionais.</p>
        </li>
    </ul>

    <div class="badges">
        <span class="badge">📊 SINISA 2023 / 2024</span>
        <span class="badge">🤖 IA + PLN</span>
        <span class="badge">🌍 UNFCCC A6.4-AMT-003</span>
        <span class="badge">💰 Créditos de Carbono</span>
        <span class="badge">🇧🇷 5.570 municípios</span>
        <span class="badge destaque">🚀 Em breve de volta ao ar</span>
    </div>

    <div class="rodape">
        Ferramenta de apoio à <strong>gestão pública de resíduos sólidos</strong>,
        desenvolvida para subsidiar o SINISA e políticas de economia circular.<br>
        Todo o conteúdo, código-fonte e metodologia são de titularidade do autor.<br><br>
        🌿 <em>Volte em breve — o Composta.IA estará novamente no ar com todas as funcionalidades.</em>
    </div>
</div>
""").strip()

# =========================================================
# 🚧 RENDERIZAÇÃO DA PÁGINA
# =========================================================
if MOSTRAR_EM_BREVE:
    # Preferência: st.html() (Streamlit >= 1.33) — renderiza HTML puro sem Markdown
    if hasattr(st, "html"):
        st.html(PAGINA_EM_BREVE)
    else:
        # Fallback para versões antigas do Streamlit
        st.markdown(PAGINA_EM_BREVE, unsafe_allow_html=True)

    st.stop()  # ⛔ Nada abaixo é executado enquanto estiver em "em breve"

# =========================================================
# ✅ APP COMPLETO (só carrega se estiver liberado)
# =========================================================
try:
    import composta_ia  # noqa: F401
except ModuleNotFoundError:
    st.error(
        "❌ Módulo `composta_ia.py` não encontrado. "
        "Certifique-se de que o arquivo com o código do aplicativo "
        "está na mesma pasta deste `app.py`."
    )
    st.stop()
