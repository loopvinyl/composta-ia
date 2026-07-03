# 🌱 Composta.IA

**Inteligência Artificial para a Gestão Inteligente de Resíduos Sólidos Urbanos**

[![Licença: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 🎯 Objetivo

O **Composta.IA** é uma ferramenta web que utiliza **Inteligência Artificial (Processamento de Linguagem Natural e Machine Learning)** para transformar os dados brutos do **Sistema Nacional de Informações sobre Saneamento (SNIS)** em inteligência estratégica para gestores públicos.

Com ele, é possível:
- 📊 **Padronizar** automaticamente os destinos dos resíduos (aterro, compostagem, reciclagem) mesmo com erros de digitação ou siglas nos relatórios.
- ♻️ **Calcular** o potencial de compostagem e vermicompostagem por município.
- 💰 **Precificar** o potencial de geração de créditos de carbono (cenário otimista GWP-20), utilizando cotações em tempo real.
- 📈 **Identificar** municípios com perfis semelhantes (clusterização) para direcionar políticas públicas.
- 🔮 **Projetar** a geração futura de resíduos, auxiliando no planejamento de aterros e usinas.

---

## 🧠 Por que usar Inteligência Artificial?

O SNIS possui mais de **100 variações textuais** para descrever o mesmo destino (ex: "Aterro Sanitário", "AS", "Aterro Sani.", "Aterro – Gerenciado"). As regras tradicionais (`if/else`) falham nesses casos. 

O **Composta.IA** utiliza um modelo de **Regressão Logística com TF-IDF** treinado com milhares de registros históricos para:
- **Generalizar** padrões textuais com alta acurácia (>95%).
- **Atuar com transparência** – o modelo exibe o nível de confiança de cada classificação.
- **Ter supervisão humana** – quando a confiança é baixa, a ferramenta recai para as regras tradicionais (fallback seguro).

---

## ⚙️ Como funciona a análise?

1. **Coleta de Dados**: O app baixa automaticamente as planilhas mais recentes do SNIS (2023/2024) disponíveis no repositório oficial do governo.
2. **Classificação Inteligente**: A IA padroniza os campos "Tipo de Coleta" e "Destinação Final".
3. **Cálculo de Emissões**: Utiliza a metodologia do **IPCC (2006)** e modelos atualizados (Wang et al., 2017; Yang et al., 2017) para calcular:
   - Emissões de CH₄ e N₂O no aterro (projeção de 20 anos com lotes diários).
   - Emissões de CH₄ e N₂O na vermicompostagem (perfis diários de 50 dias).
4. **Mercado de Carbono**: Conecta-se a APIs de cotação (Yahoo Finance e AwesomeAPI) para converter emissões evitadas em receita potencial (R$/ano).
5. **Dashboard Interativo**: Visualize os dados por município, estado ou Brasil inteiro, com opção de ocultar transbordos.

---

## 🚀 Tecnologias Utilizadas

| Camada | Tecnologia |
| :--- | :--- |
| **Front-end** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | Scikit-learn (TF-IDF, Logistic Regression, K-Means) |
| **Processamento de Dados** | Pandas, NumPy, SciPy |
| **Visualização** | Matplotlib, Plotly |
| **APIs Externas** | Yahoo Finance (carbono), AwesomeAPI (câmbio) |
| **Linguagem** | Python 3.9+ |

---

## 📁 Estrutura do Projeto

