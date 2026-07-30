with tab_diagnostico:
    st.header("🔥 Diagnóstico de Emissões de Metano (Baseline)")
    st.markdown("""
    Esta análise revela **quanto cada município emite com base nos dados mais recentes do SNIS** (ano selecionado), 
    considerando **três fatores determinantes**:
    
    1. **Quantidade de resíduos** enviada a aterros (massa real declarada);
    2. **Mix de resíduos** (composição orgânica, representada pelo DOC e taxa de decaimento k);
    3. **Destino final e gestão** (MCF – diferencia aterros sanitários, controlados e lixões).
    
    O cálculo segue a **metodologia UNFCCC A6.4-AMT-003 (modelo anual, Equação 1)**, projetando a geração de metano ao longo de **20 anos** a partir da massa de resíduos depositada no ano de referência. O valor exibido é a **média anual** desse total acumulado em 20 anos.
    
    **Use este diagnóstico para priorizar políticas públicas:** municípios com alta emissão e alta intensidade são os que mais se beneficiam com a implantação de compostagem ou melhoria da gestão de aterros.
    """)
    
    @st.cache_data
    def calcular_emissoes_brutas_por_municipio(df):
        resultados = []
        municipios = df['MUNICÍPIO'].unique()
        with st.spinner(f"🔄 Calculando emissões para {len(municipios)} municípios... (pode levar alguns segundos)"):
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
                if 'POPULACAO_TOTAL' in df_mun.columns:
                    pop = pd.to_numeric(df_mun['POPULACAO_TOTAL'].iloc[0], errors='coerce')
                else:
                    pop = 0
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
                    'MUNICÍPIO': mun,
                    'UF': uf,
                    'Massa_Aterro_Anual_t': massa_total_aterro,
                    'MCF_Medio': mcf_medio,
                    'DOC_Medio': doc_pond,
                    'k_Medio': k_pond,
                    'Emissao_Bruta_tCO2e_ano': emissao_anual,
                    'Intensidade_tCO2e_por_t': intensidade,
                    'Emissao_per_capita_kgCO2e': (emissao_anual * 1000) / pop if pop > 0 else 0,
                    'Gestao_Predominante': gestao_cat
                })
        return pd.DataFrame(resultados)

    with st.spinner("⏳ Processando dados de todos os municípios..."):
        df_emissoes = calcular_emissoes_brutas_por_municipio(df_clean)

    if df_emissoes.empty:
        st.warning("Nenhum município com resíduos enviados para aterro foi encontrado.")
    else:
        # =========================================================
        # NOVO: FILTRO POR LIMIARES SBCE (10k e 25k) - agora antes do filtro por estado
        # =========================================================
        st.markdown("---")
        st.subheader("⚖️ Municípios acima dos Limiares do SBCE (10.000 e 25.000 tCO₂e/ano)")
        st.markdown("""
        A Lei do SBCE (15.042/2024) estabelece:
        - **> 10.000 tCO₂e/ano**: Obrigação de MRV (Plano de Monitoramento e Relato).
        - **> 25.000 tCO₂e/ano**: Obrigação plena (MRV + entrega de Cotas Brasileiras de Emissão - CBEs).
        
        Abaixo estão listados **todos os municípios** que, com base nos dados atuais do SNIS, já ultrapassariam o limiar de 10.000 tCO₂e/ano,
        servindo como subsídio direto para a definição da **Etapa 2 (Resíduos)** do SBCE.
        """)
        
        # Garante que a coluna é numérica
        df_limiares = df_emissoes.copy()
        df_limiares['Emissao_Bruta_tCO2e_ano'] = pd.to_numeric(df_limiares['Emissao_Bruta_tCO2e_ano'], errors='coerce').fillna(0)
        
        df_acima_10k = df_limiares[df_limiares['Emissao_Bruta_tCO2e_ano'] > 10000].copy()
        df_acima_25k = df_limiares[df_limiares['Emissao_Bruta_tCO2e_ano'] > 25000].copy()
        
        col1, col2 = st.columns(2)
        col1.metric("🔹 Acima de 10.000 tCO₂e (MRV)", f"{len(df_acima_10k)} municípios")
        col2.metric("🔺 Acima de 25.000 tCO₂e (Obrigação Plena)", f"{len(df_acima_25k)} municípios")
        
        # --- Tabela completa de municípios acima de 10k ---
        st.markdown("#### 📋 Todos os municípios com emissão > 10.000 tCO₂e/ano")
        if not df_acima_10k.empty:
            df_exibicao_10k = df_acima_10k[['MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Emissao_Bruta_tCO2e_ano', 'Massa_Aterro_Anual_t']]
            df_exibicao_10k = df_exibicao_10k.sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
            
            st.dataframe(
                df_exibicao_10k.style.format({
                    'Emissao_Bruta_tCO2e_ano': lambda x: f"{x:,.0f}".replace(",", "."),
                    'Massa_Aterro_Anual_t': lambda x: f"{x:,.0f}".replace(",", ".")
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.info("ℹ️ Nenhum município ultrapassa 10.000 tCO₂e/ano.")
        
        # --- Destaque para os acima de 25k (prioritários) ---
        st.markdown("#### 🔺 Destaque: municípios acima de 25.000 tCO₂e/ano (obrigação plena)")
        if not df_acima_25k.empty:
            df_exibicao_25k = df_acima_25k[['MUNICÍPIO', 'UF', 'Gestao_Predominante', 'Emissao_Bruta_tCO2e_ano', 'Massa_Aterro_Anual_t']]
            df_exibicao_25k = df_exibicao_25k.sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
            
            st.dataframe(
                df_exibicao_25k.style.format({
                    'Emissao_Bruta_tCO2e_ano': lambda x: f"{x:,.0f}".replace(",", "."),
                    'Massa_Aterro_Anual_t': lambda x: f"{x:,.0f}".replace(",", ".")
                }),
                use_container_width=True,
                height=300
            )
            st.caption(f"📌 Total de {len(df_acima_25k)} municípios que, se estivessem no SBCE hoje, já teriam que entregar CBEs.")
        else:
            st.info("ℹ️ Nenhum município ultrapassa 25.000 tCO₂e/ano.")
        
        # --- Filtro por estado (agora após a seção de limiares) ---
        st.markdown("---")
        estados = sorted(df_emissoes['UF'].unique())
        estado_selecionado = st.selectbox("Filtrar por Estado:", ["Todos"] + estados)
        if estado_selecionado != "Todos":
            df_filtrado = df_emissoes[df_emissoes['UF'] == estado_selecionado]
        else:
            df_filtrado = df_emissoes

        # --- Métricas gerais (após filtro) ---
        total_emissoes = df_filtrado['Emissao_Bruta_tCO2e_ano'].sum()
        total_massa = df_filtrado['Massa_Aterro_Anual_t'].sum()
        media_intensidade = df_filtrado['Intensidade_tCO2e_por_t'].mean()
        num_lixoes = df_filtrado[df_filtrado['Gestao_Predominante'] == 'Lixão/Precário'].shape[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌍 Emissão Média Anual (20 anos)", f"{formatar_br(total_emissoes, auto_precision=False, casas_override=0)} tCO₂e")
        col2.metric("⚖️ Massa em Aterro", f"{formatar_br(total_massa, auto_precision=False, casas_override=0)} t")
        col3.metric("📊 Intensidade Média", f"{formatar_br(media_intensidade, auto_precision=False, casas_override=2)} tCO₂e/t")
        col4.metric("⚠️ Municípios com Lixão", num_lixoes)

        # --- Gráfico Pareto das emissões ---
        st.markdown("---")
        st.markdown("#### 📉 Curva de Concentração das Emissões de Metano (Pareto)")
        st.markdown("""
        **Como ler:** A linha azul mostra o percentual acumulado das emissões totais de metano (tCO₂e/ano) em função do percentual acumulado de municípios (ordenados do maior para o menor emissor). 
        Quanto mais a curva se inclina para a esquerda, maior é a concentração. 
        O ponto onde a linha cruza os 80% no eixo Y indica quantos % dos municípios são responsáveis por 80% de todas as emissões de metano do Brasil.
        """)
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
        ax_emissoes.annotate(f'{pct_municipios_80_emissoes:.1f}% dos municípios\nconcentram 80% das emissões', xy=(pct_municipios_80_emissoes, 80), xytext=(pct_municipios_80_emissoes + 15, 60), arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=11, color='red', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
        ax_emissoes.annotate(f'{pct_municipios_50_emissoes:.1f}% dos municípios\nconcentram 50% das emissões', xy=(pct_municipios_50_emissoes, 50), xytext=(pct_municipios_50_emissoes + 15, 35), arrowprops=dict(arrowstyle='->', color='orange', lw=1.5), fontsize=10, color='orange', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.9))
        ax_emissoes.plot([0, 100], [0, 100], color='gray', linestyle=':', alpha=0.5, label='Igualdade perfeita (referência)')
        ax_emissoes.set_xlabel('Percentual acumulado de municípios (%)', fontsize=12)
        ax_emissoes.set_ylabel('Percentual acumulado das emissões (%)', fontsize=12)
        ax_emissoes.set_title(f'Concentração das Emissões de Metano – Brasil ({ano_selecionado})', fontsize=14)
        ax_emissoes.grid(True, linestyle=':', alpha=0.4)
        ax_emissoes.legend(loc='lower right')
        ax_emissoes.set_xlim(0, 100)
        ax_emissoes.set_ylim(0, 100)
        ax_emissoes.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        ax_emissoes.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        plt.tight_layout()
        st.pyplot(fig_emissoes)
        plt.close(fig_emissoes)
        st.caption(f"""
        📌 **Interpretação:** A curva demonstra que os **{formatar_br(pct_municipios_80_emissoes, auto_precision=False, casas_override=1)}% maiores emissores** concentram **80% de todas as emissões de metano do Brasil**.
        Comparando com a concentração da massa, este número pode ser maior ou menor, dependendo do MCF e da composição dos resíduos (DOC/k) de cada município.
        """)

        # --- Gráfico Top 20 emissores absolutos ---
        st.markdown("---")
        st.subheader("🏆 Top 20 Municípios que mais Emitem Metano (emissão absoluta)")
        top20 = df_filtrado.nlargest(20, 'Emissao_Bruta_tCO2e_ano')
        top20 = top20.sort_values('Emissao_Bruta_tCO2e_ano', ascending=False)
        cor_map = {'Sanitário': '#2ecc71', 'Controlado': '#f39c12', 'Lixão/Precário': '#e74c3c'}
        top20['Cor'] = top20['Gestao_Predominante'].map(cor_map)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(top20['MUNICÍPIO'] + " (" + top20['UF'] + ")", top20['Emissao_Bruta_tCO2e_ano'], color=top20['Cor'])
        ax.set_xlabel('Emissão Média Anual (tCO₂e / ano)')
        ax.set_title('Ranking de Emissões de Metano por Município')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatar_br(x, auto_precision=False, casas_override=2)))
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='Aterro Sanitário (MCF≥0.8)'), Patch(facecolor='#f39c12', label='Aterro Controlado (MCF 0.4-0.8)'), Patch(facecolor='#e74c3c', label='Lixão/Precário (MCF<0.4)')]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("🔴 Vermelho = Lixões ou aterros precários | 🟡 Amarelo = Controlado | 🟢 Verde = Sanitário (bem gerenciado)")

        # --- Gráfico Top 20 emissores per capita ---
        st.markdown("---")
        st.subheader("🏆 Top 20 Municípios com Maior Emissão de Metano por Habitante")
        st.markdown("""
        **Este ranking mostra a emissão de metano por habitante (kgCO₂e/hab/ano).**  
        Municípios com alta emissão per capita geralmente têm:
        - **Grande volume de resíduos** em relação à população (geração excessiva);
        - **Destinação inadequada** (lixões ou aterros controlados, com MCF baixo);
        - **Composição orgânica elevada** (alta fração de alimentos e podas).
        
        **Interpretação:** Uma cidade pequena pode aparecer no topo se sua gestão de resíduos for ineficiente. 
        Já grandes cidades podem ter emissão per capita baixa se tiverem aterros sanitários bem gerenciados 
        (MCF alto, captura de biogás). Este indicador ajuda a identificar **municípios onde a gestão per capita é crítica**,
        independentemente do tamanho populacional.
        """)
        df_percapita = df_filtrado[df_filtrado['Emissao_per_capita_kgCO2e'] > 0].copy()
        if df_percapita.empty:
            st.info("ℹ️ Não há dados de população disponível para calcular a emissão per capita.")
        else:
            top20_percapita = df_percapita.nlargest(20, 'Emissao_per_capita_kgCO2e')
            top20_percapita = top20_percapita.sort_values('Emissao_per_capita_kgCO2e', ascending=False)
            top20_percapita['Cor'] = top20_percapita['Gestao_Predominante'].map(cor_map)
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            ax2.barh(top20_percapita['MUNICÍPIO'] + " (" + top20_percapita['UF'] + ")", top20_percapita['Emissao_per_capita_kgCO2e'], color=top20_percapita['Cor'])
            ax2.set_xlabel('Emissão per capita (kgCO₂e / habitante / ano)')
            ax2.set_title('Ranking de Emissões de Metano por Habitante')
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatar_br(x, auto_precision=False, casas_override=2)))
            ax2.legend(handles=legend_elements, loc='lower right')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption("🔴 Vermelho = Lixões ou aterros precários | 🟡 Amarelo = Controlado | 🟢 Verde = Sanitário (bem gerenciado)")

        # --- Matriz de decisão ---
        st.markdown("---")
        st.subheader("📊 Matriz de Decisão: Massa x Intensidade")
        st.markdown("""
        **Como interpretar:**
        - **🚨 CRÍTICO (Alta Massa + Alta Intensidade)**: Prioridade máxima para intervenção.
        - **⚠️ INEFICIENTE (Baixa Massa + Alta Intensidade)**: Pequenos lixões que precisam ser fechados.
        - **✅ REFERÊNCIA (Alta Massa + Baixa Intensidade)**: Grandes cidades com gestão adequada.
        - **📉 BAIXA PRIORIDADE (Baixa Massa + Baixa Intensidade)**: Pequenas cidades com gestão razoável.
        """)
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
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.3)
        ax3.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_abreviado))
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # --- Tabela detalhada (original) ---
        st.markdown("---")
        st.subheader("📋 Detalhamento por Município (Clique no cabeçalho para ordenar)")
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
            'MUNICÍPIO': 'Município',
            'UF': 'UF',
            'Gestao_Predominante': 'Gestão',
            'Massa_Aterro_Anual_t': 'Massa (t/ano)',
            'MCF_Medio': 'MCF médio',
            'DOC_Medio': 'DOC médio',
            'Intensidade_tCO2e_por_t': 'Intensidade (tCO₂e/t)',
            'Emissao_Bruta_tCO2e_ano': 'Emissão Média Anual (tCO₂e/ano)',
            'Emissao_per_capita_kgCO2e': 'Emissão per capita (kgCO₂e)'
        })
        st.dataframe(tabela_diagnostico, use_container_width=True, height=500)
        st.markdown("---")
        st.caption("""
        **Metodologia:** UNFCCC A6.4-AMT-003 (Application B) – Baseline de aterro.  
        - **Emissão Média Anual**: média aritmética do total de emissões de metano (CH₄) projetado para os 20 anos seguintes ao depósito do resíduo do ano de referência (modelo anual, Equação 1).  
        - **Emissão per capita**: emissão média anual dividida pela população do município (kgCO₂e/hab/ano).  
        - **Intensidade**: emissão média anual por tonelada de resíduo depositado. Quanto menor, melhor a gestão do aterro.  
        - **MCF**: 1,0 (Sanitário), 0,4-0,8 (Controlado), <0,4 (Lixão/Precário) – conforme Tabela 8 da norma.
        - DOC/k calculados dinamicamente pela caracterização do resíduo no SNIS (colunas GTR1501 a GTR1507).
        """)
