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
        st.caption(f"📌 O cenário realista usa o 1º quartil ({pct_25:.2f}%) como meta, baseado nos 25% menores percentuais entre os que já possuem coleta seletiva.")
    else:
        st.info("Nenhum município com coleta seletiva para referência.")

    st.markdown("---")
    st.markdown("#### 🚫 Municípios com **0% de cobertura** (sem coleta seletiva de orgânicos)")
    if not df_sem_seletiva.empty:
        # Mostrar os 10 maiores em massa total (para destacar onde a expansão teria maior impacto)
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
