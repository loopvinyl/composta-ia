import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def projetar_residuos_per_capita(populacao_atual, massa_anual_atual, 
                                 taxa_crescimento_pop=0.01, anos=10):
    """
    Projeta a geração de resíduos com base no crescimento populacional.
    Assume que a geração per capita permanece constante.
    
    Parâmetros:
    - populacao_atual: habitantes
    - massa_anual_atual: toneladas/ano
    - taxa_crescimento_pop: ex: 0.01 = 1% ao ano
    - anos: número de anos para projetar
    
    Retorna: DataFrame com Ano, Populacao_Projetada, Massa_Projetada_ton
    """
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


def simular_cenarios_compostagem(massa_aterro_ano, 
                                 co2_evitado_por_tonelada, 
                                 preco_carbono_atual, 
                                 taxa_cambio,
                                 anos_projecao=10, 
                                 taxa_crescimento_compostagem=0.10,
                                 inflacao_carbono=0.02):
    """
    Simula o ganho financeiro ao aumentar gradualmente a compostagem.
    
    Parâmetros:
    - massa_aterro_ano: toneladas de orgânico que vão para aterro atualmente
    - co2_evitado_por_tonelada: tCO2e evitado por tonelada desviada (já calculado)
    - preco_carbono_atual: em Euro
    - taxa_cambio: EUR/BRL
    - anos_projecao: quantos anos simular
    - taxa_crescimento_compostagem: % de aumento anual no desvio (ex: 0.10 = 10%)
    - inflacao_carbono: % de aumento anual no preço do carbono (ex: 0.02 = 2%)
    
    Retorna: DataFrame com Ano, Massa_Desviada_Acumulada, Receita_Acumulada_BRL, Ganho_Adicional_BRL
    """
    if massa_aterro_ano <= 0:
        raise ValueError("Massa de aterro deve ser maior que zero.")
    
    resultados = []
    # Cenário estático (sem aumento) - mantém a massa atual
    massa_estatica = massa_aterro_ano
    
    for ano in range(1, anos_projecao + 1):
        # Cenário Projetado (com aumento)
        fator_desvio = (1 + taxa_crescimento_compostagem) ** (ano - 1)
        massa_projetada = massa_aterro_ano * fator_desvio
        
        # Atualiza preço do carbono (com inflação)
        preco_atualizado = preco_carbono_atual * (1 + inflacao_carbono) ** (ano - 1)
        
        # Emissões evitadas
        co2_evitado_estatico = massa_estatica * co2_evitado_por_tonelada
        co2_evitado_projetado = massa_projetada * co2_evitado_por_tonelada
        
        # Receita em Real (acumulada)
        receita_estatico_brl = co2_evitado_estatico * preco_atualizado * taxa_cambio
        receita_projetado_brl = co2_evitado_projetado * preco_atualizado * taxa_cambio
        
        # Ganho incremental (o quanto a política de aumento gera a mais)
        ganho_incremental = receita_projetado_brl - receita_estatico_brl
        
        resultados.append({
            'Ano': datetime.now().year + ano,
            'Massa_Desviada_Acumulada(t)': massa_projetada,
            'Receita_Anual_BRL': receita_projetado_brl,
            'Ganho_Adicional_BRL': ganho_incremental
        })
    
    df = pd.DataFrame(resultados)
    # Calcula o acumulado
    df['Receita_Acumulada_BRL'] = df['Receita_Anual_BRL'].cumsum()
    return df


def plot_projecao_residuos(df_proj):
    """Gera gráfico de duplo eixo: população e massa de resíduos."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('População (habitantes)', color='blue')
    ax1.plot(df_proj['Ano'], df_proj['Populacao_Projetada'], 'o-', color='blue', linewidth=2, label='População')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Massa de Resíduos (toneladas/ano)', color='green')
    ax2.plot(df_proj['Ano'], df_proj['Massa_Projetada_ton'], 's-', color='green', linewidth=2, label='Massa')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Anotações
    for i, row in df_proj.iterrows():
        ax1.annotate(f"{row['Populacao_Projetada']:,.0f}", 
                    (row['Ano'], row['Populacao_Projetada']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
        ax2.annotate(f"{row['Massa_Projetada_ton']:,.0f}", 
                    (row['Ano'], row['Massa_Projetada_ton']), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='green')
    
    plt.title('Projeção de População e Geração de Resíduos', fontsize=14)
    fig.tight_layout()
    return fig


def plot_simulacao_compostagem(df_sim):
    """Gera gráfico da receita acumulada com créditos de carbono."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_sim['Ano'], df_sim['Receita_Acumulada_BRL'], 'o-', color='green', linewidth=2, label='Receita Acumulada')
    ax.fill_between(df_sim['Ano'], 0, df_sim['Receita_Acumulada_BRL'], alpha=0.3, color='lightgreen')
    
    # Anotações
    for i, row in df_sim.iterrows():
        ax.annotate(f"R$ {row['Receita_Acumulada_BRL']:,.0f}", 
                    (row['Ano'], row['Receita_Acumulada_BRL']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('Receita Acumulada (R$)')
    ax.set_title('Projeção de Ganhos com Créditos de Carbono (Compostagem)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return fig
