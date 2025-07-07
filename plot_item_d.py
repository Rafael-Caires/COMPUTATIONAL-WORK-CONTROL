import numpy as np
import matplotlib.pyplot as plt
from controle import analisar_estabilidade

def plotar_resultados_degrau():
    """
    Carrega os dados da simulação de degrau e gera os gráficos comparativos.
    """
    # Carregar dados do arquivo
    data = np.load("simulacao_degrau.npz", allow_pickle=True)
    sim_pequeno = data['pequeno'].item()
    sim_grande = data['grande'].item()

    # --- Gráfico 1: Comparação para Degrau Pequeno ---
    plt.figure(figsize=(10, 6))
    plt.plot(sim_pequeno['tempo'], sim_pequeno['y_lin'], 'r-', label='Saída Linearizada')
    plt.plot(sim_pequeno['tempo'], sim_pequeno['y_nl'], 'b--', label='Saída Não Linear')
    plt.title(f"Comparação das Saídas para Degrau de Amplitude {sim_pequeno['u_const']}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Saída y(k)")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparacao_degrau_pequeno.png", dpi=300)
    print("Gráfico 'comparacao_degrau_pequeno.png' salvo.")

    # --- Gráfico 2: Comparação para Degrau Grande ---
    plt.figure(figsize=(10, 6))
    plt.plot(sim_grande['tempo'], sim_grande['y_lin'], 'r-', label='Saída Linearizada')
    plt.plot(sim_grande['tempo'], sim_grande['y_nl'], 'b--', label='Saída Não Linear')
    plt.title(f"Comparação das Saídas para Degrau de Amplitude {sim_grande['u_const']}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Saída y(k)")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparacao_degrau_grande.png", dpi=300)
    print("Gráfico 'comparacao_degrau_grande.png' salvo.")

if __name__ == '__main__':
    plotar_resultados_degrau()