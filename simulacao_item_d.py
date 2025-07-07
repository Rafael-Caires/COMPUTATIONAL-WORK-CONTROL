import numpy as np
from controle import sistema_nao_linear, saida_nao_linear, sistema_linearizado, saida_linearizada

def simular_resposta_degrau(u_const, Ts, tempo_total):
    """
    Simula e compara a resposta a um degrau do sistema não-linear e linearizado.
    Inicia-se da condição de equilíbrio x_eq = [0, 0].
    
    Args:
        u_const: Amplitude do degrau de entrada.
        Ts: Período de amostragem.
        tempo_total: Duração da simulação.
        
    Returns:
        Dicionário contendo os resultados da simulação.
    """
    num_passos = int(tempo_total / Ts)
    tempo = np.linspace(0, tempo_total, num_passos)

    # Inicialização dos vetores de histórico
    x_nl_hist = np.zeros((num_passos, 2))
    y_nl_hist = np.zeros(num_passos)
    x_lin_hist = np.zeros((num_passos, 2))
    y_lin_hist = np.zeros(num_passos)

    # Condição inicial é o ponto de equilíbrio x(0) = [0, 0]
    # As saídas em t=0 também são zero, pois u só é aplicado em k>0
    
    # Simulação passo a passo
    for i in range(1, num_passos):
        # --- Sistema Não-Linear ---
        # O estado em k é usado para calcular o estado em k+1
        x_nl_atual = sistema_nao_linear(x_nl_hist[i-1], u_const, Ts)
        x_nl_hist[i] = x_nl_atual
        y_nl_hist[i] = saida_nao_linear(x_nl_atual, u_const)

        # --- Sistema Linearizado ---
        x_lin_atual = sistema_linearizado(x_lin_hist[i-1], u_const, Ts)
        x_lin_hist[i] = x_lin_atual
        y_lin_hist[i] = saida_linearizada(x_lin_atual, u_const)
        
    return {
        "tempo": tempo,
        "y_nl": y_nl_hist,
        "y_lin": y_lin_hist,
        "u_const": u_const
    }

if __name__ == "__main__":
    # --- Parâmetros da Simulação ---
    Ts = 0.0098  # Período de amostragem do seu sistema
    tempo_total = 4.0 # Duração da simulação em segundos

    # --- Simulação 1: Degrau Pequeno (δu = 0.01) ---
    print("Iniciando Simulação 1 (Degrau Pequeno)...")
    sim_pequeno = simular_resposta_degrau(u_const=0.01, Ts=Ts, tempo_total=tempo_total)
    
    # --- Simulação 2: Degrau Grande (δu = 2) ---
    print("Iniciando Simulação 2 (Degrau Grande)...")
    sim_grande = simular_resposta_degrau(u_const=2, Ts=Ts, tempo_total=tempo_total)

    # --- Salvar resultados em um único arquivo .npz ---
    np.savez("simulacao_degrau.npz", 
             pequeno=sim_pequeno, 
             grande=sim_grande)

    print("\nSimulações concluídas e dados salvos em 'simulacao_degrau.npz'")