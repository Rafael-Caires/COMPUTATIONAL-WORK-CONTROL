import numpy as np
from controle import sistema_nao_linear, saida_nao_linear, sistema_linearizado, saida_linearizada

def simular_comparacao_degrau(u_const, Ts, tempo_total):
    """
    Simula e compara a resposta a um degrau do sistema não-linear e linearizado.
    Args:
        u_const: amplitude do degrau de entrada
        Ts: período de amostragem
        tempo_total: duração da simulação
    Returns:
        arrays de tempo, saídas para ambos os sistemas
    """
    num_passos = int(tempo_total / Ts)
    tempo = np.linspace(0, tempo_total, num_passos)

    # Inicialização dos arrays
    x_nl_hist = np.zeros((num_passos, 2))
    y_nl_hist = np.zeros(num_passos)
    x_lin_hist = np.zeros((num_passos, 2))
    y_lin_hist = np.zeros(num_passos)

    # Condição inicial é o ponto de equilíbrio x_eq = [0, 0]
    x_nl_atual = np.array([0.0, 0.0])
    x_lin_atual = np.array([0.0, 0.0])

    # Simulação passo a passo
    for i in range(1, num_passos):
        # Sistema Não-Linear
        x_nl_atual = sistema_nao_linear(x_nl_hist[i-1], u_const, Ts)
        y_nl_hist[i] = saida_nao_linear(x_nl_atual, u_const)
        x_nl_hist[i] = x_nl_atual

        # Sistema Linearizado
        x_lin_atual = sistema_linearizado(x_lin_hist[i-1], u_const, Ts)
        y_lin_hist[i] = saida_linearizada(x_lin_atual, u_const)
        x_lin_hist[i] = x_lin_atual
        
    return tempo, y_nl_hist, y_lin_hist

def simular_resposta_frequencia(A_in, omega, Ts, tempo_total):
    """
    Simula a resposta em frequência do sistema linearizado a uma entrada senoidal.
    """
    num_passos = int(tempo_total / Ts)
    tempo = np.linspace(0, tempo_total, num_passos)

    # Geração do sinal de entrada senoidal
    u_input = A_in * np.sin(omega * tempo)

    # Inicialização
    x_lin_hist = np.zeros((num_passos, 2))
    y_lin_hist = np.zeros(num_passos)
    x_lin_atual = np.array([0.0, 0.0]) # Parte do repouso

    # Simulação
    for i in range(num_passos):
        x_lin_atual = sistema_linearizado(x_lin_atual, u_input[i], Ts)
        y_lin_hist[i] = saida_linearizada(x_lin_atual, u_input[i])
        x_lin_hist[i] = x_lin_atual

    return tempo, u_input, y_lin_hist

def calcular_ganho_defasagem(tempo, u, y, omega):
    """
    Calcula o ganho e a defasagem em regime permanente.
    """
    # Usar os últimos 40% dos dados para garantir regime permanente
    inicio = int(0.6 * len(tempo))
    u_ss = u[inicio:]
    y_ss = y[inicio:]
    
    # Cálculo do ganho
    ganho_linear = np.max(np.abs(y_ss)) / np.max(np.abs(u_ss))
    
    # Cálculo da defasagem
    fase_u = np.arctan2(-u_ss[1], u_ss[0]) # Estima fase da entrada
    fase_y = np.arctan2(-y_ss[1], y_ss[0]) # Estima fase da saída
    defasagem_rad = fase_y - fase_u
    
    # Ajuste para o resultado ficar entre -pi e pi
    defasagem_rad = np.arctan2(np.sin(defasagem_rad), np.cos(defasagem_rad))
    defasagem_graus = np.rad2deg(defasagem_rad)
    
    return ganho_linear, defasagem_graus

# --- Execução das Simulações ---
if __name__ == "__main__":
    # Parâmetros da ficha técnica
    Ts = 0.0098
    
    # 1. Simulação Comparativa de Resposta ao Degrau (Item d)
    # Simulação para degrau de pequena amplitude (conforme item d.1)
    u_pequeno = 0.01
    tempo_total_degrau = 4  # Ajuste conforme necessário
    tempo_comp, y_nl_comp, y_lin_comp = simular_comparacao_degrau(u_pequeno, Ts, tempo_total_degrau)
    
    np.savez("simulacao_comparacao.npz", 
             tempo=tempo_comp, 
             y_nl=y_nl_comp, 
             y_lin=y_lin_comp,
             u_const=u_pequeno)

    # 2. Simulação de Resposta em Frequência (Item f)
    A_in = 0.03333
    freqs_omega = [0.901, 9.01, 90.1]
    # Ajustar tempo total para capturar múltiplos ciclos de cada frequência
    num_ciclos = 50 
    
    resultados_freq = {}
    for omega in freqs_omega:
        periodo = 2 * np.pi / omega
        tempo_total_freq = num_ciclos * periodo
        
        tempo_f, u_f, y_f = simular_resposta_frequencia(A_in, omega, Ts, tempo_total_freq)
        ganho, defasagem = calcular_ganho_defasagem(tempo_f, u_f, y_f, omega)
        
        resultados_freq[f'omega_{omega}'] = {
            'tempo': tempo_f,
            'u_input': u_f,
            'y_output': y_f,
            'ganho': ganho,
            'defasagem': defasagem
        }

    np.savez("simulacao_frequencia.npz", **resultados_freq)

    print("Simulações concluídas e dados salvos:")
    print("- 'simulacao_comparacao.npz': Resposta ao degrau para sistemas não-linear e linearizado.")
    print("- 'simulacao_frequencia.npz': Resposta em frequência para as senoides de teste.")