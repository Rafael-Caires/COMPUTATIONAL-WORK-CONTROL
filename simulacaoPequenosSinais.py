import numpy as np
import matplotlib.pyplot as plt
from sistema_controle import sistema_nao_linear, saida_nao_linear, sistema_linearizado, saida_linearizada

# --- Simulação para Comparação de Estados e Saída ---
def simular_comparacao(x0, u_const, Ts, tempo_total):
    num_passos = int(tempo_total / Ts)
    tempo = np.linspace(0, tempo_total, num_passos)

    x_nl_hist = np.zeros((num_passos, 2))
    y_nl_hist = np.zeros(num_passos)
    x_lin_hist = np.zeros((num_passos, 2))
    y_lin_hist = np.zeros(num_passos)

    x_nl_hist[0] = x0
    y_nl_hist[0] = saida_nao_linear(x0, u_const)
    x_lin_hist[0] = x0
    y_lin_hist[0] = saida_linearizada(x0, u_const)

    x_nl_atual = x0
    x_lin_atual = x0

    for i in range(1, num_passos):
        x_nl_atual = sistema_nao_linear(x_nl_atual, u_const, Ts)
        y_nl_hist[i] = saida_nao_linear(x_nl_atual, u_const)
        x_nl_hist[i] = x_nl_atual

        x_lin_atual = sistema_linearizado(x_lin_atual, u_const, Ts)
        y_lin_hist[i] = saida_linearizada(x_lin_atual, u_const)
        x_lin_hist[i] = x_lin_atual

    return tempo, x_nl_hist, y_nl_hist, x_lin_hist, y_lin_hist

# Parâmetros da simulação
x0_comparacao = np.array([0.0, 0.0]) 
u_comparacao = 0.001 # Usando o valor do item (d)-1 do trabalho
Ts_comparacao = 0.0098
tempo_total_comparacao = 10.0

tempo_comp, x_nl_comp, y_nl_comp, x_lin_comp, y_lin_comp = simular_comparacao(x0_comparacao, u_comparacao, Ts_comparacao, tempo_total_comparacao)

# <<< MODIFICADO: Adicionado 'u_valor' para salvar o valor da entrada no arquivo.
np.savez("simulacao_comparacao.npz", tempo=tempo_comp, x_nl=x_nl_comp, y_nl=y_nl_comp, x_lin=x_lin_comp, y_lin=y_lin_comp, u_valor=u_comparacao)

print("Simulação de comparação concluída e dados salvos em simulacao_comparacao.npz")


# --- Simulação para Teste de Resposta em Frequência (sem alterações) ---
def simular_resposta_frequencia(A_in, omega, Ts, tempo_total):
    num_passos = int(tempo_total / Ts)
    tempo = np.linspace(0, tempo_total, num_passos)
    u_input = A_in * np.sin(omega * tempo * Ts)

    x_lin_hist = np.zeros((num_passos, 2))
    y_lin_hist = np.zeros(num_passos)
    x_lin_atual = np.array([0.0, 0.0]) 

    for i in range(num_passos):
        x_lin_atual = sistema_linearizado(x_lin_atual, u_input[i], Ts)
        y_lin_hist[i] = saida_linearizada(x_lin_atual, u_input[i])
        x_lin_hist[i] = x_lin_atual

    return tempo, u_input, y_lin_hist

A_in = 0.03333 
Ts_freq = 0.0098
freqs_omega = [0.901, 9.01, 90.1] 
tempo_total_freq = [1000.0, 100.0, 10.0]

resultados_freq = {}
for i, omega in enumerate(freqs_omega):
    tempo_f, u_f, y_f = simular_resposta_frequencia(A_in, omega, Ts_freq, tempo_total_freq[i])
    resultados_freq[f'omega_{omega}'] = {'tempo': tempo_f, 'u_input': u_f, 'y_output': y_f}

np.savez("simulacao_frequencia.npz", **resultados_freq)
print("Simulações de resposta em frequência concluídas e dados salvos em simulacao_frequencia.npz")