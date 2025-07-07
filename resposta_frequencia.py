import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def plot_bode_resposta_frequencia():
    """
    Gera o Diagrama de Bode e os gráficos de resposta em frequência simulada.
    """
    # --- Parâmetros e Matrizes do Sistema Linearizado (Matrícula: 2022421552) ---
    A = np.array([[1.45319, 0.25489], [-0.86893, 0.51877]])
    B = np.array([[-0.075339], [-0.115078]])
    C = np.array([[-0.108814, 0.026488]])
    D = np.array([[0.0]])
    Ts = 0.0098

    # Cria o objeto de sistema em espaço de estados em tempo discreto
    sys = signal.StateSpace(A, B, C, D, dt=Ts)

    # Frequências de teste da ficha técnica
    freqs_teste = np.array([0.901, 9.01, 90.1])

    # --- Diagrama de Bode ---
    # Calcula a resposta em frequência (magnitude e fase)
    w, mag, phase = signal.dbode(sys, n=2000) # n alto para boa resolução

    plt.figure(figsize=(12, 8))
    
    # Plot da Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag, 'b-')
    plt.title("Diagrama de Bode do Sistema Linearizado")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", ls="-")
    
    # Interpola e marca as frequências de teste no gráfico de magnitude
    mag_teste = np.interp(freqs_teste, w, mag)
    plt.semilogx(freqs_teste, mag_teste, 'ro', markersize=8, label="Frequências de Teste")
    for i, freq in enumerate(freqs_teste):
        plt.text(freq, mag_teste[i], f' {mag_teste[i]:.1f} dB', ha='left', va='bottom')

    # Plot da Fase
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase, 'b-')
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.grid(True, which="both", ls="-")
    
    # Interpola e marca as frequências de teste no gráfico de fase
    phase_teste = np.interp(freqs_teste, w, phase)
    plt.semilogx(freqs_teste, phase_teste, 'ro', markersize=8)
    for i, freq in enumerate(freqs_teste):
        plt.text(freq, phase_teste[i], f' {phase_teste[i]:.1f}°', ha='left', va='top')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("diagrama_bode.png", dpi=300)
    print("Diagrama de Bode salvo em 'diagrama_bode.png'")

    # --- Gráficos da Resposta em Frequência Simulada ---
    data_freq = np.load("simulacao_frequencia.npz", allow_pickle=True)
    
    plt.figure(figsize=(12, 12))
    
    for i, key in enumerate(data_freq.keys()):
        plt.subplot(3, 1, i + 1)
        item = data_freq[key].item()
        tempo = item["tempo"]
        u_input = item["u_input"]
        y_output = item["y_output"]
        omega = float(key.split('_')[1])
        
        # Pega apenas o regime permanente (últimos ciclos)
        inicio = int(0.8 * len(tempo))
        
        plt.plot(tempo[inicio:], u_input[inicio:], 'b-', label="Entrada u(k)")
        plt.plot(tempo[inicio:], y_output[inicio:], 'r--', label="Saída y(k)")
        
        # Adiciona informações de ganho e defasagem calculados na simulação
        ganho = item['ganho']
        defasagem = item['defasagem']
        plt.title(f"Resposta em Frequência Simulada - ω = {omega:.3f} rad/s\n"
                  f"Ganho Simulado: {ganho:.4f} | Defasagem Simulada: {defasagem:.2f}°")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("resposta_frequencia_simulada.png", dpi=300)
    print("Gráficos da resposta em frequência simulada salvos em 'resposta_frequencia_simulada.png'")

    # --- Tabela Comparativa (Bode vs. Simulação) ---
    print("\n--- Tabela Comparativa de Resposta em Frequência ---")
    print("Freq (rad/s) | Mag Bode (dB) | Fase Bode (°) | Ganho Simulado | Defasagem Simulada (°)")
    print("-" * 95)
    
    for i, omega in enumerate(freqs_teste):
        key = f'omega_{omega}'
        item = data_freq[key].item()
        
        mag_db_bode = mag_teste[i]
        phase_deg_bode = phase_teste[i]
        ganho_simulado = item['ganho']
        defasagem_simulada = item['defasagem']
        
        print(f"{omega:12.3f} | {mag_db_bode:13.2f} | {phase_deg_bode:13.1f} | {ganho_simulado:14.4f} | {defasagem_simulada:22.1f}")

if __name__ == '__main__':
    plot_bode_resposta_frequencia()