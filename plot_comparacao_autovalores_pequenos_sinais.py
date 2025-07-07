import numpy as np
import matplotlib.pyplot as plt

# --- Carregar os dados da simulação ---
try:
    data_comparacao = np.load("simulacao_comparacao.npz")
    tempo_comp = data_comparacao["tempo"]
    x_nl_comp = data_comparacao["x_nl"]
    y_nl_comp = data_comparacao["y_nl"]
    x_lin_comp = data_comparacao["x_lin"]
    y_lin_comp = data_comparacao["y_lin"]
    u_valor = data_comparacao["u_valor"]
except FileNotFoundError:
    print("Erro: O arquivo 'simulacao_comparacao.npz' não foi encontrado.")
    print("Por favor, execute o script 'simulacao.py' primeiro para gerar os dados.")
    exit()

# --- Criar a figura e os eixos para os 4 subplots ---
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# --- Plot 1: Comparação de Estados ---
axs[0, 0].plot(tempo_comp, x_lin_comp[:, 0], 'r--', label='Linearizado x1')
axs[0, 0].plot(tempo_comp, x_lin_comp[:, 1], 'r:', label='Linearizado x2')
axs[0, 0].plot(tempo_comp, x_nl_comp[:, 0], 'b-', label='Não linear x1')
axs[0, 0].plot(tempo_comp, x_nl_comp[:, 1], 'b-.', label='Não linear x2')
axs[0, 0].set_title('Comparação de Estados')
axs[0, 0].set_xlabel('Tempo (s)')
axs[0, 0].set_ylabel('Estados')
axs[0, 0].legend()
axs[0, 0].grid(True)

# --- Plot 2: Comparação de Saída ---
axs[0, 1].plot(tempo_comp, y_lin_comp, 'r-', label='Sistema linearizado')
axs[0, 1].plot(tempo_comp, y_nl_comp, 'b--', label='Sistema não linear original')
axs[0, 1].set_title('Comparação de Saída')
axs[0, 1].set_xlabel('Tempo (s)')
axs[0, 1].set_ylabel('Saída y')
axs[0, 1].legend()
axs[0, 1].grid(True)

# --- Plot 3: Gráfico de Autovalores ---
# <<< CORREÇÃO: Usando a matriz A correta e garantindo consistência ---
# Matriz A do seu sistema linearizado
A = np.array([[1.45319, 0.25489], [-0.86893, 0.51877]])
autovalores = np.linalg.eigvals(A)

# Desenha o círculo unitário
circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
axs[1, 0].add_artist(circle)
# Plota os autovalores a partir da variável 'autovalores'
axs[1, 0].plot(autovalores.real, autovalores.imag, 'rx', markersize=10, markeredgewidth=2, label='Autovalores')
axs[1, 0].set_title('Autovalores (Círculo Unitário)')
axs[1, 0].set_xlabel('Parte Real')
axs[1, 0].set_ylabel('Parte Imaginária')
axs[1, 0].axvline(0, color='k', linestyle=':', linewidth=0.5)
axs[1, 0].axhline(0, color='k', linestyle=':', linewidth=0.5)
# Ajuste de zoom para melhor visualização dos autovalores
lim_max = max(np.max(np.abs(autovalores.real)), np.max(np.abs(autovalores.imag))) * 1.2
lim_max = max(lim_max, 1.1) # Garante que pelo menos o círculo unitário apareça
axs[1, 0].set_xlim([-lim_max, lim_max])
axs[1, 0].set_ylim([-lim_max, lim_max])
axs[1, 0].set_aspect('equal', adjustable='box')
axs[1, 0].legend()
axs[1, 0].grid(True)

# --- Bloco de Texto com Informações ---
ax4 = axs[1, 1]
ax4.axis('off')

# <<< CORREÇÃO: Garantindo que o texto use a mesma variável 'autovalores' ---
info_text = (
    "---- Informações da Simulação ----\n"
    "Tipo: Discreto\n"
    f"Entrada Degrau (u): {u_valor}\n"
    f"Condição Inicial (x): {data_comparacao['x_lin'][0].tolist()}\n"
    "\n---- Autovalores Calculados ----\n"
    f"  {autovalores[0]:.4f}\n"
    f"  {autovalores[1]:.4f}"
)

ax4.text(0.05, 0.5, info_text, fontsize=11, va='center', fontfamily='monospace')

# --- Finalização e Salvamento ---
plt.tight_layout()
plt.savefig('comparacao_autovalores.png')
print("Gráfico de comparação de estados, saída e autovalores salvo em comparacao_autovalores.png")