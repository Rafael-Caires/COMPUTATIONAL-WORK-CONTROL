import numpy as np

# --- Funções do Sistema Não-Linear (Matrícula: 2022421552) ---

def sistema_nao_linear(x, u, Ts):
    """
    Implementa o sistema não-linear discreto específico do aluno.
    Args:
        x: vetor de estado [x1, x2]
        u: sinal de entrada
        Ts: período de amostragem (não utilizado diretamente nas equações, mas conceitual)
    Returns:
        próximo estado [x1_next, x2_next]
    """
    x1, x2 = x

    # Equações de estado não-lineares da ficha do trabalho
    x1_next = 1.523 * x1 + 0.3247 * x2 + 0.0491 * x1 * x2**3 - 0.06981 * x1 * np.cos(x1) \
              - 0.06981 * x2 * np.cos(x1) - u * (0.006862 * x2**3 + 0.005429 * np.cos(x1) + 0.06991) \
              + 0.0491 * x2**4

    x2_next = 0.531 * x2 - 0.8567 * x1 + 0.01587 * x1 * x2**3 - 0.01223 * x1 * np.cos(x1) \
              - 0.01223 * x2 * np.cos(x1) + 0.01587 * x2**4 \
              + u * (0.0004525 * x2**3 + 0.004222 * np.cos(x1) - 0.1193)

    return np.array([x1_next, x2_next])

def saida_nao_linear(x, u):
    """
    Calcula a saída não-linear do sistema específico do aluno.
    Args:
        x: vetor de estado [x1, x2]
        u: sinal de entrada
    Returns:
        valor da saída y
    """
    x1, x2 = x
    # Equação de saída não-linear da ficha do trabalho
    y = 0.02791 * x2 - 0.1034 * x1 - 0.05 * u * x2**3 \
        - 0.005414 * x1 * np.cos(x1) - 0.001422 * x2 * np.cos(x1)
    return y

# --- Funções do Sistema Linearizado ---

def sistema_linearizado(x, u, Ts):
    """
    Implementa o sistema linearizado em torno do ponto de equilíbrio (0,0,0).
    Args:
        x: vetor de estado [x1, x2]
        u: sinal de entrada
        Ts: período de amostragem
    Returns:
        próximo estado linearizado [x1_next, x2_next]
    """
    # Matrizes do sistema linearizado (calculadas para a matrícula 2022421552)
    A = np.array([[1.45319, 0.25489], [-0.86893, 0.51877]])
    B = np.array([[-0.075339], [-0.115078]])
    
    # Ponto de equilíbrio é a origem
    x_eq = np.array([0.0, 0.0])
    u_eq = 0.0

    # Desvios em relação ao ponto de equilíbrio
    delta_x = x - x_eq
    delta_u = np.array([u - u_eq])

    # Equação de estado linearizada: delta_x(k+1) = A * delta_x(k) + B * delta_u(k)
    delta_x_next = A @ delta_x + (B @ delta_u).flatten()

    return delta_x_next + x_eq

def saida_linearizada(x, u):
    """
    Calcula a saída linearizada do sistema.
    Args:
        x: vetor de estado [x1, x2]
        u: sinal de entrada
    Returns:
        valor da saída y linearizada
    """
    # Matrizes do sistema linearizado (calculadas para a matrícula 2022421552)
    C = np.array([[-0.108814, 0.026488]])
    D = np.array([[0.0]])
    
    # Ponto de equilíbrio é a origem
    x_eq = np.array([0.0, 0.0])
    u_eq = 0.0

    # Desvios em relação ao ponto de equilíbrio
    delta_x = x - x_eq
    delta_u = np.array([u - u_eq])

    # Equação de saída linearizada: delta_y(k) = C * delta_x(k) + D * delta_u(k)
    delta_y = C @ delta_x + D @ delta_u

    return delta_y[0]

# --- Análise de Estabilidade ---

def analisar_estabilidade():
    """
    Analisa a estabilidade do sistema linearizado a partir da matriz A.
    Returns:
        autovalores, raio espectral e uma string com a conclusão
    """
    # Matriz A do sistema linearizado
    A = np.array([[1.45319, 0.25489], [-0.86893, 0.51877]])
    autovalores = np.linalg.eigvals(A)
    raio_espectral = max(np.abs(autovalores))
    
    if raio_espectral < 1:
        estabilidade = "Sistema estável (todos autovalores dentro do círculo unitário)"
    else:
        estabilidade = "Sistema instável (pelo menos um autovalor fora do círculo unitário)"
    
    return autovalores, raio_espectral, estabilidade

if __name__ == '__main__':
    # Bloco para testes rápidos das funções
    autovalores, raio, estab = analisar_estabilidade()
    print("--- Análise de Estabilidade do Sistema Linearizado ---")
    print(f"Autovalores: {autovalores}")
    print(f"Raio espectral: {raio:.4f}")
    print(f"Conclusão: {estab}")