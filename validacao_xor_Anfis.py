import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy_model import WMModel

# Gerando Dados Sintéticos 2D (Problema do XOR Contínuo / Tabuleiro)
print("Gerando dados complexos bidimensionais (Tabuleiro 2x2)...")
np.random.seed(42)

# Criando 1000 pontos aleatórios espalhados no espaço [0, 1] x [0, 1]
X_complex = np.random.uniform(0, 1, (1000, 2)).astype("float32")

# Lógica do XOR: se X1 e X2 estiverem em cantos opostos, classe 1. Caso contrário, classe 0.
y_complex = np.zeros((1000, 1), dtype="float32")
for i in range(1000):
    x1, x2 = X_complex[i, 0], X_complex[i, 1]
    if (x1 < 0.5 and x2 >= 0.5) or (x1 >= 0.5 and x2 < 0.5):
        y_complex[i, 0] = 1.0

# Divisão simples treino/teste
X_train_np, X_test_np = X_complex[:800], X_complex[800:]
y_train_np, y_test_np = y_complex[:800], y_complex[800:]

cols = ["X1", "X2"]
X_train = pd.DataFrame(X_train_np, columns=cols)
X_test = pd.DataFrame(X_test_np, columns=cols)
y_train = pd.DataFrame(y_train_np, columns=["class"])
y_test = pd.DataFrame(y_test_np, columns=["class"])

# Configurando os labels de termos linguísticos para as 2 variáveis
termos_xor = ["Muito Baixo", "Baixo", "Medio", "Alto", "Muito Alto"]
att_labels = {"X1": termos_xor, "X2": termos_xor}

# Inicializando o modelo Wang-Mendel com os labels apropriados
model = WMModel()

print("\nTreinando o Wang-Mendel no padrão de xadrez (XOR)...")
model.fit(X_train, y_train, extrapolating=True, att_labels=att_labels)

# Criando uma Grade Densa para mapear a Superfície de Decisão em Alta Resolução
print("\nMapeando a superfície de decisão aprendida pelo WM...")
x1_grid = np.linspace(0, 1, 100)
x2_grid = np.linspace(0, 1, 100)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

# Monta o DataFrame da grade com os mesmos nomes de colunas do treino
X_grid_eval = pd.DataFrame(
    np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=-1), columns=cols
)

# === AJUSTE DE PREDIÇÃO: Usando .pred com probs=True para regressão/superfície ===
y_grid_scores = model.pred(X_grid_eval, probs=True)
y_grid_scores = np.array(y_grid_scores)  # Garante que é array para o reshape
Y_grid_mesh = y_grid_scores.reshape(X1_mesh.shape)

# Plotando e Salvando o Mapa de Calor
plt.figure(figsize=(8, 6))

# Desenhando a superfície calculada pelo Wang-Mendel
contour = plt.contourf(
    X1_mesh, X2_mesh, Y_grid_mesh, levels=50, cmap="RdYlBu_r", alpha=0.8
)
cbar = plt.colorbar(contour)
cbar.set_label("Valor Ativado da Regra (Classe 1)", fontsize=10)

# Plotando os pontos reais de teste por cima para ver as fronteiras
plt.scatter(
    X_test_np[y_test_np.ravel() == 0, 0],
    X_test_np[y_test_np.ravel() == 0, 1],
    color="blue",
    edgecolors="k",
    label="Classe 0 (Real)",
    alpha=0.6,
)
plt.scatter(
    X_test_np[y_test_np.ravel() == 1, 0],
    X_test_np[y_test_np.ravel() == 1, 1],
    color="red",
    edgecolors="k",
    label="Classe 1 (Real)",
    alpha=0.6,
)

plt.title(
    "Superfície de Decisão Wang-Mendel: Problema XOR 2D", fontsize=12, fontweight="bold"
)
plt.xlabel("Atributo X1")
plt.ylabel("Atributo X2")
plt.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

output_file = "validacao_XOR_WangMendel.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"🎉 Pronto! O mapa de calor 2D do WM foi salvo em: '{output_file}'")
