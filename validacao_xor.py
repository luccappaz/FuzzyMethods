import numpy as np
import keras
import matplotlib.pyplot as plt
from ANFIS import ANFISModel

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
X_train, X_test = X_complex[:800], X_complex[800:]
y_train, y_test = y_complex[:800], y_complex[800:]

# Configurando o ANFIS para 2 Entradas
num_inputs = 2
num_mfs = [3, 3]  # 3 MFs para X1 e 3 MFs para X2 = 9 regras cruzadas na grade

model = ANFISModel(num_inputs, num_mfs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.02),
    loss="binary_crossentropy",
    metrics=["mae"],
)

# Treino Estendido (Problemas 2D exigem mais épocas para ajustar os centros)
print("\nTreinando o ANFIS no padrão de xadrez...")
model.fit(X_train, y_train, epochs=250, batch_size=32)

# Criando uma Grade Densa para mapear a Superfície de Decisão em Alta Resolução
print("\nMapeando a superfície de decisão aprendida...")
x1_grid = np.linspace(0, 1, 100)
x2_grid = np.linspace(0, 1, 100)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
X_grid_eval = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=-1).astype("float32")

# Predição da grade inteira
y_grid_scores = model.predict(X_grid_eval)
Y_grid_mesh = y_grid_scores.reshape(X1_mesh.shape)

# Plotando e Salvando o Mapa de Calor
plt.figure(figsize=(8, 6))

# Desenhando a superfície de probabilidade calculada pelo ANFIS
contour = plt.contourf(
    X1_mesh, X2_mesh, Y_grid_mesh, levels=50, cmap="RdYlBu_r", alpha=0.8
)
cbar = plt.colorbar(contour)
cbar.set_label("Probabilidade da Classe 1", fontsize=10)

# Plotando os pontos reais de teste por cima para ver se o modelo acertou as regiões
plt.scatter(
    X_test[y_test.ravel() == 0, 0],
    X_test[y_test.ravel() == 0, 1],
    color="blue",
    edgecolors="k",
    label="Classe 0 (Real)",
    alpha=0.6,
)
plt.scatter(
    X_test[y_test.ravel() == 1, 0],
    X_test[y_test.ravel() == 1, 1],
    color="red",
    edgecolors="k",
    label="Classe 1 (Real)",
    alpha=0.6,
)

plt.title(
    "Superfície de Decisão ANFIS: Problema XOR 2D", fontsize=12, fontweight="bold"
)
plt.xlabel("Atributo X1")
plt.ylabel("Atributo X2")
plt.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

output_file = "validacao_XOR.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"🎉 Pronto! O mapa de calor 2D foi salvo em: '{output_file}'")
