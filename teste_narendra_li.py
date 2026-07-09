import numpy as np
import pandas as pd
from utils import save_results
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fuzzy_model import WMModel

# ==========================================
# GERANDO OS DADOS DE NARENDRA-LI
# ==========================================
print("🔄 Gerando dados do benchmark dinâmico de Narendra-Li...")
np.random.seed(42)
steps = 800
y = np.zeros(steps, dtype=np.float32)
u = np.random.uniform(-1, 1, steps).astype(np.float32)

for k in range(2, steps - 1):
    numerador = y[k] * y[k - 1] * y[k - 2] * u[k - 1] * (y[k - 2] - 1.0) + u[k]
    denominador = 1.0 + y[k - 1] ** 2 + y[k - 2] ** 2
    y[k + 1] = numerador / denominador

# Fatiando o histórico para criar 5 colunas de entrada coerentes
df_X = pd.DataFrame(
    {
        "y_t": y[4 : steps - 1],
        "y_t1": y[3 : steps - 2],
        "y_t2": y[2 : steps - 3],
        "u_t": u[4 : steps - 1],
        "u_t1": u[3 : steps - 2],
    }
)

df_y = pd.DataFrame({"class": y[5:steps]})

# Divisão de Treino e Teste mantendo os DataFrames do Pandas
X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.2, random_state=42, shuffle=False
)

X_train = pd.DataFrame(X_train, columns=df_X.columns)
y_train = pd.DataFrame(y_train, columns=df_y.columns)
X_test = pd.DataFrame(X_test, columns=df_X.columns)
y_test = pd.DataFrame(y_test, columns=df_y.columns)

print("Mapeando dicionário de termos linguísticos...")
termos_linguisticos = ["Baixo", "Medio", "Alto"]

att_labels = {
    "y_t": termos_linguisticos,
    "y_t1": termos_linguisticos,
    "y_t2": termos_linguisticos,
    "u_t": termos_linguisticos,
    "u_t1": termos_linguisticos,
}

model = WMModel()

print("\n Iniciando o ajuste do modelo Wang-Mendel (fit)...")
# Ativamos extrapolating=True para testar o preenchimento de vazios e geração do pkl
model.fit(X_train, y_train, att_labels=att_labels, extrapolating=True)

# ==========================================
# PREDIÇÃO E AVALIAÇÃO DE CONVERGÊNCIA
# ==========================================
print("\n Executando inferência fuzzy no conjunto de teste...")
predictions = model.pred(X_test, probs=True)

# Métricas de validação
mae = mean_absolute_error(y_test["class"], predictions)
mse = mean_squared_error(y_test["class"], predictions)

print("\n--- RESULTADOS DO BENCHMARK ---")
print(
    f"Número de regras geradas/extrapoladas: {len(model.rules) if model.rules else 0}"
)
print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")

# Amostra rápida das predições contra o gabarito
df_comparacao = pd.DataFrame(
    {"Real": y_test["class"].values[:5], "Predito": predictions[:5]}
)
df_comparacao.to_csv("logs/Narendra-Li_results")
print("\n Amostra Comparativa (Real vs Predito):")
print(df_comparacao)
save_results(mae, mse, len(model.rules), att_labels)  # type: ignore
