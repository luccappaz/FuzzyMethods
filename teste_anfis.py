import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from ANFIS import ANFISModel
from load_data import load_data
from utils import save_results

X, y = load_data()

# Divisão de Treino e Teste mantendo os DataFrames do Pandas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

X_train = pd.DataFrame(X_train, columns=X.columns)
y_train = pd.DataFrame(y_train, columns=y.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_test = pd.DataFrame(y_test, columns=y.columns)

print("Mapeando dicionário de termos linguísticos...")
termos_linguisticos = ["Baixo", "Medio", "Alto"]

att_labels = {
    "y_t": termos_linguisticos,
    "y_t1": termos_linguisticos,
    "y_t2": termos_linguisticos,
    "u_t": termos_linguisticos,
    "u_t1": termos_linguisticos,
}

model = ANFISModel(5, [3] * 5)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)


print("\n Iniciando o ajuste do modelo Wang-Mendel (fit)...")
# Ativamos extrapolating=True para testar o preenchimento de vazios e geração do pkl
model.fit(X_train, y_train, epochs=250, batch_size=120)

# ==========================================
# PREDIÇÃO E AVALIAÇÃO DE CONVERGÊNCIA
# ==========================================
print("\n Executando inferência fuzzy no conjunto de teste...")
predictions = model.predict(X_test)
predictions = predictions.flatten()

# Métricas de validação
mae = mean_absolute_error(y_test["class"], predictions)
mse = mean_squared_error(y_test["class"], predictions)

print("\n--- RESULTADOS DO BENCHMARK ---")
print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")

# Amostra rápida das predições contra o gabarito
df_comparacao = pd.DataFrame(
    {"Real": y_test["class"].values[:5], "Predito": predictions[:5]}
)
df_comparacao.to_csv("logs/anfis_results.csv")
print("\n Amostra Comparativa (Real vs Predito):")
print(df_comparacao)
save_results(mae, mse, np.prod([3] * 5), att_labels, filename="ANFIS_narendra-li.json", algoritmo="ANFIS")  # type: ignore
