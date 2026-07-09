import numpy as np
import pandas as pd


def load_data():
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

    return df_X, df_y
