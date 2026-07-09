import pandas as pd
import numpy as np
from pathlib import Path
import json


def fuzzy_inference(
    client,
    limits: dict[str, tuple],
    orig_cols: list[str],
    rules: dict,
    labels: dict,
    probs: bool = False,
    threshold: float = 0.5,
) -> float:

    client_ms = {}
    for idx, attribute in enumerate(orig_cols):
        val = client[idx]
        client_ms[attribute] = mf_function(val, limits[attribute])

    fire_strengths_per_rule = {}

    for rule in rules.keys():
        pertinencias_da_regra = []
        for idx, attribute in enumerate(orig_cols):
            label_obrigatoria = rule[idx]
            posicao_label = labels[attribute].index(label_obrigatoria)
            pertinencias_da_regra.append(client_ms[attribute][posicao_label])

        fire_strengths_per_rule[rule] = np.prod(pertinencias_da_regra)

    pesos_regras = np.array(list(fire_strengths_per_rule.values()))
    denominador = np.sum(pesos_regras)

    if denominador == 0:
        # Em sistemas dinâmicos que oscilam entre negativos e positivos (Narendra-Li),
        # retornar 0.5 (antigo padrão para classificação) joga a predição para um viés estático.
        # Se nenhuma regra ativar, retornamos a média neutra do sinal (0.0).
        y = 0.0
    else:
        # Garante a extração limpa do float de weighted_average (que é o primeiro elemento da tupla)
        yc_values = np.array([float(val[0]) for val in rules.values()])
        y = np.sum(yc_values * pesos_regras) / denominador

    if not probs:
        if y < threshold:
            return float(0)
        else:
            float(1)
    return float(y)


def get_input_center(
    rule: tuple,
    limits: dict[str, tuple],
    orig_cols: list[str],
    labels_dict: dict[str, list[str]],
) -> list:
    input_center = []

    for attribute, quality in zip(orig_cols, rule):
        labels = labels_dict[attribute]
        idx = labels.index(quality)
        safe_idx = min(idx, len(limits[attribute]) - 1)
        x_center = limits[attribute][safe_idx]
        input_center.append(x_center)
    return input_center


def get_mf_limits(X: pd.DataFrame, att: str) -> tuple:
    column = np.array(X[att].values)
    min_val: float = np.min(column)
    max_val: float = np.max(column)
    q25: float = np.quantile(column, 0.25)
    q50: float = np.quantile(column, 0.5)
    q75: float = np.quantile(column, 0.75)

    return min_val, q25, q50, q75, max_val


def mf_function(val: float, limits: tuple) -> tuple:
    """
    Retorna os graus de pertinência fuzzy adaptados dinamicamente.
    Se limits contiver os 5 parâmetros clássicos, calcula as 5 partições.
    """
    min_val, q25, q50, q75, max_val = limits

    y0 = trimf(val, (min_val, min_val, q25))
    y1 = trimf(val, (min_val, q25, q50))
    y2 = trimf(val, (q25, q50, q75))
    y3 = trimf(val, (q50, q75, max_val))
    y4 = trimf(val, (q75, max_val, max_val))

    return (y0, y1, y2, y3, y4)


def trimf(val: float, params: tuple[float, float, float]) -> float:
    a, b, c = params
    if val <= a or val >= c:
        return 0.0
    if a < val < b:
        return (val - a) / (b - a) if b != a else 1.0
    elif b < val < c:
        return (val - c) / (b - c) if b != c else 1.0
    else:
        return float(1.0)


def save_results(
    mae: float,
    mse: float,
    num_rules: int,
    labels: dict,
    filename: str = "narendra_li.json",
) -> None:
    log = {
        "Benchmark": "Narendra-Li",
        "Algoritmo": "Wang-Mendel (WMModel)",
        "Metricas": {"MAE": float(mae), "MSE": float(mse)},
        "Estrutura": {
            "Numero_Regras_Geradas": int(num_rules),
            "Variaveis_Labels": labels,
        },
    }

    save_path = Path("logs") / filename
    save_path.parent.mkdir(exist_ok=True, parents=True)

    if save_path.exists():
        parent, base, extension = save_path.parent, save_path.stem, save_path.suffix
        counter = 1
        while True:
            save_path = Path(parent / f"{base}{counter}{extension}")
            if not save_path.exists():
                break
            counter += 1

    save_path.write_text(
        json.dumps(log, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Resultados do Narendra-Li salvos com sucesso em: {save_path}")
