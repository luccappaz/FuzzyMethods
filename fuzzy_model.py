import pandas as pd
from pathlib import Path
import pickle
from itertools import product
import numpy as np
from utils import mf_function, get_input_center, fuzzy_inference, get_mf_limits


class WMModel:
    def __init__(self):
        self.defined_mfs: bool = False
        self.rules: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        self.data_gen_rules: (
            dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
        ) = None
        self.labels: dict[str, list[str]] | None = None
        self.orig_cols: list[str] | None = None
        self.limits: dict[str, tuple] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        att_labels: dict[str, list[str]],
        extrapolating: bool = False,
    ) -> None:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.labels = att_labels
        self.orig_cols = list(X.columns)
        self.limits = {att: get_mf_limits(X, att) for att in self.orig_cols}
        self.__generate_rules(X, y, att_labels)
        if extrapolating:
            file_path = Path("regras_fuzzy_extrapoladas.pkl")
            if file_path.exists():
                print(
                    f"Arquivo '{file_path}' encontrado! Carregando a base de regras pré-treinada..."
                )

                with open(file_path, "rb") as f:
                    self.rules = pickle.load(f)
            else:
                print(
                    f"Arquivo '{file_path}' não encontrado. Iniciando a geração das regras extrapoladas..."
                )
                self.__extrapolate_rules(file_path)
        else:
            pass

    def pred(
        self, X_test: pd.DataFrame, probs: bool = False, threshold: float = 0.5
    ) -> np.ndarray:
        assert self.rules is not None, "É necessário treinar o modelo primeiro"
        assert self.labels is not None
        assert self.orig_cols is not None
        assert self.limits is not None
        y = np.zeros(len(X_test))

        for idx, (_, row) in enumerate(X_test.iterrows()):
            x_client = row[self.orig_cols].values  # type: ignore
            y_val = fuzzy_inference(
                x_client,
                self.limits,
                self.orig_cols,
                self.rules,
                self.labels,
                probs,
                threshold,
            )
            y[idx] = y_val  # type: ignore

        return y

    def __generate_rules(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        att_labels: dict[str, list[str]],
    ) -> None:
        assert self.orig_cols is not None
        assert self.limits is not None

        rules: dict[int, dict[str, str]] = {}
        fire_strengths = []

        for index, row in X.iterrows():
            max_labels = {}
            w = 1
            for col in self.orig_cols:
                val = float(row[col])  # type: ignore
                ms = mf_function(val, self.limits[col])
                ms = ms[: len(att_labels[col])]
                w = w * np.max(ms)
                max_labels[col] = att_labels[col][np.argmax(ms)]

            rules[index] = max_labels  # type: ignore
            fire_strengths.append(w)

        fire_strengths = np.array(fire_strengths)

        idx_per_rule: dict[tuple, list[int]] = {}
        for key, value in rules.items():
            rule_tuple = tuple(value.values())
            if rule_tuple not in idx_per_rule.keys():
                idx_per_rule[rule_tuple] = []
                idx_per_rule[rule_tuple].append(key)
            else:
                idx_per_rule[rule_tuple].append(key)

        rules_w_value: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, value in idx_per_rule.items():
            if np.sum(fire_strengths[value]) != 0:
                weighted_average = np.divide(
                    np.sum(np.multiply(fire_strengths[value], y.loc[value, "class"])),
                    np.sum(fire_strengths[value]),
                )
            else:
                weighted_average = np.array(0)

            if np.sum(fire_strengths[value]) != 0:
                weighted_mad = np.divide(
                    np.sum(
                        np.multiply(
                            np.abs(y.loc[value, "class"] - weighted_average),
                            fire_strengths[value],
                        )
                    ),
                    np.sum(fire_strengths[value]),
                )
            else:
                weighted_mad = np.array(0)

            if np.max(y.loc[value, "class"]) == np.min(y.loc[value, "class"]):
                doc = np.array(1)
            else:
                doc = 1 - np.divide(
                    weighted_mad,
                    np.max(y.loc[value, "class"]) - np.min(y.loc[value, "class"]),
                )

            rules_w_value[key] = (weighted_average, weighted_mad, doc)

        self.rules = rules_w_value
        self.data_gen_rules = self.rules

    def __extrapolate_rules(self, file_path: Path) -> None:
        assert self.rules is not None, "Não há regras para extrapolar!"
        assert self.labels is not None
        assert self.orig_cols is not None
        assert self.limits is not None

        all_rules = list(product(*list(self.labels.values())))

        print("Pré-computando o centro do input de todas as regras possíveis...")
        center_per_rule = {
            r: np.array(get_input_center(r, self.limits, self.orig_cols, self.labels))
            for r in all_rules
        }

        while len(all_rules) != len(list(self.rules.keys())):
            data_generated_rules = list(self.rules.keys())
            rules_neighbors: dict[tuple, list] = {}

            def dist_by_rules(rule, ref_rule):
                sims = 0
                for idx in range(len(rule)):
                    if rule[idx] == ref_rule[idx]:
                        sims = sims + 1
                    else:
                        continue
                return sims

            for rule in all_rules:
                if rule in data_generated_rules:
                    continue
                else:
                    neighbors = [
                        ref_rule
                        for ref_rule in data_generated_rules
                        if dist_by_rules(rule, ref_rule) == 4
                    ]
                    if neighbors:
                        rules_neighbors[rule] = neighbors
                    else:
                        continue

            max_neighbors = max(len(val) for val in rules_neighbors.values())
            max_group = [
                k for (k, v) in rules_neighbors.items() if len(v) == max_neighbors
            ]
            for rule in max_group:
                rule_input_center = center_per_rule[rule]
                neighbors_yc = []
                neighbors_doc = []
                neighbors_dis = []
                neighbors = rules_neighbors[rule]
                for neighbor in neighbors:
                    neighbor_input_center = center_per_rule[neighbor]
                    (weighted_average, weighted_mad, doc) = self.rules[neighbor]
                    neighbors_yc.append(weighted_average)
                    neighbors_doc.append(doc)
                    neighbors_dis.append(
                        np.linalg.norm(neighbor_input_center - rule_input_center)
                    )

                neighbors_yc = np.array(neighbors_yc)
                neighbors_doc = np.array(neighbors_doc)
                neighbors_dis = np.array(neighbors_dis)

                yc = np.divide(
                    np.sum(neighbors_yc * neighbors_doc * neighbors_dis),
                    np.sum(np.multiply(neighbors_doc, neighbors_dis)),
                )

                weighted_mad = np.divide(
                    np.sum(np.multiply(np.abs(neighbors_yc - yc), neighbors_doc)),
                    np.sum(neighbors_doc),
                )

                if np.max(neighbors_yc) != np.min(neighbors_yc):
                    doc = 1 - np.divide(
                        weighted_mad, np.max(neighbors_yc) - np.min(neighbors_yc)
                    )
                else:
                    doc = np.array(1)

                self.rules[rule] = (yc, weighted_mad, doc)

        with open(file_path, "wb") as f:
            pickle.dump(self.rules, f)
        print(
            f"Regras fuzzy completas, com tamanho {len(self.rules)} salvas em {file_path}"
        )
