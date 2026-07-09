import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, initializers


class ANFISLayer(layers.Layer):
    def __init__(self, num_inputs: int, num_mfs: list[int], **kwargs):
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        self.num_rules = int(np.prod(num_mfs))
        super().__init__(**kwargs)

    def build(self, input_shape):

        mu_init = np.zeros((self.num_inputs, max(self.num_mfs)), dtype=np.float32)
        for i in range(self.num_inputs):
            mu_init[i, : self.num_mfs[i]] = np.linspace(0.1, 0.9, self.num_mfs[i])

        self.mu = self.add_weight(
            name="mu",
            shape=(self.num_inputs, max(self.num_mfs)),
            initializer=lambda _, dtype: tf.convert_to_tensor(mu_init, dtype=dtype),
            trainable=True,
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=(self.num_inputs, max(self.num_mfs)),
            initializer=initializers.Constant(value=0.3),
            trainable=True,
        )
        self.consequents = self.add_weight(
            name="Consequents",
            shape=(self.num_rules, self.num_inputs + 1),
            initializer=initializers.glorot_normal(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        inputs_expnd = tf.expand_dims(inputs, axis=-1)
        mu_expnd = tf.expand_dims(self.mu, axis=0)
        sigma_expnd = tf.expand_dims(self.sigma, axis=0)

        ms_degrees = tf.math.exp(
            tf.math.divide_no_nan(
                -tf.math.pow(inputs_expnd - mu_expnd, 2),
                2 * tf.math.pow(sigma_expnd, 2),
            )
        )
        # shape: (batch_size, num_inputs, max(num_mfs))

        mfs_list = [ms_degrees[:, i, : self.num_mfs[i]] for i in range(self.num_inputs)]

        grid = tf.meshgrid(*[tf.range(n) for n in self.num_mfs], indexing="ij")
        # Achata o grid para obter uma matriz de coordenadas de tamanho (num_rules, num_inputs)
        flat_grid = tf.stack([tf.reshape(g, [-1]) for g in grid], axis=-1)

        gathered_rules_list = []
        for i in range(self.num_inputs):
            # flat_grid[:, i] pega os índices de regras específicos para a variável i
            # O gather resulta em um formato (batch_size, num_rules)
            var_rules = tf.gather(mfs_list[i], flat_grid[:, i], axis=1)
            gathered_rules_list.append(var_rules)

        w_rules_gathered = tf.stack(gathered_rules_list, axis=2)

        # T-NORMA DO MÍNIMO: Pegamos o menor valor de pertinência entre os atributos para cada regra
        w = tf.reduce_min(w_rules_gathered, axis=2)
        # Formato final cravado de 'w': (batch_size, num_rules)
        # shape: (batch_size, num_rules)
        sum_w = tf.reduce_sum(w, axis=-1, keepdims=True)
        w_norm = tf.math.divide_no_nan(w, sum_w)

        bias = tf.ones(shape=[batch_size, 1])
        inputs_w_bias = tf.concat([inputs, bias], axis=-1)

        rule_outputs = tf.matmul(inputs_w_bias, self.consequents, transpose_b=True)
        output = tf.reduce_sum(w_norm * rule_outputs, axis=-1, keepdims=True)
        return output


class ANFISModel(models.Model):
    def __init__(self, num_inputs: int, num_mfs: list[int]):
        super().__init__()
        self.anfis_layer = ANFISLayer(num_inputs, num_mfs)
        # self.sigmoid = layers.Activation("sigmoid") # Ativar Sigmoid para o XOR

    def call(self, inputs):
        y = self.anfis_layer(inputs)
        return y


def plot_mfs(model: ANFISModel, X: pd.DataFrame):
    output_dir = "graficos_anfis"
    os.makedirs(output_dir, exist_ok=True)

    anfis_layer = model.anfis_layer
    pesos_mu = anfis_layer.mu.numpy()
    pesos_sigma = anfis_layer.sigma.numpy()
    num_inputs = anfis_layer.num_inputs
    num_mfs = anfis_layer.num_mfs

    nome_atributos = (
        X.columns
        if hasattr(X, "columns")
        else [f"Atributo_{i}" for i in range(num_inputs)]
    )
    x_plot = np.linspace(0, 1, 500)

    print(f"Salvando gráficos individuais na pasta '{output_dir}'...")

    for i in range(num_inputs):
        plt.figure(figsize=(7, 4))

        num_curvas = num_mfs[i]
        for j in range(num_curvas):
            mu_atual = pesos_mu[i, j]
            sigma_atual = pesos_sigma[i, j]

            # Cálculo da curva Gaussiana
            y_plot = np.exp(-((x_plot - mu_atual) ** 2) / (2 * (sigma_atual**2)))

            plt.plot(x_plot, y_plot, label=f"MF {j + 1}", linewidth=2)

        # Customização do gráfico
        plt.title(
            f"Funções de Pertencimento: {nome_atributos[i]}",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Valor Normalizado [0, 1]", fontsize=10)
        plt.ylabel("Grau de Pertencimento (μ)", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize="small")
        plt.tight_layout()

        # Cria um nome de arquivo limpo (removendo espaços ou caracteres estranhos se houver)
        filename = f"{nome_atributos[i].replace(' ', '_').lower()}_mfs.png"  # type: ignore
        filepath = os.path.join(output_dir, filename)

        # Salva em alta resolução (300 DPI é o padrão para impressão/relatórios)
        plt.savefig(filepath, dpi=300)
        plt.close()  # Fecha a figura atual para liberar a memória do computador

    print("Todos os gráficos foram salvos com sucesso!")
