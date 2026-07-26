# FuzzyMethods: Adaptive Fuzzy Systems and Neuro-Fuzzy Modeling 📊

This repository contains implementations and experiments with **fuzzy logic-based systems** focused on nonlinear function approximation, regression, and predictive modeling.

The main goal of this project is to investigate and validate methods for:

- Automatic fuzzy rule extraction;
- Neuro-fuzzy learning systems;
- Nonlinear function approximation;
- Dynamic system modeling;
- Prediction of complex and chaotic behaviors.

The project explores two main approaches:

- **Wang-Mendel fuzzy rule extraction**, focused on interpretability and symbolic knowledge generation;
- **ANFIS (Adaptive Neuro-Fuzzy Inference System)**, focused on parameter optimization through neural learning.

---

# 🛠 Technologies

- Python
- NumPy
- TensorFlow / Keras
- Fuzzy Logic Systems
- ANFIS
- Wang-Mendel Algorithm
- Neuro-Fuzzy Networks
- Scientific Computing
- Data Analysis

---

# 🧠 Implemented Models

## 1. Wang-Mendel Fuzzy Rule Extraction (`WMModel`)

The classical **Wang-Mendel (WM)** algorithm is implemented to generate an interpretable fuzzy knowledge base from numerical data.

The model maps input-output relationships by extracting fuzzy rules directly from observations.

---

## Dynamic Fuzzy Partitioning

The input space is partitioned dynamically using **five triangular membership functions** based on statistical properties of the dataset:

```text
min, q25, q50, q75, max
```

This approach allows the model to automatically adapt the fuzzy universe of discourse according to the distribution of the input data.

---

## Antecedent Combination

The fuzzy antecedents are combined using the **product T-norm**:

$$
\mu_A(x)=\prod_i \mu_i(x_i)
$$

where each membership value contributes to the final rule activation degree.

---

## Defuzzification

The output prediction is obtained using a weighted center-of-area approach:

$$
y=\frac{\sum_i w_i c_i}{\sum_i w_i}
$$

where:

- $w_i$ represents the rule activation degree;
- $c_i$ represents the consequent center.

---

## Adaptive Geometric Extrapolation

One of the main contributions of this implementation is an adaptive extrapolation mechanism.

When a combination of fuzzy antecedents has no corresponding training samples, the model performs a geometric neighborhood search.

The algorithm searches for rules sharing exactly:

$$
N-1
$$

common antecedents.

The final prediction is estimated using:

- Euclidean distance between membership-function centers;
- Degree of certainty (`DOC`) weighting.

This allows the model to generalize beyond observed regions of the input space.

---

# 2. ANFIS (Adaptive Neuro-Fuzzy Inference System)

The project also implements a neuro-fuzzy model based on the **Takagi-Sugeno architecture**.

Unlike Wang-Mendel, which generates static fuzzy rules, ANFIS represents the fuzzy inference system as a trainable neural architecture.

The model optimizes:

- Membership function parameters (antecedent layer);
- Linear consequent coefficients.

Optimization is performed through learning algorithms such as:

- Gradient descent;
- Hybrid optimization strategies.

---

## ANFIS Architecture

The model follows the classical five-layer structure:

1. **Fuzzification layer**
   - Computes membership values.

2. **Rule layer**
   - Calculates rule activation strengths.

3. **Normalization layer**
   - Normalizes firing strengths.

4. **Consequent layer**
   - Computes Takagi-Sugeno functions.

5. **Output layer**
   - Aggregates the final prediction.

---

# 🔬 Experiments

## XOR Logical Problem

The classical **XOR (Exclusive OR)** problem was used to evaluate the ability of fuzzy systems to represent nonlinear decision boundaries.

Since XOR is not linearly separable, it is a useful benchmark for testing whether fuzzy rules and overlapping membership functions can generate complex decision surfaces.

The experiment evaluates:

- Rule granularity;
- Membership overlap;
- Nonlinear separation capability.

Both models were evaluated:

- Wang-Mendel;
- ANFIS.

Although both achieved satisfactory results, **ANFIS obtained better performance due to its adaptive parameter optimization capability**.

![XOR Result](validacao_XOR.png)

---

# 📈 Dynamic Benchmark

## Narendra-Li Chaotic System Approximation

To evaluate the capability of fuzzy models to approximate nonlinear and chaotic dynamics, the classical **Narendra-Li benchmark** was used.

The dataset is transformed into an autoregressive temporal representation using a lag window.

The model receives five input variables.

## Inputs

$$
X = \{y_t, y_{t-1}, y_{t-2}, u_t, u_{t-1}\}
$$

## Target

The target variable represents the continuous nonlinear dynamic signal:

$$
y
$$

The benchmark evaluates the ability of fuzzy systems to model temporal dependencies and nonlinear behavior.

---

# 📊 Results

## XOR Benchmark

| Model | Result |
|---|---|
| Wang-Mendel | Successful nonlinear separation |
| ANFIS | Better classification performance |

---

## Narendra-Li Benchmark

| Model | RMSE | MAE |
|---|---|---|
| Wang-Mendel | TBD | TBD |
| ANFIS | TBD | TBD |

---

# 📁 Project Structure

```text
├── fuzzy_model.py          # WMModel implementation and rule extrapolation logic
├── anfis_model.py          # ANFIS neural fuzzy architecture
├── utils.py                # Membership functions, inference utilities, and helpers
├── teste_narendra_li.py    # Narendra-Li benchmark training and evaluation
├── validacao_xor.py        # XOR experiment and validation
├── pyproject.toml          # Project configuration and dependencies (uv)
└── logs/                   # Training logs, JSON metrics, and CSV results
```

---

# 🚀 Future Improvements

Possible future extensions:

- Add additional fuzzy partitioning strategies;
- Compare different membership functions (Gaussian, trapezoidal, triangular);
- Implement alternative T-norm operators;
- Add GPU acceleration for ANFIS training;
- Compare against traditional neural networks;
- Create visualization dashboards for fuzzy rules;
- Evaluate additional nonlinear dynamic benchmarks.

---

# 📚 References

- Wang, L. X., & Mendel, J. M. (1992). Generating fuzzy rules by learning from examples.
- Jang, J. S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System.
- Narendra, K. S., & Li, C. (1996). Identification and control of dynamical systems using neural networks.
