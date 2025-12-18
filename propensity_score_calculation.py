import numpy as np
from simi_ite.propensity import propensity_score_training, load_propensity_score
import pandas as pd
import matplotlib.pyplot as plt
# rng = np.random.default_rng(42)
# X = rng.normal(size=(2048, 20)).astype(np.float32)
# w = rng.normal(size=(20,))
# logits = X.dot(w)
# T = (logits + rng.normal(scale=0.5, size=logits.shape) > 0).astype(np.float32)

df = pd.read_csv("Datasets/ACIC/traineval.csv",header=None)
# print("Data shape:", df.shape)
print("First five rows:\n", df.head(10))

treatment_col =  51
T = df.iloc[:, treatment_col].values.astype(np.float32)
X = df.drop([0,51,52,53], axis=1).values.astype(np.float32)
print(f"Features (X) shape: {X.shape}")
print(f"Treatment (T) shape: {T.shape}")
probs, artifacts = propensity_score_training(X, T, mode="mlp", epochs=10)
artifacts.save("./simi_ite/ACIC/propensity_model")
print("First five probs:", probs[:5])
# plt.hist(probs[T==0], bins=50, alpha=0.5, label='Control (T=0)')
# plt.hist(probs[T==1], bins=50, alpha=0.5, label='Treated (T=1)')
# plt.legend()
# plt.title("Propensity Score Distribution")
# plt.xlabel("Propensity Score")
# plt.show()