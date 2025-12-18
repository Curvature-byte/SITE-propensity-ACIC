from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

Mode = Literal["logistic", "mlp"]


class PropensityDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.x = torch.from_numpy(features).float()
        self.t = torch.from_numpy(targets.astype(np.float32)).float().unsqueeze(1)

    def __len__(self) -> int:  # pragma: no cover
        return self.x.shape[0]

    def __getitem__(self, idx: int):  # pragma: no cover
        return self.x[idx], self.t[idx]


class LogisticPropensity(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPPropensity(nn.Module):
    def __init__(self, dim: int, hidden: int, depth: int, dropout: float, activation: str):
        super().__init__()
        layers = []
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        in_dim = dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PropensityArtifacts:
    model: nn.Module
    scaler: StandardScaler
    device: torch.device
    mode: Mode
    #预测倾向评分
    def predict_proba(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        feats = self.scaler.transform(features)
        preds = []
        with torch.no_grad():
            for start in range(0, feats.shape[0], batch_size):
                batch = torch.from_numpy(feats[start : start + batch_size]).float().to(self.device)
                logits = self.model(batch)
                prob = torch.sigmoid(logits).cpu().numpy()
                preds.append(prob)
        return np.vstack(preds)[:, 0]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "propensity.pt")
        joblib.dump(self.scaler, directory / "scaler.pkl")
        (directory / "meta.txt").write_text(self.mode, encoding="utf-8")

    @staticmethod
    def load(directory: str | Path, input_dim: int, mode: Mode = "logistic", **kwargs) -> "PropensityArtifacts":
        directory = Path(directory)
        scaler = joblib.load(directory / "scaler.pkl")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(mode, input_dim, **kwargs).to(device)
        state = torch.load(directory / "propensity.pt", map_location=device)
        model.load_state_dict(state)
        return PropensityArtifacts(model=model, scaler=scaler, device=device, mode=mode)


def _build_model(mode: Mode, dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.1, activation: str = "relu") -> nn.Module:
    if mode == "logistic":
        return LogisticPropensity(dim)
    return MLPPropensity(dim, hidden, depth, dropout, activation)

#计算类别权重
def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    pos = float(labels.sum())
    neg = float(labels.shape[0] - pos)
    if pos == 0 or neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def propensity_score_training(
    data: np.ndarray,
    label: np.ndarray,
    mode: Mode = "logistic",
    test_size: float = 0.3,
    random_state: int = 42,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    depth: int = 2,
    dropout: float = 0.1,
    activation: str = "relu",
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, PropensityArtifacts]:
    """PyTorch drop-in replacement for ``simi_ite.propensity_score_training``."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, eval_x, train_t, eval_t = train_test_split(data, label, test_size=test_size, random_state=random_state)

    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    #对于测试集，我们必须使用从训练集学到的均值和标准差来进行标准化。
    eval_x = scaler.transform(eval_x)
    data_scaled = scaler.transform(data)

    dataset = PropensityDataset(train_x, train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _build_model(mode, train_x.shape[1], hidden, depth, dropout, activation).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_compute_class_weights(train_t).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_t in loader:
            batch_x = batch_x.to(device)
            batch_t = batch_t.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probs_eval = _predict_tensor(model, eval_x, device)
    preds_eval = (probs_eval > 0.5).astype(np.int32)
    acc_eval = accuracy_score(eval_t, preds_eval)
    f1_eval = f1_score(eval_t, preds_eval)

    probs_train = _predict_tensor(model, train_x, device)
    preds_train = (probs_train > 0.5).astype(np.int32)
    acc_train = accuracy_score(train_t, preds_train)
    f1_train = f1_score(train_t, preds_train)

    print(f"Train Acc: {acc_train:.4f} | Train F1: {f1_train:.4f}")
    print(f"Eval  Acc: {acc_eval:.4f} | Eval  F1: {f1_eval:.4f}")

    probs_all = _predict_tensor(model, data_scaled, device)
    artifacts = PropensityArtifacts(model=model, scaler=scaler, device=device, mode=mode)
    return probs_all, artifacts


def _predict_tensor(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            outputs.append(prob)
    return np.vstack(outputs)[:, 0]


def load_propensity_score(model_dir: str | Path, input_dim: Optional[int] = None, mode: Mode = "logistic", **kwargs) -> np.ndarray:
    """Helper that mirrors the legacy ``load_propensity_score`` API."""

    directory = Path(model_dir)
    if input_dim is None:
        raise ValueError("input_dim must be provided to rebuild the network architecture")
    artifacts = PropensityArtifacts.load(directory, input_dim=input_dim, mode=mode, **kwargs)
    return artifacts


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2048, 20)).astype(np.float32)
    w = rng.normal(size=(20,))
    logits = X.dot(w)
    T = (logits + rng.normal(scale=0.5, size=logits.shape) > 0).astype(np.float32)
    probs, artifacts = propensity_score_training(X, T, mode="mlp", epochs=10)
    print("First five probs:", probs[:5])