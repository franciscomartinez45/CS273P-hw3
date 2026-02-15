"""
HW3 - Problem 1: Binary Logistic Regression (from scratch)

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training in this homework.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np


def load_iris_binary(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load iris data from `path` and create a binary dataset using ONLY classes {0,1}.

    Keep only samples with labels 0 or 1.
    Map label 0 -> 0 and label 1 -> 1.

    Returns
    -------
    X : np.ndarray, shape (N, 4)
        Standardized using only the retained subset.
    y : np.ndarray, shape (N,)
        Labels in {0,1}.
    """
    # TODO
    # raise NotImplementedError

    data = np.genfromtxt(path)
    X = data[:, :4]
    y = data [:, 4]

    mask = y <= 1
    X = X[mask]
    y = y[mask]

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X,y




def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid, elementwise.

    Returns
    -------
    s : np.ndarray, same shape as z
    """
    # TODO
    # raise NotImplementedError
    pos = z >=0
    # print(pos)
    s = np.empty_like(z, dtype=float)
    s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = np.exp(z[~pos])
    s[~pos] = neg / (1.0 + neg)
    return s



def logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg: float = 0.0) -> float:
    """
    Average negative log-likelihood with optional L2 regularization on w[1:].

    Model: p(y=1|x) = sigmoid(w0 + x @ w[1:])

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {0,1}
    w : (d+1,) bias is w[0]
    reg : nonnegative float

    Returns
    -------
    loss : float
    """
    # TODO
    # raise NotImplementedError
    n = X.shape[0]
    z = w[0] + X @ w[1:]
    p = sigmoid(z)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(p)+(1-y)*np.log(1-p))
    loss += reg * np.sum(w[1:] ** 2)
    return loss


def logistic_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    Gradient of logistic_loss w.r.t. w.

    Returns
    -------
    grad : np.ndarray, shape (d+1,)
    """
    # TODO
    # raise NotImplementedError

    n = X.shape[0]
    z = w[0] + X @ w[1:]
    p = sigmoid(z)
    error = p - y 
    grad = np.zeros_like(w)
    grad[0] = np.mean(error)
    grad[1:] = (X.T @ error) / n + 2 * reg * w[1:]
    return grad 


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Return p(y=1|x) for each row of X.

    Returns
    -------
    p : np.ndarray, shape (N,)
    """
    # TODO
    # raise NotImplementedError
    return sigmoid(w[0] + X @ w[1:])



def predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Return predicted labels in {0,1} using threshold on predicted probability.

    Returns
    -------
    yhat : np.ndarray, shape (N,)
    """
    # TODO
    # raise NotImplementedError
    yhat = predict_proba(X,w)
    return (yhat >= threshold).astype(float)


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    step_size: float = 0.1,
    max_epochs: int = 2000,
    tol: float = 1e-6,
    batch_size: int = 0,
    reg: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train logistic regression starting from w = zeros(d+1).

    If batch_size == 0: full-batch GD (one gradient step per epoch).
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(loss_t - loss_{t-1}) < tol (loss computed on full data).

    Returns dict with at least:
      - "w": np.ndarray (d+1,)
      - "loss_history": list[float] (or np.ndarray)
      - "err_history": list[float] training error rate (0..1)
      - "epochs": int
    """
    # TODO
    # raise NotImplementedError
    n, d = X.shape
    w = np.zeros(d+1)
    loss_history = []
    err_history = []
    rng = np.random.RandomState(seed)

    for epoch in range(max_epochs):
        if batch_size == 0:
            grad = logistic_grad(X,y,w,reg)
            w -= step_size * grad
        else:
            idx = np.arange(n)
            rng.shuffle(idx)
            for start in range(0, n, batch_size):
                Xb = X[idx[start:start + batch_size]]
                yb = y[idx[start:start + batch_size]]
                grad = logistic_grad(Xb, yb, w, reg)
                w -= step_size * grad
        loss = logistic_loss(X,y, w, reg)
        yhat = predict(X,w)
        err = np.mean(yhat!=y)
        loss_history.append(loss)
        err_history.append(err)

        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return{
        "w": w,
        "loss_history": loss_history,
        "err_history" : err_history,
        "epochs" : epoch + 1
    }


if __name__ == "__main__":
    # Quick self-check (not graded)
    try:
        X, y = load_iris_binary("data/iris.txt")
        out = train_logreg(X, y, step_size=0.2, max_epochs=2000, tol=1e-9, batch_size=0, reg=1e-3, seed=0)
        w = out["w"]
        p = predict_proba(X, w)
        yhat = predict(X, w)
        print("X shape:", X.shape, "y mean:", float(np.mean(y)))
        print("Final loss:", out["loss_history"][-1], "final error:", out["err_history"][-1])
        print("Prob range:", float(np.min(p)), float(np.max(p)))
        print("Accuracy:", float(np.mean(yhat == y)))
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
