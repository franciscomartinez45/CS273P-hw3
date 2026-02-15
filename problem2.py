"""
HW3 - Problem 2: Softmax Regression (multiclass, from scratch)

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Stable softmax applied row-wise.

    Parameters
    ----------
    Z : np.ndarray, shape (N, K)

    Returns
    -------
    P : np.ndarray, shape (N, K)
        Each row sums to 1.
    """
    # TODO
    # raise NotImplementedError
    Z_max = np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z-Z_max)
    P = E / np.sum(E, axis=1, keepdims=True)
    return P



def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    """
    Convert y in {0,...,K-1} to one-hot.

    Returns
    -------
    Y : np.ndarray, shape (N, K)
    """
    # TODO
    # raise NotImplementedError

    N = y.shape[0]
    Y = np.zeros((N,K))
    Y[np.arange(N), y] = 1.0
    return Y


def softmax_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float = 0.0) -> float:
    """
    Average cross-entropy loss with L2 regularization on W[1:,:] (exclude bias row).

    Model logits:
      Z = b + X @ V
    where bias row is W[0,:] and weights are W[1:,:].

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {0,...,K-1}
    W : (d+1, K)
    reg : nonnegative float

    Returns
    -------
    loss : float
    """
    # TODO
    # raise NotImplementedError
    N = X.shape[0]
    Z = X @ W[1:, :] + W[0,:]
    P = softmax(Z)
    log = -np.log(P[np.arange(N), y]+1e-12)
    loss = np.mean(log) + (reg/2) * np.sum(W[1:,:]**2)
    return loss


def softmax_grad(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    Gradient of softmax_loss w.r.t. W.

    Returns
    -------
    grad : np.ndarray, shape (d+1, K)
    """
    # TODO
    # raise NotImplementedError

    N = X.shape[0]
    Z = X @W[1:, :] + W[0, :]
    P = softmax(Z)
    P[np.arange(N), y] -=1.0

    grad = np.zeros_like(W)
    grad[0, :] = np.mean(P, axis=0)
    grad[1:, :] = (X.T @ P) / N+2 * reg * W[1:,:]
    return grad



def predict_proba_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Return class probabilities for each row.

    Returns
    -------
    P : np.ndarray, shape (N, K)
    """
    # TODO
    # raise NotImplementedError

    Z = X @ W[1:, :] + W[0, :]
    P = softmax(Z)
    return P


def predict_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Return predicted class labels (argmax over probabilities/logits).

    Returns
    -------
    yhat : np.ndarray, shape (N,)
    """
    # TODO
    # raise NotImplementedError

    Z = X @ W[1:, :] + W[0, :]
    yhat = np.argmax(Z, axis=1)
    return yhat


def train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    step_size: float = 0.1,
    max_epochs: int = 3000,
    tol: float = 1e-6,
    batch_size: int = 0,
    reg: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train softmax regression from W = zeros(d+1, K).

    If batch_size == 0: full-batch GD per epoch.
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(loss_t - loss_{t-1}) < tol (loss computed on full data).

    Returns dict with at least:
      - "W": np.ndarray (d+1, K)
      - "loss_history": list[float]
      - "acc_history": list[float] training accuracy
      - "epochs": int
    """
    # TODO
    # raise NotImplementedError

    rng = np.random.RandomState(seed)
    N, d = X.shape
    W = np.zeros((d+1, K))
    loss_history = []
    acc_history = []

    for epoch in range(max_epochs):
        if batch_size > 0:
            index = np.arange(N)
            rng.shuffle(index)
            for start in range(0, N, batch_size):
                mini_b = index[start:start + batch_size]
                grad = softmax_grad(X[mini_b], y[mini_b], W, reg)
                W -= step_size * grad
        else:
            grad = softmax_grad(X, y,W,reg)
            W-= step_size * grad

        loss = softmax_loss(X,y,W, reg)
        yhat = predict_softmax(X, W)
        acc = float(np.mean(yhat ==y))

        loss_history.append(loss)
        acc_history.append(acc)

        if epoch > 0 and abs(loss_history[-1] - loss_history[-2])< tol:
            break
    return {
        "W": W,
        "loss_history" : loss_history,
        "acc_history" : acc_history,
        "epochs" : epoch + 1

    }

if __name__ == "__main__":
    # Quick self-check (not graded): load iris if present
    try:
        data = np.loadtxt("data/iris.txt")
        X = data[:, :4]
        y = data[:, 4].astype(int)

        # Standardize using all data for this demo
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        K = int(np.max(y)) + 1
        out = train_softmax(X, y, K=K, step_size=0.2, max_epochs=3000, tol=1e-9, batch_size=0, reg=1e-3, seed=0)
        W = out["W"]
        yhat = predict_softmax(X, W)
        print("Train acc:", float(np.mean(yhat == y)))
        print("Final loss:", out["loss_history"][-1], "epochs:", out["epochs"])
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
