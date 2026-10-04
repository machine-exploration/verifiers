"""Generate a corpus of small training runs.

Each run trains a 2-layer ReLU MLP (the student) with Adam on a teacher-student
regression task. Configs vary width, learning rate, data size, input dim and label
noise. Every run logs test loss and gradient norm at fixed checkpoints, so a
forecaster can read the start of a run and predict how it ends.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STEPS = 400
EVERY = 10
BATCH = 32
N_TEST = 2000
LOSS_CAP = 1e3


def sample_config(rng: np.random.Generator, run_id: int) -> dict:
    return {
        "id": run_id,
        "width": round(float(2 ** rng.uniform(3, 8))),  # 8 .. 256
        "lr": float(10 ** rng.uniform(-4, -0.5)),
        "n_train": round(float(2 ** rng.uniform(6, 12))),  # 64 .. 4096
        "dim": round(float(2 ** rng.uniform(2, 5))),  # 4 .. 32
        "noise": float(rng.uniform(0.0, 0.5)),
        "seed": int(rng.integers(2**31)),
    }


def _teacher(rng, dim):
    w1 = rng.normal(size=(dim, 16)) / np.sqrt(dim)
    w2 = rng.normal(size=(16, 1)) / 4.0
    return lambda x: np.tanh(x @ w1) @ w2


def train(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    d, w, n = cfg["dim"], cfg["width"], cfg["n_train"]
    teacher = _teacher(rng, d)
    x_tr, x_te = rng.normal(size=(n, d)), rng.normal(size=(N_TEST, d))
    y_tr = teacher(x_tr) + cfg["noise"] * rng.normal(size=(n, 1))
    y_te = teacher(x_te) + cfg["noise"] * rng.normal(size=(N_TEST, 1))

    params = {
        "w1": rng.normal(size=(d, w)) * np.sqrt(2 / d),
        "b1": np.zeros(w),
        "w2": rng.normal(size=(w, 1)) * np.sqrt(1 / w),
        "b2": np.zeros(1),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    def forward(p, x):
        h = np.maximum(x @ p["w1"] + p["b1"], 0)
        return h, h @ p["w2"] + p["b2"]

    def test_loss(p):
        return float(np.mean((forward(p, x_te)[1] - y_te) ** 2))

    curve, grads, diverged = [test_loss(params)], [], False
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, n, size=BATCH)
        x, y = x_tr[idx], y_tr[idx]
        h, out = forward(params, x)
        g_out = 2 * (out - y) / BATCH
        g = {
            "w2": h.T @ g_out,
            "b2": g_out.sum(0),
        }
        g_h = (g_out @ params["w2"].T) * (h > 0)
        g["w1"], g["b1"] = x.T @ g_h, g_h.sum(0)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            m_hat, v_hat = m[k] / (1 - b1**step), v[k] / (1 - b2**step)
            params[k] -= cfg["lr"] * m_hat / (np.sqrt(v_hat) + eps)
        if step % EVERY == 0:
            loss = test_loss(params)
            gnorm = float(np.sqrt(sum((gk**2).sum() for gk in g.values())))
            if not np.isfinite(loss) or loss > LOSS_CAP:
                diverged = True
                break
            curve.append(loss)
            grads.append(gnorm)
    return {
        **cfg,
        "curve": curve,
        "grad_norms": grads,
        "diverged": diverged,
        "final": LOSS_CAP if diverged else curve[-1],
    }


def main(n_runs: int, out: str, workers: int = 4) -> None:
    rng = np.random.default_rng(0)
    configs = [sample_config(rng, i) for i in range(n_runs)]
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool, open(path, "w") as f:
        for i, run in enumerate(pool.map(train, configs, chunksize=8)):
            f.write(json.dumps(run) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{n_runs} runs", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
