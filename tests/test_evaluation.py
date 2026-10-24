"""Testes das métricas de avaliação com incerteza e por fatia."""

from __future__ import annotations

import numpy as np

from src.evaluation.classification import (
    bootstrap_f1_macro,
    evaluate_by_slice,
    evaluate_classification,
)


def test_perfect_prediction_has_f1_one():
    """Predição perfeita produz F1-macro igual a 1."""
    y = np.array(["Positivo", "Negativo", "Neutro", "Positivo"])
    report = evaluate_classification(y, y, bootstrap_iterations=50)
    assert report.f1_macro == 1.0
    assert report.f1_ci95[0] <= report.f1_macro <= report.f1_ci95[1] + 1e-9


def test_bootstrap_ci_is_ordered():
    """O limite inferior do IC não pode exceder o superior."""
    rng = np.random.default_rng(0)
    y_true = rng.choice(["Positivo", "Negativo", "Neutro"], size=200)
    y_pred = rng.choice(["Positivo", "Negativo", "Neutro"], size=200)
    lo, hi = bootstrap_f1_macro(y_true, y_pred, iterations=100)
    assert lo <= hi


def test_bootstrap_ci_matches_sklearn_resampling():
    """O IC vetorizado concorda com o bootstrap clássico via f1_score do sklearn."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    labels = ["Positivo", "Negativo", "Neutro"]
    y_true = rng.choice(labels, size=2000, p=[0.5, 0.3, 0.2])
    y_pred = np.where(rng.random(2000) < 0.7, y_true, rng.choice(labels, size=2000))
    lo, hi = bootstrap_f1_macro(y_true, y_pred, iterations=1000)

    scores = []
    for _ in range(300):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    ref_lo, ref_hi = np.percentile(scores, [2.5, 97.5])
    assert abs(lo - ref_lo) < 0.01
    assert abs(hi - ref_hi) < 0.01


def test_evaluate_by_slice_returns_score_per_group():
    """A avaliação por fatia retorna um F1 para cada valor de fatia."""
    y_true = np.array(["Positivo", "Negativo", "Positivo", "Negativo"])
    y_pred = np.array(["Positivo", "Negativo", "Negativo", "Negativo"])
    slices = np.array(["curto", "curto", "longo", "longo"])
    result = evaluate_by_slice(y_true, y_pred, slices)
    assert set(result.keys()) == {"curto", "longo"}
