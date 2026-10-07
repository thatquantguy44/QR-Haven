"""Frozen experiment, leakage, selection, metrics, persistence and CLI integrity."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn", reason="Install .[research] for classification tests")
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from qr_haven.data.banknotes import FEATURES, load_banknote_dataset, sha256
from qr_haven.ml.classification import (
    ClassificationConfig,
    artifacts,
    evaluate_banknote_run,
    load_classifier,
    prepare_banknote_split,
    train_banknote_classifier,
    training,
)
from qr_haven.ml.classification.__main__ import main
from qr_haven.ml.classification.contracts import EvaluationConfig
from qr_haven.ml.classification.evaluation import (
    classification_metrics,
    numerical_gates,
    wilson_interval,
)
from qr_haven.ml.classification.models import candidates, predict_frame
from qr_haven.ml.classification.splits import validate_split


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(80, 4))
    labels = np.tile([0, 1], 40)
    features[:, 0] += labels * 2
    path = tmp_path / "fixture.txt"
    np.savetxt(path, np.column_stack([features, labels]), delimiter=",")
    return load_banknote_dataset(path, reference_sha256=None, canonical=False)


@pytest.fixture
def config():
    return ClassificationConfig.model_validate(
        {
            "models": {
                "logistic_regression": {"C": [1.0]},
                "svm": {"C": [1.0], "gamma": ["scale"]},
                "random_forest": {"n_estimators": 5, "max_depth": [None], "min_samples_leaf": [1]},
            },
        }
    )


@pytest.fixture
def trained(tmp_path, dataset, config):
    run = tmp_path / "artifacts" / "fixture"
    split = prepare_banknote_split(dataset, config)
    result = train_banknote_classifier(dataset, split, config, run)
    return run, split, result


def test_config_exact_grid_and_rejected_settings():
    config = ClassificationConfig()
    assert config.canonical_protocol()
    assert len(candidates(config)) == 29
    assert [
        len([c for c in candidates(config) if c.family == f])
        for f in ("logistic_regression", "svm", "random_forest")
    ] == [4, 16, 9]


@pytest.mark.parametrize(
    "settings",
    [
        {"schema_version": 2},
        {"unknown": 1},
        {"split": {"typo": 1}},
        {"split": {"seed": -1}},
        {"split": {"seed": True}},
        {"split": {"seed": 2**32}},
        {"split": {"test_size": 1}},
        {"split": {"test_size": float("nan")}},
        {"split": {"cv_folds": 1}},
        {"models": {"svm": {"probability": True}}},
        {"models": {"svm": {"C": []}}},
        {"models": {"logistic_regression": {"C": [-1]}}},
        {"dataset": {"feature_order": ["entropy", "skewness", "kurtosis", "variance"]}},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        ClassificationConfig.model_validate(settings)


def test_split_membership_replays_and_folds_cover_once(dataset, config):
    split = prepare_banknote_split(dataset, config)
    assert split == prepare_banknote_split(dataset, config)
    assert set(split.development_ids).isdisjoint(split.test_ids)
    assert set(split.development_ids) | set(split.test_ids) == set(dataset.sample_ids)
    assert set(split.validation_folds) == set(split.development_ids)
    for fold in range(5):
        valid = [key for key, assignment in split.validation_folds.items() if assignment == fold]
        assert set(dataset.target.loc[valid]) == {0, 1}
    with pytest.raises(ValueError, match="disjoint"):
        validate_split(dataset, replace(split, test_ids=split.development_ids), 5)
    with pytest.raises(ValueError, match="one fold"):
        validate_split(dataset, replace(split, validation_folds={}), 5)


def test_too_few_per_class(tmp_path, config):
    path = tmp_path / "tiny.txt"
    path.write_text("\n".join(f"{i},1,2,3,{i % 2}" for i in range(8)))
    data = load_banknote_dataset(path, reference_sha256=None, canonical=False)
    with pytest.raises(ValueError, match="Too few"):
        prepare_banknote_split(data, config)


def _scores(name, family, ba, f1, order=0, status="ok"):
    return [
        {
            "candidate_id": name,
            "family": family,
            "grid_order": order,
            "fold": fold,
            "balanced_accuracy": ba,
            "f1_macro": f1,
            "accuracy": ba,
            "status": status,
        }
        for fold in range(5)
    ]


def test_unrounded_ranking_ties_and_failures():
    rows = (
        _scores("svm", "svm", 0.9, 0.8)
        + _scores("lr1", "logistic_regression", 0.9, 0.8, 1)
        + _scores("lr0", "logistic_regression", 0.9, 0.8)
        + _scores("forest", "random_forest", 0.9, 0.81)
        + _scores("best", "svm", 0.9000000000001, 0.1)
        + _scores("failed", "svm", 1, 1, status="failed")
        + _scores("baseline", "baseline", 1, 1)
        + _scores("incomplete", "svm", 1, 1)[:4]
    )
    assert [r["candidate_id"] for r in training.rank_candidates(rows, 5)] == [
        "best",
        "forest",
        "lr0",
        "lr1",
        "svm",
    ]


def test_scaler_fit_never_sees_test_or_fold_validation(dataset, config, tmp_path, monkeypatch):
    split = prepare_banknote_split(dataset, config)
    observed = []
    original = StandardScaler.fit
    model_fits = []
    original_fit = training._fit

    def record_model(estimator, X, y):
        model_fits.append(set(X.index))
        return original_fit(estimator, X, y)

    def record(self, X, y=None, **kwargs):
        observed.append(set(X.index))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record)
    monkeypatch.setattr(training, "_fit", record_model)
    train_banknote_classifier(dataset, split, config, tmp_path / "leakage")
    for i, seen in enumerate(observed[:10]):
        fold = i % 5
        expected = {key for key in split.development_ids if split.validation_folds[key] != fold}
        assert seen == expected
        assert seen.isdisjoint(split.test_ids)
    assert all(seen == set(split.development_ids) for seen in observed[10:])
    assert len(model_fits) == 22  # four candidates x five folds, plus two final fits
    for i, seen in enumerate(model_fits[:20]):
        assert seen == {
            key for key in split.development_ids if split.validation_folds[key] != i % 5
        }
    assert model_fits[20:] == [set(split.development_ids)] * 2


def test_changing_test_labels_cannot_change_selection_or_development_fit(
    dataset,
    config,
    tmp_path,
):
    split = prepare_banknote_split(dataset, config)
    changed = replace(dataset, target=dataset.target.copy())
    changed.target.loc[list(split.test_ids)] = 1 - changed.target.loc[list(split.test_ids)]
    first = train_banknote_classifier(dataset, split, config, tmp_path / "first")
    second = train_banknote_classifier(changed, split, config, tmp_path / "second")
    assert first.selection["chosen"] == second.selection["chosen"]
    assert first.selection["ranking"] == second.selection["ranking"]
    a = load_classifier(tmp_path / "first").predict(
        dataset.features.loc[list(split.development_ids)]
    )
    b = load_classifier(tmp_path / "second").predict(
        dataset.features.loc[list(split.development_ids)]
    )
    pd.testing.assert_frame_equal(a, b)


def test_metrics_orientation_scores_wilson_and_undefined_diagnostics():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    predictions = pd.DataFrame(
        {"predicted_class": [0, 0, 1, 0, 1, 1], "score_class_1": [0.1, 0.2, 0.6, 0.4, 0.8, 0.9]}
    )
    result = classification_metrics(y, predictions)
    assert result["accuracy"] == 4 / 6
    assert result["balanced_accuracy"] == 2 / 3
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert result["per_class"]["1"] == {
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
        "support": 3,
    }
    assert result["roc_auc"] == pytest.approx(8 / 9)
    assert result["average_precision"] == pytest.approx((1 + 1 + 0.75) / 3)
    assert wilson_interval(90, 100) == pytest.approx([0.8256343384950865, 0.9447708629393249])
    one_class = classification_metrics(y.iloc[:3], predictions.iloc[:3])
    assert one_class["roc_auc"] is None and one_class["average_precision"] is None
    json.dumps(one_class, allow_nan=False)
    predictions["predicted_class"] = 0
    assert "zero predicted support" in classification_metrics(y, predictions)["diagnostics"][0]


def test_gates_use_exact_unrounded_boundaries():
    winner = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.9,
        "per_class": {"0": {"recall": 0.85}, "1": {"recall": 0.85}},
    }
    baseline = {"accuracy": 0.75}
    assert all(numerical_gates(winner, baseline, EvaluationConfig()).values())
    assert numerical_gates(winner, {"accuracy": 0.8}, EvaluationConfig())["accuracy_gain"]
    assert numerical_gates(
        {**winner, "correct": 9, "total": 10},
        {"accuracy": 0.8, "correct": 8, "total": 10},
        EvaluationConfig(),
    )["accuracy_gain"]
    for metric in ("accuracy", "balanced_accuracy"):
        changed = copy.deepcopy(winner)
        changed[metric] = np.nextafter(0.9, 0)
        assert not numerical_gates(changed, baseline, EvaluationConfig())[metric]
    winner["per_class"]["1"]["recall"] = np.nextafter(0.85, 0)
    assert not numerical_gates(winner, baseline, EvaluationConfig())["class_1_recall"]
    gate_config = EvaluationConfig(minimum_accuracy_gain=0.25)
    winner["accuracy"] = 0.75
    assert numerical_gates(winner, {"accuracy": 0.5}, gate_config)["accuracy_gain"]
    winner["accuracy"] = np.nextafter(0.75, 0)
    assert not numerical_gates(winner, {"accuracy": 0.5}, gate_config)["accuracy_gain"]


def test_score_orientation_uses_classes_not_column_position():
    class Reversed:
        classes_ = np.array([1, 0])

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.9, 0.1], (len(X), 1))

    class Margin:
        classes_ = np.array([1, 0])

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def decision_function(self, X):
            return np.full(len(X), -2.0)

    frame = pd.DataFrame([[1, 2, 3, 4]], columns=FEATURES)
    assert predict_frame(Reversed(), frame)["score_class_1"].iloc[0] == 0.9
    result = predict_frame(Margin(), frame)
    assert result["score_class_1"].iloc[0] == 2
    assert result["score_kind"].iloc[0] == "decision_margin"


def test_persistence_feature_order_validation_and_no_fitting(trained, dataset, monkeypatch):
    run, split, result = trained
    assert result.manifest["state"] == "trained"
    assert not (run / "metrics.json").exists()
    cv_predictions = pd.read_csv(run / "cv_predictions.csv")
    assert cv_predictions.groupby(["candidate_id", "sample_id"]).size().eq(1).all()
    assert set(cv_predictions.sample_id) == set(split.development_ids)
    model = load_classifier(run)
    features = dataset.features.iloc[:4].copy()
    features.index = pd.Index([8, 2, 2, 4], name="input_id")
    expected = model.predict(features)

    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not fit")

    monkeypatch.setattr(model.estimator, "fit", forbidden)
    pd.testing.assert_frame_equal(expected, model.predict(features[list(reversed(FEATURES))]))
    pd.testing.assert_frame_equal(expected, load_classifier(run).predict(features))
    for invalid in (
        features.drop(columns="entropy"),
        features.assign(target=0),
        pd.concat([features, features[["entropy"]]], axis=1),
        features.assign(variance=np.inf),
    ):
        with pytest.raises(ValueError):
            model.predict(invalid)
    outside = features.copy()
    outside["variance"] = 1e5
    assert model.predict(outside).attrs["out_of_source_range_counts"]["variance"] == 4


def test_evaluation_repeat_preserves_evidence_and_reused_holdout_is_exploratory(
    trained,
    dataset,
    config,
):
    run, split, _ = trained
    first = evaluate_banknote_run(run, dataset)
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    history = (run.parent / "evaluation_history.jsonl").read_bytes()
    second = evaluate_banknote_run(run, dataset)
    assert first.metrics == second.metrics
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    assert history == (run.parent / "evaluation_history.jsonl").read_bytes()
    replay = run.parent / "replay"
    train_banknote_classifier(dataset, split, config, replay)
    reused = evaluate_banknote_run(replay, dataset)
    assert reused.metrics["benchmark_status"] == "exploratory"
    assert len(reused.metrics["prior_exposures"]) == 1
    assert any("previously exposed" in reason for reason in reused.metrics["reasons"])
    assert all(
        (run / name).exists() for name in ["report.md", "metrics.json", "test_predictions.csv"]
    )


def test_overwrite_invalid_state_corruption_and_version_guards(
    trained, dataset, config, monkeypatch
):
    run, split, _ = trained
    with pytest.raises(FileExistsError):
        train_banknote_classifier(dataset, split, config, run)
    original_versions = artifacts.versions()
    monkeypatch.setattr(
        artifacts, "versions", lambda: {**original_versions, "scikit-learn": "wrong"}
    )
    with pytest.raises(ValueError, match="version mismatch"):
        load_classifier(run)
    monkeypatch.undo()
    changed = replace(dataset, target=1 - dataset.target)
    with pytest.raises(ValueError, match="modified after parsing"):
        evaluate_banknote_run(run, changed)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = "prepared"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid experiment state"):
        evaluate_banknote_run(run, dataset)
    manifest["state"] = "trained"
    manifest_path.write_text(json.dumps(manifest))
    with (run / "model.pkl").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_classifier(run)


def test_failure_and_nonconvergence_exclude_candidates_and_invalidate_run(
    dataset,
    config,
    tmp_path,
    monkeypatch,
):
    def fail(*args):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(training, "make_estimator", fail)
    split = prepare_banknote_split(dataset, config)
    run = tmp_path / "failed"
    with pytest.raises(ValueError, match="Every learned candidate failed"):
        train_banknote_classifier(dataset, split, config, run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["state"] == "invalid"
    assert not (run / "model.pkl").exists()
    assert pd.read_csv(run / "cv_results.csv").failure.str.contains("fixture failure").all()

    class Nonconverged:
        def fit(self, X, y):
            warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)

    with pytest.raises(ValueError, match="Nonconverged fit"):
        training._fit(Nonconverged(), dataset.features, dataset.target)


def test_cli_local_train_evaluate_predict_and_exit_codes(tmp_path, dataset, config, capsys):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"raw_sha256": dataset.source_manifest["raw_sha256"]}))
    settings = config.model_dump(mode="json")
    settings["dataset"].update(
        path=dataset.source_manifest["path"],
        reference_manifest=str(reference),
        expected_raw_rows=80,
    )
    settings["artifacts"]["root"] = str(tmp_path / "runs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    assert main(["train", "--config", str(config_path), "--run-id", "fixture"]) == 0
    assert "Next:" in capsys.readouterr().out
    run = tmp_path / "runs" / "fixture"
    assert main(["evaluate", "--run-dir", str(run)]) == 2  # deliberately noncanonical fixture
    assert "exploratory" in capsys.readouterr().out
    features = tmp_path / "features.csv"
    output = tmp_path / "predictions.csv"
    dataset.features.iloc[:3].to_csv(features, index=False)
    args = ["predict", "--run-dir", str(run), "--input", str(features), "--output", str(output)]
    assert main(args) == 0
    assert len(pd.read_csv(output)) == 3
    original_hash = sha256(output.read_bytes())
    assert main(args) == 1
    assert sha256(output.read_bytes()) == original_hash
    assert main(["train", "--config", str(config_path), "--run-id", "../unsafe"]) == 1


def test_base_ml_import_without_sklearn():
    code = """
import sys
from importlib.abc import MetaPathFinder
class NoSklearn(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ModuleNotFoundError("sklearn intentionally unavailable")
sys.meta_path.insert(0, NoSklearn())
import qr_haven
from qr_haven.ml import LSTMReturnForecaster
from qr_haven.ml.classification.contracts import require_sklearn
assert "sklearn" not in sys.modules
try:
    require_sklearn()
except ImportError as error:
    assert '.[research]' in str(error)
else:
    raise AssertionError("Missing optional dependency was not diagnosed")
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_modified_development_data_and_reassigned_folds_rejected(dataset, config, tmp_path):
    split = prepare_banknote_split(dataset, config)
    changed = replace(dataset, target=dataset.target.copy())
    first = split.development_ids[0]
    changed.target.loc[first] = 1 - changed.target.loc[first]
    with pytest.raises(ValueError, match="Development data differ"):
        train_banknote_classifier(changed, split, config, tmp_path / "changed")
    folds = {key: (value + 1) % 5 for key, value in split.validation_folds.items()}
    with pytest.raises(ValueError, match="Split membership differs"):
        train_banknote_classifier(
            dataset, replace(split, validation_folds=folds), config, tmp_path / "folds"
        )


@pytest.mark.parametrize("filename", ["split.csv", "config.resolved.yaml", "selection.json"])
def test_evaluation_rejects_frozen_artifact_tampering(trained, dataset, filename):
    run, _, _ = trained
    with (run / filename).open("ab") as stream:
        stream.write(b"\nmodified\n")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        evaluate_banknote_run(run, dataset)
    assert not (run.parent / "evaluation_history.jsonl").exists()


def test_interrupted_evaluation_records_exposure_and_invalid_state(
    trained,
    dataset,
    monkeypatch,
):
    from qr_haven.ml.classification import reporting

    run, _, _ = trained

    def fail_report(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(reporting, "write_report", fail_report)
    with pytest.raises(OSError, match="disk unavailable"):
        evaluate_banknote_run(run, dataset)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["state"] == "invalid"
    assert "failed after exposure" in manifest["failure"]
    assert len((run.parent / "evaluation_history.jsonl").read_text().splitlines()) == 1


def test_cli_malformed_yaml_and_duplicate_feature_headers(trained, tmp_path, capsys):
    config = tmp_path / "broken.yaml"
    config.write_text("models: [")
    assert main(["train", "--config", str(config), "--run-id", "broken"]) == 1
    assert "Error:" in capsys.readouterr().err
    run, _, _ = trained
    features = tmp_path / "duplicates.csv"
    features.write_text("variance,variance,kurtosis,entropy\n1,2,3,4\n")
    assert (
        main(
            [
                "predict",
                "--run-dir",
                str(run),
                "--input",
                str(features),
                "--output",
                str(tmp_path / "unused.csv"),
            ]
        )
        == 1
    )
    assert "Duplicate feature columns" in capsys.readouterr().err
