"""Offline-first UCI banknote acquisition, validation and exact-record lineage."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

FEATURES: Final = ("variance", "skewness", "kurtosis", "entropy")
DUPLICATE_POLICY: Final = "exact_features_keep_first_v1"
FILENAME = "data_banknote_authentication.txt"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/267/banknote+authentication"
REFERENCE_SHA256 = "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9"
CITATION = (
    "Lohweg, V. (2012). Banknote Authentication [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55P57."
)
MAX_BYTES = 5 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write(path: Path, data: bytes, *, replace: bool = False) -> None:
    """Publish a complete file; by default even concurrent overwrites are refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class BanknoteDataset:
    features: pd.DataFrame
    target: pd.Series
    lineage: pd.DataFrame
    source_manifest: dict[str, Any]
    audit: dict[str, Any]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.features.index)


def sample_id(values: tuple[float, ...]) -> str:
    """Target-independent identity of normalized big-endian IEEE-754 doubles."""
    return sha256(struct.pack(">4d", *(0.0 if value == 0 else value for value in values)))


def load_banknote_dataset(
    path: Path,
    *,
    reference_sha256: str | None,
    canonical: bool = True,
    expected_raw_rows: int = 1372,
) -> BanknoteDataset:
    """Validate every row; alternate data must be explicitly marked noncanonical."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing banknote file: {path}. Provide a local file or fetch-banknotes.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Banknote input exceeds the 5 MiB limit")
    raw = path.read_bytes()
    digest = sha256(raw)
    if canonical and not reference_sha256:
        raise ValueError("Canonical data require a reviewed reference SHA-256")
    if reference_sha256 is not None and digest != reference_sha256:
        raise ValueError(f"Raw-data hash mismatch: expected {reference_sha256}, observed {digest}")
    records: dict[str, tuple[tuple[float, ...], int, int]] = {}
    lineage: list[dict[str, Any]] = []
    raw_counts = {"0": 0, "1": 0}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Banknote input must be UTF-8 text") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 5:
            raise ValueError(f"Line {line_number}: expected exactly five fields")
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: missing or nonnumeric field") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Line {line_number}: nonfinite value")
        if values[4] not in (0.0, 1.0):
            raise ValueError(f"Line {line_number}: target must be exactly 0 or 1")
        features = tuple(0.0 if value == 0 else value for value in values[:4])
        label = int(values[4])
        identity = sample_id(features)
        if identity in records and records[identity][1] != label:
            raise ValueError(
                f"Line {line_number}: conflicting labels for features from "
                f"line {records[identity][2]}"
            )
        records.setdefault(identity, (features, label, line_number))
        lineage.append(
            {
                "source_line": line_number,
                "sample_id": identity,
                "representative_line": records[identity][2],
            }
        )
        raw_counts[str(label)] += 1
    if not all(raw_counts.values()):
        raise ValueError("Banknote training data must contain both classes 0 and 1")
    if canonical and len(lineage) != expected_raw_rows:
        raise ValueError(f"Expected {expected_raw_rows} raw rows; observed {len(lineage)}")
    ids = sorted(records)
    frame = pd.DataFrame(
        [records[key][0] for key in ids], index=ids, columns=FEATURES, dtype="float64"
    )
    frame.index.name = "sample_id"
    target = pd.Series(
        [records[key][1] for key in ids], index=frame.index, name="target", dtype="int64"
    )
    lineage_frame = pd.DataFrame(lineage)
    groups = lineage_frame.groupby("sample_id")["source_line"].agg(list)
    source: dict[str, Any] = {
        "path": str(path),
        "raw_sha256": digest,
        "byte_count": len(raw),
        "dataset_url": DATASET_URL,
        "citation": CITATION,
        "license": "CC BY 4.0",
        "acquisition": "local_file",
        "retrieved_at_utc": None,
        "inspected_at_utc": utc_now(),
        "source_url": None,
        "canonical": canonical,
        "label_mapping_status": "unverified",
        "class_names": {"0": "class_0", "1": "class_1"},
    }
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if cached.get("raw_sha256") != digest or cached.get("byte_count") != len(raw):
            raise ValueError(
                "Cache provenance hash/size mismatch; preserve and inspect the snapshot"
            )
        source.update(cached)
        source.update(path=str(path), canonical=canonical)
    audit = {
        "schema_version": 1,
        "raw_rows": len(lineage),
        "retained_rows": len(ids),
        "removed_duplicates": len(lineage) - len(ids),
        "raw_class_counts": raw_counts,
        "retained_class_counts": {str(k): int(v) for k, v in target.value_counts().items()},
        "missingness": dict.fromkeys((*FEATURES, "target"), 0),
        "feature_ranges": {
            name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
            for name in FEATURES
        },
        "duplicate_policy": DUPLICATE_POLICY,
        "duplicate_groups": [
            {"sample_id": key, "source_lines": value}
            for key, value in groups.items()
            if len(value) > 1
        ],
        "lineage_artifact": "lineage.csv",
        "label_mapping_status": "unverified",
        "canonical": canonical,
        "summary_usage": "Full-population ranges are audit diagnostics only, never model selection",
    }
    return BanknoteDataset(frame, target, lineage_frame, source, audit)


def _download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data: bytes = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("Download exceeds the 5 MiB response limit")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise ValueError(f"Official UCI download failed: {exc}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise ValueError(f"Official UCI download failed: {exc}") from exc
        time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def fetch_banknote_dataset(
    cache_dir: Path,
    *,
    reference_sha256: str = REFERENCE_SHA256,
) -> Path:
    """Explicitly fetch the official archive, or verify a complete cache offline."""
    destination = Path(cache_dir) / FILENAME
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or sidecar.exists():
        if not destination.is_file() or not sidecar.is_file():
            raise ValueError(
                "Incomplete cache; use a new cache directory, preserving this snapshot"
            )
        load_banknote_dataset(destination, reference_sha256=reference_sha256)
        return destination
    archive = _download(SOURCE_URL)
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            members = [member for member in zipped.infolist() if member.filename == FILENAME]
            if len(members) != 1 or members[0].file_size > MAX_BYTES:
                raise ValueError(
                    "Archive must contain one bounded, explicitly named banknote member"
                )
            with zipped.open(members[0]) as stream:
                raw = stream.read(MAX_BYTES + 1)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("Invalid official UCI archive") from exc
    if len(raw) > MAX_BYTES or sha256(raw) != reference_sha256:
        raise ValueError("Official raw-data hash/size mismatch; a reviewed new version is required")
    manifest = {
        "source_url": SOURCE_URL,
        "retrieved_at_utc": utc_now(),
        "raw_sha256": sha256(raw),
        "archive_sha256": sha256(archive),
        "byte_count": len(raw),
        "archive_byte_count": len(archive),
        "license": "CC BY 4.0",
        "citation": CITATION,
        "acquisition": "official_https_archive",
    }
    atomic_write(destination, raw)
    atomic_write(sidecar, json_bytes(manifest))
    load_banknote_dataset(destination, reference_sha256=reference_sha256)
    return destination
