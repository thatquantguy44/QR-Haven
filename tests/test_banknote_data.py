"""Offline data-contract tests. No accuracy claims or live downloads."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pytest

from qr_haven.data import banknotes as data


def load_text(tmp_path: Path, text: str):
    path = tmp_path / "banknotes.txt"
    path.write_text(text)
    return data.load_banknote_dataset(path, reference_sha256=None, canonical=False)


def test_headerless_identity_duplicate_lineage_and_signed_zero(tmp_path):
    dataset = load_text(tmp_path, "-0.0,-2,3,4,0\n\n0,-2,3,4,0\n1,2,-3,4,1\n")
    identity = hashlib.sha256(struct.pack(">4d", 0.0, -2.0, 3.0, 4.0)).hexdigest()
    assert dataset.sample_ids == tuple(sorted(dataset.sample_ids))
    assert dataset.features.dtypes.tolist() == [np.dtype("float64")] * 4
    assert dataset.target.loc[identity] == 0
    assert dataset.lineage["source_line"].tolist() == [1, 3, 4]
    assert dataset.lineage["representative_line"].tolist() == [1, 1, 4]
    assert dataset.lineage["sample_id"].tolist()[:2] == [identity, identity]
    assert dataset.audit["removed_duplicates"] == 1
    assert dataset.audit["raw_class_counts"] == {"0": 2, "1": 1}
    assert dataset.audit["label_mapping_status"] == "unverified"
    assert dataset.source_manifest["retrieved_at_utc"] is None


@pytest.mark.parametrize(
    "row,message",
    [
        ("1,2,3,0", "five fields"),
        ("1,2,3,4,0,6", "five fields"),
        ("1,,3,4,0", "nonnumeric"),
        ("1,hello,3,4,0", "nonnumeric"),
        ("variance,skewness,kurtosis,entropy,target", "nonnumeric"),
        ("nan,2,3,4,0", "nonfinite"),
        ("1,inf,3,4,0", "nonfinite"),
        ("1,2,3,4,-inf", "nonfinite"),
        ("1,2,3,4,0.5", "exactly 0 or 1"),
        ("1,2,3,4,2", "exactly 0 or 1"),
    ],
)
def test_invalid_rows_have_line_numbers(tmp_path, row, message):
    with pytest.raises(ValueError, match=f"Line 2:.*{message}"):
        load_text(tmp_path, f"1,2,3,4,0\n{row}\n2,3,4,5,1")


def test_conflicting_labels_and_one_class_rejected(tmp_path):
    with pytest.raises(ValueError, match="conflicting labels.*line 1"):
        load_text(tmp_path, "1,2,3,4,0\n1,2,3,4,1")
    with pytest.raises(ValueError, match="both classes"):
        load_text(tmp_path, "1,2,3,4,0")


def test_hash_and_population_validation(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("1,2,3,4,0\n2,3,4,5,1")
    with pytest.raises(ValueError, match="reviewed reference"):
        data.load_banknote_dataset(path, reference_sha256=None)
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_banknote_dataset(path, reference_sha256="0" * 64)
    with pytest.raises(ValueError, match="Expected 1372"):
        data.load_banknote_dataset(path, reference_sha256=data.sha256(path.read_bytes()))


def _archive(raw: bytes, name=data.FILENAME) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr(name, raw)
        zipped.writestr("../ignored.txt", "never extracted")
    return stream.getvalue()


def test_fetch_provenance_and_offline_cache(tmp_path, monkeypatch):
    raw = b"".join(f"{i},2,3,4,{i % 2}\n".encode() for i in range(1372))
    archive = _archive(raw)
    monkeypatch.setattr(data, "_download", lambda url: archive)
    path = data.fetch_banknote_dataset(tmp_path, reference_sha256=data.sha256(raw))
    assert path.read_bytes() == raw
    manifest = json.loads(path.with_suffix(".txt.manifest.json").read_text())
    assert manifest["archive_sha256"] == data.sha256(archive)
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["retrieved_at_utc"].endswith("+00:00")
    assert not (tmp_path.parent / "ignored.txt").exists()

    def no_network(url):
        pytest.fail("A verified cache must not fetch")

    monkeypatch.setattr(data, "_download", no_network)
    assert data.fetch_banknote_dataset(tmp_path, reference_sha256=data.sha256(raw)) == path
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.fetch_banknote_dataset(tmp_path, reference_sha256=data.sha256(raw))


def test_incomplete_cache_and_wrong_member_preserved(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / data.FILENAME).write_bytes(b"local snapshot")
    with pytest.raises(ValueError, match="Incomplete cache"):
        data.fetch_banknote_dataset(cache)
    monkeypatch.setattr(data, "_download", lambda url: _archive(b"data", "nested/" + data.FILENAME))
    with pytest.raises(ValueError, match="named banknote member"):
        data.fetch_banknote_dataset(tmp_path / "new")
    assert (cache / data.FILENAME).read_bytes() == b"local snapshot"


def test_response_limit_and_bounded_transient_retries(monkeypatch):
    calls = []

    def transient(url, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(url, 503, "temporary", {}, None)

    monkeypatch.setattr(data.urllib.request, "urlopen", transient)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="download failed"):
        data._download(data.SOURCE_URL)
    assert calls == [30, 30, 30]
    calls.clear()

    def permanent(url, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(url, 404, "missing", {}, None)

    monkeypatch.setattr(data.urllib.request, "urlopen", permanent)
    with pytest.raises(ValueError):
        data._download(data.SOURCE_URL)
    assert calls == [30]
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 11))
    monkeypatch.setattr(data, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="response limit"):
        data._download(data.SOURCE_URL)


def test_atomic_no_overwrite(tmp_path):
    path = tmp_path / "snapshot"
    data.atomic_write(path, b"original")
    with pytest.raises(FileExistsError):
        data.atomic_write(path, b"changed")
    assert path.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snapshot"]
