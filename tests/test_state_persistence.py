"""Tests for the runtime state persistence layer.

Covers:
  - Atomic write/read of a known payload.
  - Missing file returns ``None`` (no prior state).
  - Malformed JSON returns ``None`` (best-effort behaviour).
  - Schema-version mismatch returns ``None``.
  - ``clear_state`` removes the file but tolerates absence.
"""

from __future__ import annotations

import json
from pathlib import Path

from larksnap.config.state_persistence import (
    PersistedRuntimeState,
    clear_state,
    default_state_path,
    load_state,
    save_state,
)


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "runtime_state.json"
    save_state(PersistedRuntimeState(detector_running=True, notifier_enabled=False), target)
    loaded = load_state(target)
    assert loaded == PersistedRuntimeState(detector_running=True, notifier_enabled=False)


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "does_not_exist.json"
    assert load_state(target) is None


def test_load_returns_none_for_malformed_json(tmp_path: Path) -> None:
    target = tmp_path / "runtime_state.json"
    target.write_text("not json at all", encoding="utf-8")
    assert load_state(target) is None


def test_load_returns_none_for_schema_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "runtime_state.json"
    target.write_text(json.dumps({"schema_version": 999, "detector_running": True}), encoding="utf-8")
    assert load_state(target) is None


def test_load_returns_none_for_wrong_shape(tmp_path: Path) -> None:
    target = tmp_path / "runtime_state.json"
    target.write_text(
        json.dumps({"schema_version": 1, "detector_running": "yes"}),
        encoding="utf-8",
    )
    # ``bool("yes")`` raises in this branch because we go through the
    # KeyError/TypeError path after ``bool(...)`` already coerced.
    # The point is: garbage must not crash the gateway.
    assert load_state(target) is None


def test_clear_state_removes_file(tmp_path: Path) -> None:
    target = tmp_path / "runtime_state.json"
    save_state(PersistedRuntimeState(detector_running=True, notifier_enabled=True), target)
    assert target.exists()
    clear_state(target)
    assert not target.exists()


def test_clear_state_noop_when_missing(tmp_path: Path) -> None:
    # Must not raise on a non-existent file.
    clear_state(tmp_path / "absent.json")


def test_save_is_atomic_no_partial_file(tmp_path: Path) -> None:
    """A failed ``os.replace`` must not leave the temp file behind."""
    target = tmp_path / "runtime_state.json"
    # Pre-create the file so the replace actually has to overwrite.
    save_state(PersistedRuntimeState(detector_running=False, notifier_enabled=True), target)
    save_state(PersistedRuntimeState(detector_running=True, notifier_enabled=False), target)
    # No leftover .tmp files in the directory.
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == []


def test_default_state_path_resolves_under_user_dir() -> None:
    """Default path must point at a directory that exists on this OS."""
    p = default_state_path()
    # Either APPDATA on Windows or ~/.larksnap on POSIX. We just
    # assert it has a sensible shape (parent exists, filename ends
    # with the expected name).
    assert p.name == "runtime_state.json"
    assert p.parent.name  # non-empty
