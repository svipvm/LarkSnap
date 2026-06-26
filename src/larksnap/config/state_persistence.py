"""Runtime state persistence for detector and notifier.

Captures the on/off state of the detector and notifier at
``close_camera`` time and replays it on the next ``open_camera`` so
the user returns to the same working setup, even across a full
process restart.

Design notes:
  - The file lives under the user data directory
    (``%APPDATA%/LarkSnap`` on Windows, ``~/.larksnap`` elsewhere) so
    the application runs portably without needing a config-file
    change.
  - The payload is a tiny JSON document, written atomically
    (write-to-temp + ``os.replace``) so a crash mid-write cannot
    leave a half-written file that would fail to parse on next read.
  - Reads are best-effort: a missing or malformed file is treated as
    "no prior state" and the controller falls back to its default
    initial behaviour. This keeps the persistence layer forgiving —
    the gateway must work even if the file is corrupted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

_logger = logging.getLogger("larksnap.state_persistence")

# Schema version. Bump when the payload changes shape so we can
# detect / migrate old files in the future.
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistedRuntimeState:
    """Snapshot of the user-facing on/off state at close time.

    ``detector_running`` reflects whether detection was active
    (``GatewayState.DETECTING``) when the camera was closed.
    ``notifier_enabled`` reflects whether notification dispatch was
    enabled when the camera was closed.
    """

    detector_running: bool
    notifier_enabled: bool


def _default_state_dir() -> Path:
    """Resolve the directory used to hold the persisted state file.

    Prefers the platform's user-data location
    (``%APPDATA%/LarkSnap`` on Windows, ``~/.larksnap`` elsewhere)
    so we don't pollute the project working directory.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "LarkSnap"
    return Path.home() / ".larksnap"


def default_state_path() -> Path:
    """Return the default state-file path. The directory is NOT created."""
    return _default_state_dir() / "runtime_state.json"


def save_state(
    state: PersistedRuntimeState,
    path: Path | None = None,
) -> None:
    """Persist ``state`` to ``path`` (or the default location).

    Writes are atomic (write-to-temp + ``os.replace``) so a crash
    mid-write cannot leave a half-written file. Exceptions are
    logged and swallowed — a failed save must never abort the
    camera-close path.
    """
    target = path or default_state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "detector_running": state.detector_running,
            "notifier_enabled": state.notifier_enabled,
        }
        # ``delete=False`` so Windows can ``os.replace`` onto an
        # existing file (Windows refuses to replace an open file).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".runtime_state.", suffix=".tmp", dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, target)
        except Exception:
            # Clean up the temp file if the replace never happened.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:  # noqa: BLE001 — best-effort, log all
        _logger.error("Failed to persist runtime state to %s: %s", target, e)


def load_state(path: Path | None = None) -> PersistedRuntimeState | None:
    """Read the persisted state from ``path`` (or the default location).

    Returns ``None`` if the file does not exist, is empty, or
    contains unparseable / schema-incompatible data. Callers should
    treat this as "no prior state" and fall back to defaults.
    """
    target = path or default_state_path()
    try:
        with open(target, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        _logger.warning("Could not read runtime state from %s: %s", target, e)
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") != _SCHEMA_VERSION:
        return None
    try:
        return PersistedRuntimeState(
            detector_running=bool(raw["detector_running"]),
            notifier_enabled=bool(raw["notifier_enabled"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        _logger.warning("Runtime state payload is malformed: %s", e)
        return None


def clear_state(path: Path | None = None) -> None:
    """Delete the persisted state file if it exists.

    Best-effort: errors are logged but never raised. Used by tests
    and by callers that want to force the next open_camera to use
    defaults.
    """
    target = path or default_state_path()
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        _logger.warning("Failed to clear runtime state at %s: %s", target, e)
