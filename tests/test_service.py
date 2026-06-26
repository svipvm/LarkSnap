"""Tests for the service layer.

The Windows SCM wrapper can only be exercised on a real Windows
machine with pywin32 installed. The tests below use mocking to
verify the public contract (CLI dispatch, install/uninstall shims,
unit-file rendering) on every platform.

Tests are marked with the ``platform_windows`` / ``platform_linux``
markers defined in ``pyproject.toml`` so the suite can be filtered
when a CI matrix runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from larksnap.config.models import AppConfig, ServiceConfig
from larksnap.service import (
    ServicePlatform,
    current_service_platform,
    install_service,
    uninstall_service,
)
from larksnap.service.linux import (
    UNIT_TEMPLATE,
    _build_exec_start,
    _render_unit,
    install_linux_service,
    uninstall_linux_service,
)
from larksnap.service.platform_utils import is_linux, is_windows
from larksnap.service.runner import (
    ServiceRunner,
    install_posix_stop_handlers,
    run_service_blocking,
)
from larksnap.service.tray import SystemTray


def _set_geteuid(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """Stub ``os.geteuid`` for the current test (POSIX-only attr)."""
    import os

    monkeypatch.setattr(os, "geteuid", lambda: value, raising=False)


# ---------------------------------------------------------------------------
# Tray
# ---------------------------------------------------------------------------
class TestSystemTray:
    def test_tray_creation(self) -> None:
        from larksnap.config.models import AppConfig
        from larksnap.gateway.controller import GatewayController

        config = AppConfig()
        controller = GatewayController(config)
        tray = SystemTray(controller)
        assert tray is not None


# ---------------------------------------------------------------------------
# platform_utils
# ---------------------------------------------------------------------------
class TestPlatformUtils:
    def test_current_service_platform_matches_sys_platform(self) -> None:
        if sys.platform == "win32":
            assert current_service_platform() is ServicePlatform.WINDOWS
        elif sys.platform.startswith("linux"):
            assert current_service_platform() is ServicePlatform.LINUX
        else:
            assert current_service_platform() is ServicePlatform.OTHER

    def test_is_windows_consistent(self) -> None:
        assert is_windows() == (sys.platform == "win32")

    def test_is_linux_consistent(self) -> None:
        assert is_linux() == sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# service package dispatch
# ---------------------------------------------------------------------------
class TestServiceDispatch:
    def test_install_service_dispatches_per_platform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        win_called = linux_called = False

        def fake_windows() -> int:
            nonlocal win_called
            win_called = True
            return 0

        def fake_linux() -> int:
            nonlocal linux_called
            linux_called = True
            return 0

        monkeypatch.setattr(
            "larksnap.service.windows.install_windows_service", fake_windows
        )
        monkeypatch.setattr(
            "larksnap.service.linux.install_linux_service", fake_linux
        )

        for forced_platform, expected_var in (
            (ServicePlatform.WINDOWS, "win_called"),
            (ServicePlatform.LINUX, "linux_called"),
        ):
            win_called = linux_called = False
            with mock.patch(
                "larksnap.service.current_service_platform",
                return_value=forced_platform,
            ):
                rc = install_service()
            assert rc == 0
            assert (expected_var == "win_called" and win_called) or (
                expected_var == "linux_called" and linux_called
            )

    def test_uninstall_service_dispatches_per_platform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        win_called = linux_called = False

        def fake_windows() -> int:
            nonlocal win_called
            win_called = True
            return 0

        def fake_linux() -> int:
            nonlocal linux_called
            linux_called = True
            return 0

        monkeypatch.setattr(
            "larksnap.service.windows.uninstall_windows_service", fake_windows
        )
        monkeypatch.setattr(
            "larksnap.service.linux.uninstall_linux_service", fake_linux
        )

        for forced_platform, attr in (
            (ServicePlatform.WINDOWS, "win_called"),
            (ServicePlatform.LINUX, "linux_called"),
        ):
            win_called = linux_called = False
            with mock.patch(
                "larksnap.service.current_service_platform",
                return_value=forced_platform,
            ):
                rc = uninstall_service()
            assert rc == 0
            assert (attr == "win_called" and win_called) or (
                attr == "linux_called" and linux_called
            )

    def test_install_service_on_unsupported_platform(self, capsys: pytest.CaptureFixture[str]) -> None:
        with mock.patch(
            "larksnap.service.current_service_platform",
            return_value=ServicePlatform.OTHER,
        ):
            rc = install_service()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Automatic service installation" in captured.err


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class TestServiceRunner:
    def test_runner_lifecycle_with_factory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The runner must call factory → initialize → start → stop."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("service:\n  name: test-svc\n", encoding="utf-8")

        from larksnap.config.loader import load_config
        from larksnap.gateway.controller import GatewayController

        config = load_config(str(cfg_path))
        real_controller = GatewayController(config)

        # Wire the runner to use a real controller, but stub the
        # heavy I/O calls so the test is hermetic.
        monkeypatch.setattr(real_controller, "initialize", mock.Mock())
        monkeypatch.setattr(real_controller, "start", mock.Mock())
        monkeypatch.setattr(real_controller, "stop", mock.Mock())

        factory = mock.Mock(return_value=real_controller)
        runner = ServiceRunner(config_path=str(cfg_path), controller_factory=factory)

        runner.start()
        factory.assert_called_once()
        real_controller.initialize.assert_called_once()
        real_controller.start.assert_called_once()

        # request_stop must be idempotent.
        runner.request_stop()
        runner.request_stop()
        assert runner.wait_for_stop(timeout=0.0) is True

        runner.stop()
        real_controller.stop.assert_called_once()

        # Idempotent stop.
        runner.stop()
        assert real_controller.stop.call_count == 1

    def test_runner_double_start_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("logging:\n  console_output: false\n", encoding="utf-8")
        runner = ServiceRunner(config_path=str(cfg_path))
        with mock.patch(
            "larksnap.service.runner.GatewayController", autospec=True
        ) as gc_cls:
            gc_cls.return_value.initialize = mock.Mock()
            gc_cls.return_value.start = mock.Mock()
            runner.start()
            with pytest.raises(RuntimeError, match="already started"):
                runner.start()


# ---------------------------------------------------------------------------
# POSIX signal handler
# ---------------------------------------------------------------------------
class TestPosixSignalHandlers:
    def test_install_posix_stop_handlers_requests_stop(self) -> None:
        runner = ServiceRunner()
        runner._logger = mock.Mock()  # noqa: SLF001 – test-only

        install_posix_stop_handlers(runner)

        # The handler must call request_stop on the runner.
        # SIGINT works on every platform including Windows.
        import signal as _signal

        previous = _signal.getsignal(_signal.SIGINT)
        try:
            install_posix_stop_handlers(runner)
            _signal.raise_signal(_signal.SIGINT)
            assert runner.wait_for_stop(timeout=1.0) is True
        finally:
            _signal.signal(_signal.SIGINT, previous)


# ---------------------------------------------------------------------------
# Linux: unit file rendering + install / uninstall
# ---------------------------------------------------------------------------
class TestLinuxUnitRendering:
    def test_render_unit_default(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "service:\n"
            "  name: larksnap\n"
            "  description: 'unit test'\n"
            "  systemd_type: simple\n"
            "  systemd_restart: always\n"
            "  systemd_wanted_by: graphical.target\n"
            "logging:\n"
            "  console_output: false\n",
            encoding="utf-8",
        )
        from larksnap.config.loader import load_config

        with mock.patch("larksnap.service.linux.load_config", return_value=load_config(str(cfg_path))):
            body = _render_unit(tmp_path, tmp_path / "out.log")
        assert "[Unit]" in body
        assert "Description=unit test" in body
        assert "Type=simple" in body
        assert "Restart=always" in body
        assert "WantedBy=graphical.target" in body
        assert "ExecStart=" in body
        assert f"StandardOutput=append:{tmp_path / 'out.log'}" in body

    def test_render_unit_respects_user(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "service:\n"
            "  systemd_user: larkuser\n"
            "logging:\n  console_output: false\n",
            encoding="utf-8",
        )
        from larksnap.config.loader import load_config

        with mock.patch("larksnap.service.linux.load_config", return_value=load_config(str(cfg_path))):
            body = _render_unit(tmp_path, tmp_path / "out.log")
        assert "User=larkuser" in body

    def test_build_exec_start_with_uv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("larksnap.service.linux._detect_uv", lambda: "/usr/bin/uv")
        cmd = _build_exec_start(tmp_path)
        assert "/usr/bin/uv" in cmd
        assert "larksnap.main service" in cmd

    def test_build_exec_start_without_uv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("larksnap.service.linux._detect_uv", lambda: None)
        cmd = _build_exec_start(tmp_path)
        assert "python" in cmd
        assert "larksnap.main service" in cmd

    def test_install_linux_requires_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_geteuid(monkeypatch, 1000)
        rc = install_linux_service()
        assert rc == 1

    def test_uninstall_linux_requires_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_geteuid(monkeypatch, 1000)
        rc = uninstall_linux_service()
        assert rc == 1

    def test_install_linux_writes_unit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        unit_path = tmp_path / "larksnap.service"
        cfg = AppConfig(
            service=ServiceConfig(
                name="larksnap",
                systemd_unit_path=str(unit_path),
            )
        )
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("service: {}\nlogging: {console_output: false}\n", encoding="utf-8")

        from larksnap.config.loader import load_config

        _set_geteuid(monkeypatch, 0)
        monkeypatch.setattr("larksnap.service.linux.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr(
            "larksnap.service.linux._run_systemctl",
            mock.Mock(return_value=mock.Mock(returncode=0, stderr="")),
        )
        # Re-patch the in-module load_config so _render_unit uses cfg.
        with mock.patch("larksnap.service.linux.load_config", return_value=cfg):
            rc = install_linux_service()
        assert rc == 0
        assert unit_path.exists()
        body = unit_path.read_text(encoding="utf-8")
        assert "ExecStart=" in body
        assert "Type=notify" in body

    def test_uninstall_linux_removes_unit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        unit_path = tmp_path / "larksnap.service"
        unit_path.write_text(UNIT_TEMPLATE, encoding="utf-8")
        cfg = AppConfig(
            service=ServiceConfig(
                name="larksnap",
                systemd_unit_path=str(unit_path),
            )
        )
        _set_geteuid(monkeypatch, 0)
        monkeypatch.setattr(
            "larksnap.service.linux._run_systemctl",
            mock.Mock(return_value=mock.Mock(returncode=0, stderr="")),
        )
        with mock.patch("larksnap.service.linux.load_config", return_value=cfg):
            rc = uninstall_linux_service()
        assert rc == 0
        assert not unit_path.exists()

    def test_uninstall_linux_idempotent_when_unit_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = AppConfig(
            service=ServiceConfig(
                name="larksnap",
                systemd_unit_path=str(tmp_path / "missing.service"),
            )
        )
        _set_geteuid(monkeypatch, 0)
        monkeypatch.setattr(
            "larksnap.service.linux._run_systemctl",
            mock.Mock(return_value=mock.Mock(returncode=0, stderr="")),
        )
        with mock.patch("larksnap.service.linux.load_config", return_value=cfg):
            rc = uninstall_linux_service()
        assert rc == 0


# ---------------------------------------------------------------------------
# Windows: install / uninstall shims
# ---------------------------------------------------------------------------
class TestWindowsServiceShims:
    def test_install_without_pywin32_prints_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from larksnap.service import windows as win_mod

        monkeypatch.setattr(win_mod, "win32serviceutil", None)
        rc = win_mod.install_windows_service()
        assert rc == 1
        captured = capsys.readouterr()
        assert "pywin32" in captured.err

    def test_uninstall_without_pywin32_prints_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from larksnap.service import windows as win_mod

        monkeypatch.setattr(win_mod, "win32serviceutil", None)
        rc = win_mod.uninstall_windows_service()
        assert rc == 1
        captured = capsys.readouterr()
        assert "pywin32" in captured.err


# ---------------------------------------------------------------------------
# Cross-platform runner integration
# ---------------------------------------------------------------------------
class TestRunServiceBlocking:
    def test_run_service_blocking_uses_factory_and_stops(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("logging:\n  console_output: false\n", encoding="utf-8")

        from larksnap.gateway.controller import GatewayController

        controller = GatewayController.__new__(GatewayController)
        controller.initialize = mock.Mock()  # type: ignore[method-assign]
        controller.start = mock.Mock()  # type: ignore[method-assign]
        controller.stop = mock.Mock()  # type: ignore[method-assign]

        runner = ServiceRunner(
            config_path=str(cfg_path),
            controller_factory=mock.Mock(return_value=controller),
        )

        # Patch the helper inside runner so it returns our pre-built
        # runner and the wait_for_stop fires on a timer.
        monkeypatch.setattr(
            "larksnap.service.runner.ServiceRunner", mock.Mock(return_value=runner)
        )
        ready_hook = mock.Mock()

        # Trigger the stop in a background thread.
        import threading
        import time

        def _trigger() -> None:
            time.sleep(0.1)
            runner.request_stop()

        threading.Thread(target=_trigger, daemon=True).start()
        run_service_blocking(config_path=str(cfg_path), on_ready=ready_hook)

        controller.initialize.assert_called_once()
        controller.start.assert_called_once()
        controller.stop.assert_called_once()
        ready_hook.assert_called_once_with(runner)
