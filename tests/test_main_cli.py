"""Tests for the main.py CLI dispatch.

The CLI surface is small but it ties together the three run modes
(Qt / tray / service) and the two installer subcommands
(install / uninstall). The tests below exercise the parser and
verify that the right platform wrapper gets called for each command.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from larksnap.main import _build_parser, main
from larksnap.service import ServicePlatform


class TestArgparse:
    def test_parser_accepts_all_commands(self) -> None:
        parser = _build_parser()
        for cmd in ("qt", "tray", "service", "install", "uninstall"):
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_parser_default_command_is_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_parser_config_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["-c", "config/custom.yaml", "qt"])
        assert args.config == "config/custom.yaml"


class TestMainDispatch:
    """`main()` must dispatch the right subcommand to the right entry point."""

    def test_qt_dispatches_to_run_with_qt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = mock.Mock()
        monkeypatch.setattr("larksnap.main.run_with_qt", called)
        monkeypatch.setattr("sys.argv", ["larksnap", "qt"])
        main()
        called.assert_called_once()

    def test_tray_dispatches_to_run_with_tray(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = mock.Mock()
        monkeypatch.setattr("larksnap.main.run_with_tray", called)
        monkeypatch.setattr("sys.argv", ["larksnap", "tray"])
        main()
        called.assert_called_once()

    def test_default_command_runs_qt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = mock.Mock()
        monkeypatch.setattr("larksnap.main.run_with_qt", called)
        monkeypatch.setattr("sys.argv", ["larksnap"])
        main()
        called.assert_called_once()

    @pytest.mark.parametrize(
        "platform,expected_attr",
        [
            (ServicePlatform.WINDOWS, "run_windows_service"),
            (ServicePlatform.LINUX, "run_linux_service"),
        ],
    )
    def test_service_dispatches_per_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        platform: ServicePlatform,
        expected_attr: str,
    ) -> None:
        win_called = linux_called = False

        def fake_windows() -> int:
            nonlocal win_called
            win_called = True
            return 0

        def fake_linux(_config_path: object) -> int:
            nonlocal linux_called
            linux_called = True
            return 0

        # `main.py` does `from ... import current_service_platform`,
        # so the local binding lives in `larksnap.main`. Patch there.
        monkeypatch.setattr("larksnap.main.current_service_platform", lambda: platform)
        monkeypatch.setattr(
            "larksnap.service.windows.run_windows_service", fake_windows
        )
        monkeypatch.setattr(
            "larksnap.service.linux.run_linux_service", fake_linux
        )

        with mock.patch(
            "larksnap.main.sys.exit", side_effect=SystemExit
        ) as exit_mock:
            monkeypatch.setattr("sys.argv", ["larksnap", "service"])
            with pytest.raises(SystemExit):
                main()

        if platform is ServicePlatform.WINDOWS:
            assert win_called is True
        else:
            assert linux_called is True
        exit_mock.assert_called_once_with(0)

    def test_install_dispatches_to_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = mock.Mock(return_value=0)
        monkeypatch.setattr("larksnap.service.install_service", called)
        with mock.patch("larksnap.main.sys.exit", side_effect=SystemExit) as exit_mock:
            monkeypatch.setattr("sys.argv", ["larksnap", "install"])
            with pytest.raises(SystemExit):
                main()
        called.assert_called_once()
        exit_mock.assert_called_once_with(0)

    def test_uninstall_dispatches_to_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = mock.Mock(return_value=0)
        monkeypatch.setattr("larksnap.service.uninstall_service", called)
        with mock.patch("larksnap.main.sys.exit", side_effect=SystemExit) as exit_mock:
            monkeypatch.setattr("sys.argv", ["larksnap", "uninstall"])
            with pytest.raises(SystemExit):
                main()
        called.assert_called_once()
        exit_mock.assert_called_once_with(0)
