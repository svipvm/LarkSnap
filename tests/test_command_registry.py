"""Tests for the Feishu chat-command registry and ``/help`` rendering."""

from __future__ import annotations

import pytest

from larksnap.adapters.notifier.command_registry import (
    CommandRegistry,
    CommandSpec,
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Snapshot the registry around each test so the test order is irrelevant.

    The built-in command set is registered at import time, so we save
    and restore around each test instead of ``clear()``-ing — that way
    a test that registers a throwaway spec doesn't leak into the next
    one, and the defaults stay available for the tests that need them.
    """
    saved_specs = dict(CommandRegistry._specs)
    saved_aliases = dict(CommandRegistry._aliases)
    try:
        yield
    finally:
        CommandRegistry._specs = saved_specs
        CommandRegistry._aliases = saved_aliases


class TestCommandSpec:
    def test_is_frozen(self) -> None:
        spec = CommandSpec(name="ping", description="ping the bot", syntax="/ping")
        with pytest.raises(Exception):
            spec.name = "pong"  # type: ignore[misc]

    def test_optional_fields_default_empty(self) -> None:
        spec = CommandSpec(name="ping", description="d", syntax="/ping")
        assert spec.examples == ()
        assert spec.aliases == ()


class TestCommandRegistryDefaults:
    def test_builtin_commands_registered(self) -> None:
        names = {s.name for s in CommandRegistry.all_specs()}
        # The five commands the controller already handles must be in
        # the registry so ``/help`` can describe them and the parser
        # can accept them.
        assert {"init", "start", "stop", "status", "help"} <= names

    def test_status_alias_resolved(self) -> None:
        spec = CommandRegistry.get("state")
        assert spec is not None
        assert spec.name == "status"

    def test_lookup_is_case_insensitive(self) -> None:
        assert CommandRegistry.get("START") is not None
        assert CommandRegistry.get("Start") is not None
        assert CommandRegistry.get("start").name == "start"

    def test_unknown_command_returns_none(self) -> None:
        assert CommandRegistry.get("definitely-not-a-command") is None

    def test_empty_or_whitespace_name_returns_none(self) -> None:
        # ``get`` is documented to refuse empty / falsy names so
        # an unparseable command never accidentally matches.
        assert CommandRegistry.get("") is None

    def test_known_names_includes_aliases(self) -> None:
        known = CommandRegistry.known_names()
        assert "start" in known
        assert "state" in known  # alias for status

    def test_all_specs_sorted_by_name(self) -> None:
        names = [s.name for s in CommandRegistry.all_specs()]
        assert names == sorted(names)


class TestCommandRegistryRegistration:
    def test_register_new_command(self) -> None:
        spec = CommandSpec(
            name="ping",
            description="Ping the bot",
            syntax="/ping",
            examples=("/ping",),
        )
        CommandRegistry.register(spec)
        assert CommandRegistry.get("ping") is spec
        assert "ping" in CommandRegistry.known_names()

    def test_register_duplicate_overwrites(self) -> None:
        first = CommandSpec(name="ping", description="first", syntax="/ping")
        second = CommandSpec(name="ping", description="second", syntax="/ping")
        CommandRegistry.register(first)
        CommandRegistry.register(second)
        assert CommandRegistry.get("ping").description == "second"

    def test_alias_cannot_collide_with_existing_command(self) -> None:
        # ``start`` is already a primary name; trying to register an
        # alias that collides with it must fail loudly rather than
        # silently shadowing the real command.
        bad = CommandSpec(
            name="restart",
            description="restart",
            syntax="/restart",
            aliases=("start",),
        )
        with pytest.raises(ValueError):
            CommandRegistry.register(bad)

    def test_command_name_cannot_collide_with_existing_alias(self) -> None:
        # The inverse: ``state`` is already an alias of ``status``;
        # trying to register a primary called ``state`` must fail.
        bad = CommandSpec(name="state", description="d", syntax="/state")
        with pytest.raises(ValueError):
            CommandRegistry.register(bad)

    def test_unregister_removes_spec_and_aliases(self) -> None:
        spec = CommandSpec(
            name="ping",
            description="ping",
            syntax="/ping",
            aliases=("pong",),
        )
        CommandRegistry.register(spec)
        CommandRegistry.unregister("ping")
        assert CommandRegistry.get("ping") is None
        assert CommandRegistry.get("pong") is None
        assert "ping" not in CommandRegistry.known_names()

    def test_unregister_unknown_is_noop(self) -> None:
        # A typo in a teardown script must not raise — the
        # registry is process-global and a missed unregister
        # is no worse than a missing register.
        CommandRegistry.unregister("never-registered-cmd")
        # Built-in commands are still present.
        assert CommandRegistry.get("start") is not None


class TestRenderHelp:
    def test_lists_all_default_commands(self) -> None:
        text = CommandRegistry.render_help()
        for cmd in ("init", "start", "stop", "status", "help"):
            assert f"/{cmd}" in text

    def test_includes_descriptions(self) -> None:
        text = CommandRegistry.render_help()
        # Spot-check the description text for one of the builtins.
        assert "开启告警通知" in text

    def test_includes_syntax_and_examples(self) -> None:
        text = CommandRegistry.render_help()
        assert "语法:" in text
        assert "示例:" in text

    def test_includes_aliases_when_present(self) -> None:
        text = CommandRegistry.render_help()
        # ``status`` is the only default with an alias.
        assert "/state" in text

    def test_omits_aliases_line_when_none(self) -> None:
        # ``/start`` has no aliases, so its block should not contain
        # the "别名:" line.
        text = CommandRegistry.render_help()
        start_block = text.split("/start", 1)[1].split("/", 1)[0]
        assert "别名" not in start_block

    def test_drilldown_for_specific_command(self) -> None:
        from larksnap.gateway.controller import _render_single_help

        spec = CommandRegistry.get("start")
        assert spec is not None
        text = _render_single_help(spec)
        assert "/start" in text
        assert "语法:" in text
        # Single-command block should not include other commands.
        assert "/stop" not in text
        assert "/init" not in text
