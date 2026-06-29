"""Command registry for the Feishu chat-command protocol.

Each command is described declaratively via a :class:`CommandSpec` and
registered through :class:`CommandRegistry`. The registry is the single
source of truth for:

* What commands the bot accepts (used by the parser for validation)
* The descriptions, syntax, and examples that ``/help`` renders back to
  the user (used by the controller's ``_handle_command``)

Keeping the spec data here (instead of hard-coding it in the WS client
or the controller) lets new commands be added with a single ``register``
call and lets the ``/help`` response stay automatically in sync with the
parser's accepted set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CommandSpec:
    """Declarative description of a single bot command.

    Attributes:
        name: Primary command name without the leading ``/`` (e.g. ``"start"``).
        description: One-line, user-facing description shown in ``/help``.
        syntax: The full usage string, including the leading ``/`` and any
            placeholders (e.g. ``"/camera <index>"``).
        examples: Example invocations shown alongside the command in
            ``/help``. Kept as a tuple so the spec is hashable / frozen.
        aliases: Alternative names that map to the same command. Resolved
            transparently by :py:meth:`CommandRegistry.get`.
    """

    name: str
    description: str
    syntax: str
    examples: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


class CommandRegistry:
    """Process-wide registry of :class:`CommandSpec` entries.

    The class-level store is intentionally global because the protocol
    surface is fixed for the lifetime of the process and is read from
    both the WS-client parser thread and the controller thread.
    """

    _specs: dict[str, CommandSpec] = {}
    # Reverse map: alias -> primary name, so ``get`` can resolve aliases
    # without scanning the full spec list on every call.
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, spec: CommandSpec) -> CommandSpec:
        """Register ``spec``. Returns the spec to allow decorator usage.

        Re-registering the same primary name overwrites the previous
        entry. Aliases that collide with an existing primary name are
        rejected with ``ValueError`` to prevent silent shadowing.
        """
        key = spec.name.lower()
        if key in cls._aliases and cls._aliases[key] != key:
            raise ValueError(
                f"Command name {spec.name!r} collides with an existing alias"
            )
        cls._specs[key] = spec
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in cls._specs:
                raise ValueError(
                    f"Alias {alias!r} collides with an existing command name"
                )
            cls._aliases[alias_key] = key
        return spec

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a spec (and its aliases) from the registry.

        Mostly useful in tests that need a clean slate. ``KeyError`` is
        not swallowed so callers know when they removed nothing.
        """
        key = name.lower()
        spec = cls._specs.pop(key, None)
        if spec is not None:
            for alias in spec.aliases:
                cls._aliases.pop(alias.lower(), None)
        # Also clear any alias-only entry that pointed at ``name``.
        for alias_key, target in list(cls._aliases.items()):
            if target == key and alias_key not in cls._specs:
                cls._aliases.pop(alias_key, None)

    @classmethod
    def clear(cls) -> None:
        """Drop every registered spec. Test-only convenience."""
        cls._specs.clear()
        cls._aliases.clear()

    @classmethod
    def get(cls, name: str) -> CommandSpec | None:
        """Return the spec for ``name``, resolving aliases transparently.

        Returns ``None`` if the name is not registered and not an alias
        of any registered command.
        """
        if not name:
            return None
        key = name.lower()
        if key in cls._specs:
            return cls._specs[key]
        target = cls._aliases.get(key)
        if target is not None:
            return cls._specs.get(target)
        return None

    @classmethod
    def known_names(cls) -> set[str]:
        """Set of all accepted command names (primary + aliases)."""
        return set(cls._specs.keys()) | set(cls._aliases.keys())

    @classmethod
    def all_specs(cls) -> list[CommandSpec]:
        """All registered specs, sorted by primary name for stable output."""
        return sorted(cls._specs.values(), key=lambda s: s.name)

    @classmethod
    def render_help(
        cls,
        header: str = "[LarkSnap] 可用命令：",
        footer: str = "发送 /help 再次查看本帮助。",
    ) -> str:
        """Build the multi-line help text returned by the ``/help`` command.

        The output is plain text so it can be delivered through any
        text-message channel (Feishu ``text`` msg_type, logs, etc.).
        Unknown command names are never rendered here — the registry is
        the source of truth, so the list is always consistent with what
        the parser will actually accept.
        """
        lines: list[str] = [header, ""]
        for spec in cls.all_specs():
            lines.append(f"/{spec.name} - {spec.description}")
            lines.append(f"  语法: {spec.syntax}")
            if spec.examples:
                examples = "，".join(spec.examples)
                lines.append(f"  示例: {examples}")
            if spec.aliases:
                alias_str = "，".join(f"/{a}" for a in spec.aliases)
                lines.append(f"  别名: {alias_str}")
            lines.append("")
        lines.append(footer)
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Built-in command set
# ---------------------------------------------------------------------------
#
# These match the commands the controller already understands. Each spec
# documents the user-facing behaviour, the on-the-wire syntax, and one
# or more examples. Adding a new command is a two-step process:
#
#   1. Register a ``CommandSpec`` below (or in a downstream module).
#   2. Handle the new ``cmd.name`` in ``GatewayController._handle_command``.
#
# Re-registering a built-in (e.g. in tests) replaces the default spec.


def _register_builtins() -> None:
    """Install the default command set. Idempotent."""
    defaults: Iterable[CommandSpec] = (
        CommandSpec(
            name="init",
            description="初始化并保存当前 chat_id，后续通知将发送至此会话。",
            syntax="/init",
            examples=("/init",),
        ),
        CommandSpec(
            name="start",
            description="开启告警通知。检测到目标时将通过 Feishu 推送。",
            syntax="/start",
            examples=("/start",),
        ),
        CommandSpec(
            name="stop",
            description="关闭告警通知。不会再向 Feishu 推送检测结果。",
            syntax="/stop",
            examples=("/stop",),
        ),
        CommandSpec(
            name="status",
            description="查询当前网关状态（摄像头、检测、通知开关）。",
            syntax="/status",
            examples=("/status",),
            aliases=("state",),
        ),
        CommandSpec(
            name="help",
            description="显示本帮助信息，列出所有可用命令及其语法与示例。",
            syntax="/help [command]",
            examples=("/help", "/help start"),
        ),
        CommandSpec(
            name="config",
            description=(
                "查看或修改 config.yaml 配置。值用 JSON 字面量表示"
                "（数字、字符串、列表、布尔均可）。修改后立即落盘，"
                "无需重启即可生效的字段会热生效，需要重启的字段会"
                "在回复中明确标注。"
            ),
            syntax="/config <get|set|show|paths> [path] [json_value]",
            examples=(
                "/config",
                "/config paths",
                "/config get detector.confidence_threshold",
                "/config set detector.confidence_threshold 0.6",
                '/config set detector.target_classes ["person","car","dog"]',
                "/config set gateway.notification_interval 60",
                "/config show notifier",
            ),
        ),
    )
    for spec in defaults:
        CommandRegistry.register(spec)


_register_builtins()
