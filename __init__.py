# -*- coding: utf-8 -*-
"""Ask Claude (Anthropic) from Albert.

Modes (suffix on the prompt):
  c <prompt>          Placeholder; Enter sends async. Alt menu picks output channel.
                      Default: copy + notify. Other channels: paste at cursor,
                      window (markdown viewer), browser (rendered HTML), CopyQ history,
                      notification only.
  c <prompt>?         "Fact mode" — terse one-line answer rendered live in the
                      result list (Haiku, ~300ms debounced).
  c <prompt>!         "Command mode" — shell/code snippet only, rendered live.
                      Default action is paste-at-cursor; review before pressing Enter.

Pre-requisites:
  pip install anthropic
  Set ANTHROPIC_API_KEY env var or paste key into plugin settings.
"""

from albert import *
from pathlib import Path
import hashlib
import html as _html
import os
import shutil
import subprocess
import tempfile
import threading
import time

import anthropic


md_iid = "5.0"
md_version = "1.0"
md_name = "Claude"
md_description = "Ask Claude (Anthropic) from Albert"
md_license = "MIT"
md_url = "https://github.com/Bierchermuesli/albert-claude"
md_authors = ["@Bierchermuesli"]
md_lib_dependencies = ["anthropic"]


SYSTEM_PROMPT_DEFAULT = """\
You are an assistant running inside a quick keyboard launcher. Output rules:
- Plain text only — no markdown formatting (no **bold**, no _italic_, no headings, no bullets).
- No code fences. If you must show code or a command, output it inline with no fencing.
- Be concise. The launcher window is narrow; users want answers, not preambles.
- No "Sure!", "Of course!", "I'd be happy to" preambles. Start with the answer.
- Avoid trailing pleasantries like "Let me know if you have other questions."
- If the question is ambiguous, pick the most likely interpretation and answer it.
"""

SYSTEM_PROMPT_FACT = """\
You answer questions in ONE short sentence. No more.
- Plain text. No markdown, no preamble, no caveats.
- Lead with the fact. End with a period.
- For unknown answers, say "Unknown." and stop.
"""

SYSTEM_PROMPT_COMMAND = """\
Output a single shell command or code snippet that accomplishes the user's request.
Output rules:
- Output ONLY the command/code. Nothing else.
- No prose, no explanation, no leading or trailing prose.
- No markdown code fences.
- If the request is ambiguous, output the most likely interpretation.
- Default to bash/POSIX shell unless the request mentions a specific language.
- Multiple commands: chain with && or use a single line.
"""

# ~300ms typing debounce for `?`/`!` modes — avoids firing API on every keystroke.
LIVE_DEBOUNCE_SECONDS = 0.3
CACHE_TTL_SECONDS = 1800  # 30 min in-memory cache for identical prompts
# When the clipboard holds a path to a text file, read up to this many bytes.
MAX_CLIPBOARD_FILE_BYTES = 256 * 1024  # 256 KB
# Empty `c ` waits this long before adding the usage-stats row, so the
# default ready hint shows immediately and stats only appear if you pause.
EMPTY_QUERY_STATS_DELAY_SECONDS = 0.4

# USD per 1M tokens. Rough but good enough for a launcher cost estimate.
# Cache-read ≈ 10% of input rate; cache-write ≈ 125% of input rate.
PRICES_PER_MTOK = {
    "claude-fable-5-1":  {"in": 10.0, "out": 50.0, "cache_read": 0.25, "cache_write": 12.5},
    "claude-opus-5-5":   {"in": 4.0,  "out": 20.0, "cache_read": 0.2,  "cache_write": 5.0},
    "claude-sonnet-5-5": {"in": 2.0,  "out": 10.0, "cache_read": 0.2,  "cache_write": 2.5},
    "claude-haiku-5-5":  {"in": 0.1,  "out": 0.5,  "cache_read": 0.01, "cache_write": 0.125},
    "claude-opus-4-7":   {"in": 5.0,  "out": 25.0, "cache_read": 0.5,  "cache_write": 6.25},
    "claude-opus-4-6":   {"in": 5.0,  "out": 25.0, "cache_read": 0.5,  "cache_write": 6.25},
    "claude-opus-4-5":   {"in": 5.0,  "out": 25.0, "cache_read": 0.5,  "cache_write": 6.25},
    "claude-sonnet-4-6": {"in": 3.0,  "out": 15.0, "cache_read": 0.3,  "cache_write": 3.75},
    "claude-sonnet-4-5": {"in": 3.0,  "out": 15.0, "cache_read": 0.3,  "cache_write": 3.75},
    "claude-haiku-4-5":  {"in": 1.0,  "out": 5.0,  "cache_read": 0.1,  "cache_write": 1.25},
}

# Previous built-in defaults. Saved configs still holding one of these are
# upgraded to the current default on load (user-picked models are kept).
LEGACY_DEFAULTS = {
    "model_fast": {"claude-haiku-4-5"},
    "model_deep": {"claude-opus-4-7"},
}

# Notification body length for copy/notify output.
NOTIFY_MAX_CHARS = 2000

# Models that reject `output_config.effort`.
NO_EFFORT_PREFIXES = ("claude-haiku-4", "claude-sonnet-4-5", "claude-3")


class Plugin(PluginInstance, GeneratorQueryHandler):
    _api_key = ""
    _model_fast = "claude-haiku-5-5"  # default for bare `c`, `?`, `!`
    _model_deep = "claude-opus-5-5"   # secondary "deep" item on bare `c`
    _system_prompt = ""

    def __init__(self):
        GeneratorQueryHandler.__init__(self)
        PluginInstance.__init__(self)
        self.icon_path = Path(__file__).parent / "icon.svg"
        self.client = None
        self._init_error = None
        self._cache = {}  # (mode, prompt, model) -> (text, timestamp)
        self._cache_lock = threading.Lock()

        # Per-model usage counters (this session only). Reset on plugin reload.
        self._usage = {}
        self._usage_started_at = time.time()
        self._usage_lock = threading.Lock()

        self._init_configuration()
        self._setup_client()

    # --- config -----------------------------------------------------------

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value
        self.writeConfig("api_key", value)
        self._setup_client()

    @property
    def model_fast(self):
        return self._model_fast

    @model_fast.setter
    def model_fast(self, value):
        self._model_fast = value
        self.writeConfig("model_fast", value)

    @property
    def model_deep(self):
        return self._model_deep

    @model_deep.setter
    def model_deep(self, value):
        self._model_deep = value
        self.writeConfig("model_deep", value)

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value
        self.writeConfig("system_prompt", value)

    def _init_configuration(self):
        for key, type_, default in [
            ("api_key", str, self._api_key),
            ("model_fast", str, self._model_fast),
            ("model_deep", str, self._model_deep),
            ("system_prompt", str, self._system_prompt),
        ]:
            conf = self.readConfig(key, type_)
            if conf is None or conf in LEGACY_DEFAULTS.get(key, ()):
                self.writeConfig(key, default)
                setattr(self, f"_{key}", default)
            else:
                setattr(self, f"_{key}", conf)

    def _setup_client(self):
        self._init_error = None
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.client = None
            return
        try:
            self.client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
        except Exception as e:
            self._init_error = f"Setup failed: {e}"
            self.client = None
            warning(f"Claude: client setup failed: {e}")

    def configWidget(self):
        return [
            {"type": "label", "text": "Configure Anthropic API access for Claude."},
            {"type": "lineedit", "label": "API Key", "property": "api_key",
             "widget_properties": {"placeholderText": "sk-ant-... (or set ANTHROPIC_API_KEY env var)"}},
            {"type": "label",
             "text": "<a href='https://console.anthropic.com/settings/keys'>Get an API key</a>",
             "widget_properties": {"textFormat": "RichText", "openExternalLinks": True}},
            {"type": "label", "text": "<b>Models</b>"},
            {"type": "lineedit", "label": "Fast model (default)", "property": "model_fast",
             "widget_properties": {"placeholderText": "claude-haiku-4-5"}},
            {"type": "lineedit", "label": "Deep model (secondary 'Send (deep)' row)", "property": "model_deep",
             "widget_properties": {"placeholderText": "claude-opus-4-7"}},
            {"type": "label", "text": "<b>System prompt (chat mode)</b>"},
            {"type": "lineedit", "label": "Custom system prompt", "property": "system_prompt",
             "widget_properties": {"placeholderText": "Leave empty for the default launcher prompt"}},
        ]

    def defaultTrigger(self):
        return "c "

    def synopsis(self, query):
        return "<prompt> | <prompt>? (fact) | <prompt>! (command)"

    def _icon_factory(self):
        return Icon.image(str(self.icon_path))

    # --- API call ---------------------------------------------------------

    def _ask(self, prompt: str, mode: str) -> str:
        """Call Claude. Cached. Returns response text or 'Error: ...' on failure.

        mode values: chat-fast, chat-deep, fact, command.
        """
        if self._init_error:
            return f"Error: {self._init_error}"
        if not self.client:
            return "Error: API key not configured. Set ANTHROPIC_API_KEY or open plugin settings."

        if mode == "fact":
            model = self.model_fast
            system = SYSTEM_PROMPT_FACT
            max_tokens = 1024
            effort = "low"
        elif mode == "command":
            model = self.model_fast
            system = SYSTEM_PROMPT_COMMAND
            max_tokens = 2048
            effort = "low"
        elif mode == "chat-deep":
            model = self.model_deep
            system = self.system_prompt or SYSTEM_PROMPT_DEFAULT
            max_tokens = 16000
            effort = "high"
        else:  # chat-fast (default)
            model = self.model_fast
            system = self.system_prompt or SYSTEM_PROMPT_DEFAULT
            max_tokens = 4096
            effort = "low"

        cache_key = (mode, prompt, model)
        now = time.time()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and now - cached[1] < CACHE_TTL_SECONDS:
                return cached[0]

        extra = {}
        if not model.startswith(NO_EFFORT_PREFIXES):
            extra["output_config"] = {"effort": effort}

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                **extra,
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            if not text:
                text = "(empty response)"
            self._record_usage(model, response)
            with self._cache_lock:
                self._cache[cache_key] = (text, now)
            return text
        except anthropic.APIStatusError as e:
            return f"Error: {e.status_code} {e.message}"
        except Exception as e:
            return f"Error: {e}"

    def _record_usage(self, model: str, response):
        """Accumulate per-model token usage from a successful response."""
        u = response.usage
        if u is None:
            return
        with self._usage_lock:
            entry = self._usage.setdefault(model, {
                "calls": 0, "input": 0, "output": 0,
                "cache_read": 0, "cache_create": 0,
            })
            entry["calls"] += 1
            entry["input"] += getattr(u, "input_tokens", 0) or 0
            entry["output"] += getattr(u, "output_tokens", 0) or 0
            entry["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            entry["cache_create"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    def _estimate_cost(self):
        total = 0.0
        with self._usage_lock:
            for model, u in self._usage.items():
                rates = PRICES_PER_MTOK.get(model)
                if not rates:
                    continue
                total += (
                    u["input"] * rates["in"]
                    + u["output"] * rates["out"]
                    + u["cache_read"] * rates["cache_read"]
                    + u["cache_create"] * rates["cache_write"]
                ) / 1_000_000
        return total

    def _stats_item(self):
        with self._usage_lock:
            usage_snapshot = {m: dict(u) for m, u in self._usage.items()}
        if not usage_snapshot:
            subtext = "No calls yet this session."
        else:
            total_calls = sum(u["calls"] for u in usage_snapshot.values())
            total_in = sum(u["input"] for u in usage_snapshot.values())
            total_out = sum(u["output"] for u in usage_snapshot.values())
            total_cr = sum(u["cache_read"] for u in usage_snapshot.values())
            cost = self._estimate_cost()
            cost_str = f" · ~${cost:.4f}" if cost > 0 else ""
            subtext = (
                f"{total_calls} calls · in {total_in:,} (cached {total_cr:,}) · "
                f"out {total_out:,}{cost_str}"
            )
        return StandardItem(
            id="claude-stats",
            text="Usage (this session)",
            subtext=subtext,
            icon_factory=self._icon_factory,
            actions=[
                Action("reset", "Reset session counters", self._reset_usage),
            ],
        )

    def _reset_usage(self):
        with self._usage_lock:
            self._usage.clear()
            self._usage_started_at = time.time()
        Notification(title="Claude", text="Session usage counters reset.").send()

    # --- output channels --------------------------------------------------

    @staticmethod
    def _output_copy(text: str):
        setClipboardText(text)
        Notification(title="Claude", text=Plugin._clip_notify(text)).send()

    @staticmethod
    def _output_paste(text: str):
        # Always put it on the clipboard so the user can Ctrl+V manually
        # if synthesized paste fails.
        setClipboardText(text)

        # Prefer external helpers — Albert's built-in setClipboardTextAndPaste
        # races with the launcher closing on some setups, so we synthesize
        # Ctrl+V ourselves after a short delay that lets focus return.
        helpers = [
            ("xdotool", ["xdotool", "key", "--clearmodifiers", "ctrl+v"]),
            ("wtype",   ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]),
            ("ydotool", ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]),
        ]
        for binary, cmd in helpers:
            if shutil.which(binary):
                subprocess.Popen(
                    ["sh", "-c", f"sleep 0.15 && {' '.join(cmd)}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return

        # No external helper — try Albert's built-in as a last resort.
        if havePasteSupport():
            setClipboardTextAndPaste(text)
            return

        Notification(
            title="Claude",
            text="Paste not available — install xdotool (X11) or wtype/ydotool (Wayland). Copied to clipboard.",
        ).send()

    @staticmethod
    def _clip_notify(text: str) -> str:
        if len(text) <= NOTIFY_MAX_CHARS:
            return text
        return text[:NOTIFY_MAX_CHARS].rstrip() + "…"

    @staticmethod
    def _output_notify(text: str):
        Notification(title="Claude", text=Plugin._clip_notify(text)).send()

    @staticmethod
    def _output_copyq(text: str, question: str = ""):
        if not shutil.which("copyq"):
            Notification(title="Claude", text="copyq not installed; falling back to clipboard").send()
            setClipboardText(text)
            return
        note = f"claude:{question}" if question else "claude"
        try:
            subprocess.run(
                ["copyq", "write", "0",
                 "application/x-copyq-item-notes", note,
                 "text/plain", text],
                check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            Notification(title="Claude", text=f"Added to CopyQ history\n{Plugin._clip_notify(text)}").send()
        except Exception as e:
            warning(f"Claude: copyq add failed: {e}")
            setClipboardText(text)

    def _output_window(self, text: str, prompt: str):
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                             delete=False, prefix="claude-") as f:
                f.write(f"# {prompt}\n\n{text}\n")
                path = f.name
            subprocess.Popen(["xdg-open", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            warning(f"Claude: open window failed: {e}")
            setClipboardText(text)

    def _output_browser(self, text: str, prompt: str):
        try:
            page = self._render_html(prompt, text)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html",
                                             delete=False, prefix="claude-") as f:
                f.write(page)
                path = f.name
            subprocess.Popen(["xdg-open", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            warning(f"Claude: open browser failed: {e}")

    @staticmethod
    def _render_html(prompt: str, text: str) -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Claude — {_html.escape(prompt[:60])}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 760px;
         margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #1a1a1a; }}
  h1 {{ font-size: 1em; color: #666; font-weight: 500; }}
  pre {{ background: #f5f3ee; padding: 1em; overflow-x: auto;
         white-space: pre-wrap; border-radius: 4px; font-size: 0.95em; }}
</style></head>
<body>
<h1>{_html.escape(prompt)}</h1>
<pre>{_html.escape(text)}</pre>
</body></html>
"""

    def _async_send(self, prompt: str, output_fn, mode: str = "chat-fast"):
        """Run the API call in a daemon thread, then invoke output_fn(text)."""
        def _run():
            text = self._ask(prompt, mode=mode)
            output_fn(text)
        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _get_clipboard():
        """Read system clipboard text. Returns trimmed text or None."""
        for cmd in (
            ["xclip", "-selection", "clipboard", "-o"],
            ["wl-paste", "--no-newline"],
        ):
            if not shutil.which(cmd[0]):
                continue
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout
            except Exception as e:
                warning(f"Claude: clipboard read via {cmd[0]} failed: {e}")
                continue
        return None

    @staticmethod
    def _combined_prompt(prompt: str, clipboard: str) -> str:
        return f"{prompt}\n\nInput (from clipboard):\n{clipboard}"

    @staticmethod
    def _combined_prompt_file(prompt: str, path: Path, content: str, truncated: bool) -> str:
        marker = f" (first {MAX_CLIPBOARD_FILE_BYTES} bytes)" if truncated else ""
        return f"{prompt}\n\nInput (file {path}{marker}):\n{content}"

    @staticmethod
    def _clipboard_preview(text: str, n: int = 80) -> str:
        flat = " ".join(text.split())
        return flat[:n] + ("…" if len(flat) > n else "")

    @staticmethod
    def _resolve_clipboard_path(clipboard: str):
        """If `clipboard` is a single-line path to an existing regular file, return Path. Else None."""
        s = clipboard.strip()
        if not s or "\n" in s or "\r" in s:
            return None
        # Tolerate file:// URIs and ~ expansion.
        if s.startswith("file://"):
            s = s[len("file://"):]
        try:
            p = Path(s).expanduser()
        except Exception:
            return None
        if not p.is_absolute() or not p.is_file():
            return None
        return p

    @staticmethod
    def _read_text_file(path: Path, max_bytes: int):
        """Read up to `max_bytes` from `path`. Returns (text, truncated) or None for binary/unreadable."""
        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                raw = f.read(max_bytes)
        except OSError as e:
            warning(f"Claude: read failed for {path}: {e}")
            return None
        # Binary heuristic: any NUL byte in the first 8 KB → treat as non-text.
        if b"\x00" in raw[:8192]:
            return None
        text = raw.decode("utf-8", errors="replace")
        return text, size > max_bytes

    @classmethod
    def _classify_clipboard(cls, clipboard, query):
        """Return ('file', path, text, truncated) | ('text', clipboard) | None."""
        if not clipboard:
            return None
        path = cls._resolve_clipboard_path(clipboard)
        if path is not None:
            result = cls._read_text_file(path, MAX_CLIPBOARD_FILE_BYTES)
            if result is not None:
                text, truncated = result
                return ("file", path, text, truncated)
            # File exists but is binary/unreadable: don't fall through to the
            # plain-text variant — sending the bare path string as context is
            # never what the user wants.
            return None
        if clipboard.strip() and clipboard.strip() != query:
            return ("text", clipboard)
        return None

    # --- main generator ---------------------------------------------------

    def items(self, ctx):
        if self._init_error:
            yield [StandardItem(
                id="claude-error",
                text="Claude initialization error",
                subtext=self._init_error,
                icon_factory=self._icon_factory,
                actions=[Action("settings", "Open Settings", lambda: openConfig())],
            )]
            return

        if not self.client:
            yield [StandardItem(
                id="claude-no-key",
                text="API Key Missing",
                subtext="Set ANTHROPIC_API_KEY env var or paste key in plugin settings.",
                icon_factory=self._icon_factory,
                actions=[Action("settings", "Open Settings", lambda: openConfig())],
            )]
            return

        query = ctx.query.strip()
        if not query:
            ready = StandardItem(
                id="claude-ready",
                text="Ask Claude",
                subtext=f"Type a prompt. Default: {self._short_model_name(self.model_fast)} · '?' fact · '!' command",
                icon_factory=self._icon_factory,
            )
            yield [ready]
            # If the user starts typing within the delay we skip the stats row
            # entirely so it doesn't briefly flash on screen.
            time.sleep(EMPTY_QUERY_STATS_DELAY_SECONDS)
            if not ctx.isValid:
                return
            yield [ready, self._stats_item()]
            return

        # Live preview modes: yield "Thinking..." → debounce → check ctx → fire.
        if query.endswith("?"):
            real = query[:-1].strip()
            if not real:
                return
            yield from self._handle_live(ctx, real, mode="fact",
                                         label_subtitle=f"Fact · {self._short_model_name(self.model_fast)}",
                                         id_prefix="fact",
                                         actions_for=self._actions_fact)
            return

        if query.endswith("!"):
            real = query[:-1].strip()
            if not real:
                return
            yield from self._handle_live(ctx, real, mode="command",
                                         label_subtitle=f"Command · {self._short_model_name(self.model_fast)}",
                                         id_prefix="cmd",
                                         actions_for=self._actions_command)
            return

        # Bare chat mode: emit fast + deep rows. If clipboard holds a path or
        # text, emit "with file/clipboard" variants in fast/deep too. All rows
        # require Enter to send — typing alone never spends API tokens here.
        items = [
            self._chat_placeholder(query, attachment=None, kind="fast"),
            self._chat_placeholder(query, attachment=None, kind="deep"),
        ]

        attachment = self._classify_clipboard(self._get_clipboard(), query)
        if attachment is not None:
            items.append(self._chat_placeholder(query, attachment=attachment, kind="fast"))
            items.append(self._chat_placeholder(query, attachment=attachment, kind="deep"))

        yield items

    def _chat_placeholder(self, query: str, attachment, kind: str):
        """Build one row for bare-chat mode. `kind` is 'fast' or 'deep'."""
        if kind == "deep":
            mode = "chat-deep"
            model = self.model_deep
            label_tag = "Ask Deep"
        else:
            mode = "chat-fast"
            model = self.model_fast
            label_tag = "Ask Fast"

        # Build the prompt body and the metadata pieces.
        if attachment is None:
            prompt = query
            label = label_tag
            attach_meta = ""
            id_suffix = ""
        elif attachment[0] == "file":
            _, path, content, truncated = attachment
            prompt = self._combined_prompt_file(query, path, content, truncated)
            size = self._human_size(len(content.encode("utf-8")))
            trunc_tag = " (truncated)" if truncated else ""
            label = f"{label_tag} · file as input"
            attach_meta = f" · {path.name} ({size}{trunc_tag})"
            id_suffix = f"-file-{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
        else:  # ("text", clipboard)
            _, clipboard = attachment
            prompt = self._combined_prompt(query, clipboard)
            label = f"{label_tag} · clipboard as input"
            attach_meta = f" · clip: {self._clipboard_preview(clipboard, n=40)}"
            id_suffix = f"-clip-{hashlib.sha1(clipboard.encode()).hexdigest()[:8]}"

        subtext = f"Enter: copy + notify{attach_meta} · {self._short_model_name(model)}"
        id_ = f"claude-{kind}{id_suffix}-{hashlib.sha1(query.encode()).hexdigest()[:8]}"

        return StandardItem(
            id=id_,
            text=label,
            subtext=subtext,
            icon_factory=self._icon_factory,
            actions=[
                Action("send-copy", "Copy + notify",
                       lambda p=prompt, m=mode: self._async_send(p, self._output_copy, mode=m)),
                Action("send-paste", "Paste at cursor",
                       lambda p=prompt, m=mode: self._async_send(p, self._output_paste, mode=m)),
                Action("send-window", "Open in markdown viewer",
                       lambda p=prompt, q=query, m=mode: self._async_send(p, lambda t, qq=q: self._output_window(t, qq), mode=m)),
                Action("send-browser", "Open in browser",
                       lambda p=prompt, q=query, m=mode: self._async_send(p, lambda t, qq=q: self._output_browser(t, qq), mode=m)),
                Action("send-copyq", "Add to CopyQ history",
                       lambda p=prompt, q=query, m=mode: self._async_send(p, lambda t, qq=q: self._output_copyq(t, qq), mode=m)),
                Action("send-notify", "Notification only",
                       lambda p=prompt, m=mode: self._async_send(p, self._output_notify, mode=m)),
            ],
        )

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ("B", "KB", "MB"):
            if n < 1024:
                return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"

    @staticmethod
    def _short_model_name(model_id: str) -> str:
        """Map e.g. 'claude-haiku-4-5' -> 'Haiku 4.5'. Unknown shapes pass through."""
        parts = model_id.split("-")
        if len(parts) >= 4 and parts[0] == "claude":
            return f"{parts[1].capitalize()} {'.'.join(parts[2:])}"
        return model_id

    def _handle_live(self, ctx, real_query, mode, label_subtitle, id_prefix, actions_for):
        # Debounce: short sleep so rapid typing doesn't fire an API call per keystroke.
        # No placeholder: Albert appends later batches instead of replacing it, or
        # fetches them only on scrolling.
        time.sleep(LIVE_DEBOUNCE_SECONDS)
        if not ctx.isValid:
            return

        text = self._ask(real_query, mode=mode)

        if not ctx.isValid:
            return

        item_id = f"claude-{id_prefix}-{hashlib.sha1(real_query.encode()).hexdigest()[:8]}"
        items = [StandardItem(
            id=item_id,
            text=text,
            subtext=label_subtitle,
            icon_factory=self._icon_factory,
            actions=actions_for(text, real_query),
        )]

        # Optional follow-up that uses the clipboard (or a file at the path in
        # the clipboard) as additional context. Not auto-fired — Enter required,
        # so we don't double live-mode cost.
        attachment = self._classify_clipboard(self._get_clipboard(), real_query)
        if attachment is not None:
            model_short = self._short_model_name(self.model_fast)
            if attachment[0] == "file":
                _, path, content, truncated = attachment
                combined = self._combined_prompt_file(real_query, path, content, truncated)
                size_label = self._human_size(len(content.encode("utf-8")))
                trunc = " (truncated)" if truncated else ""
                follow_text = "Ask with file as input"
                follow_sub = f"Enter: copy + notify · {path.name} ({size_label}{trunc}) · {model_short}"
            else:
                _, clipboard = attachment
                combined = self._combined_prompt(real_query, clipboard)
                follow_text = "Ask with clipboard as input"
                follow_sub = f"Enter: copy + notify · clip: {self._clipboard_preview(clipboard, n=40)} · {model_short}"

            items.append(StandardItem(
                id=f"{item_id}-attach",
                text=follow_text,
                subtext=follow_sub,
                icon_factory=self._icon_factory,
                actions=[
                    Action("send-copy", "Send + copy + notify",
                           lambda p=combined, m=mode: self._async_send(p, self._output_copy, mode=m)),
                    Action("send-paste", "Send + paste at cursor",
                           lambda p=combined, m=mode: self._async_send(p, self._output_paste, mode=m)),
                    Action("send-copyq", "Send + add to CopyQ history",
                           lambda p=combined, q=real_query, m=mode: self._async_send(p, lambda t, qq=q: self._output_copyq(t, qq), mode=m)),
                ],
            ))

        yield items

    def _actions_fact(self, text, prompt):
        return [
            Action("copy", "Copy", lambda t=text: self._output_copy(t)),
            Action("paste", "Paste at cursor", lambda t=text: self._output_paste(t)),
            Action("copyq", "Add to CopyQ history", lambda t=text, p=prompt: self._output_copyq(t, p)),
            Action("notify", "Show notification", lambda t=text: self._output_notify(t)),
        ]

    def _actions_command(self, text, prompt):
        return [
            Action("paste", "Paste at cursor", lambda t=text: self._output_paste(t)),
            Action("copy", "Copy", lambda t=text: self._output_copy(t)),
            Action("copyq", "Add to CopyQ history", lambda t=text, p=prompt: self._output_copyq(t, p)),
            Action("window", "Open in viewer",
                   lambda t=text, p=prompt: self._output_window(t, p)),
        ]
