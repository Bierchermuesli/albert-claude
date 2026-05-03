"""Smoke-tests for the pure-Python helpers in the Claude plugin.

The plugin module imports `albert` and `anthropic` at top level, so we stub them
before importing the plugin. This file can be run standalone:

    python3 tests/test_helpers.py
    python3 -m unittest tests.test_helpers   # from the plugin directory

It exercises only the helpers that don't need a live Anthropic API key:
path detection, file reading, clipboard classification, and size formatting.
For end-to-end behaviour (the API call, output channels, items() yields) test
manually inside Albert.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _stub_albert_and_anthropic():
    """Install minimal stubs so the plugin module imports without Albert/Anthropic.

    Each albert class gets its own stub class — the plugin uses multiple
    inheritance (PluginInstance, GeneratorQueryHandler), and Python rejects
    duplicate base classes.
    """
    albert = types.ModuleType("albert")

    def make_stub_class():
        class _Stub:
            def __init__(self, *a, **kw): pass
        return _Stub

    for name in (
        "PluginInstance", "GeneratorQueryHandler",
        "StandardItem", "Action", "Icon", "Notification",
    ):
        setattr(albert, name, make_stub_class())
    for name in (
        "setClipboardText", "setClipboardTextAndPaste", "havePasteSupport",
        "openUrl", "openConfig", "warning", "info", "debug", "critical",
    ):
        setattr(albert, name, lambda *a, **kw: None)
    sys.modules["albert"] = albert

    anthropic = types.ModuleType("anthropic")

    class _Anthropic:
        def __init__(self, **kw): pass

    class _APIStatusError(Exception):
        pass

    anthropic.Anthropic = _Anthropic
    anthropic.APIStatusError = _APIStatusError
    sys.modules["anthropic"] = anthropic


def _load_plugin_module():
    plugin_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "claude_plugin", plugin_root / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stub_albert_and_anthropic()
_plugin = _load_plugin_module()
Plugin = _plugin.Plugin


class ResolveClipboardPathTests(unittest.TestCase):
    def test_absolute_file(self):
        self.assertEqual(Plugin._resolve_clipboard_path("/etc/hostname"),
                         Path("/etc/hostname"))

    def test_whitespace_trim(self):
        self.assertEqual(Plugin._resolve_clipboard_path("  /etc/hostname  "),
                         Path("/etc/hostname"))

    def test_home_expansion(self):
        self.assertEqual(Plugin._resolve_clipboard_path("~/.bashrc"),
                         Path.home() / ".bashrc")

    def test_file_uri(self):
        self.assertEqual(Plugin._resolve_clipboard_path("file:///etc/hostname"),
                         Path("/etc/hostname"))

    def test_nonexistent(self):
        self.assertIsNone(Plugin._resolve_clipboard_path("/nope/this/does/not/exist"))

    def test_relative_rejected(self):
        # Relative paths depend on cwd — skip them to avoid surprises.
        self.assertIsNone(Plugin._resolve_clipboard_path("relative/path"))

    def test_multiline_rejected(self):
        self.assertIsNone(Plugin._resolve_clipboard_path("hi\nthere"))

    def test_directory_rejected(self):
        self.assertIsNone(Plugin._resolve_clipboard_path("/tmp"))

    def test_empty(self):
        self.assertIsNone(Plugin._resolve_clipboard_path(""))


class ReadTextFileTests(unittest.TestCase):
    def test_text_file(self):
        result = Plugin._read_text_file(Path("/etc/hostname"), 1024)
        self.assertIsNotNone(result)
        text, truncated = result
        self.assertGreater(len(text), 0)
        self.assertFalse(truncated)

    def test_binary_file_returns_none(self):
        # /bin/ls is an ELF binary; first 8KB contains NUL bytes.
        self.assertIsNone(Plugin._read_text_file(Path("/bin/ls"), 1024))

    def test_truncation_flag(self):
        # Read /etc/hostname with a tiny cap so the size > cap branch fires.
        # Pick a file we know is non-empty; pass max_bytes=1.
        result = Plugin._read_text_file(Path("/etc/hostname"), 1)
        self.assertIsNotNone(result)
        text, truncated = result
        self.assertEqual(len(text), 1)
        self.assertTrue(truncated)


class ClassifyClipboardTests(unittest.TestCase):
    def test_path_routes_to_file(self):
        kind, *rest = Plugin._classify_clipboard("/etc/hostname", "summarize")
        self.assertEqual(kind, "file")
        path, content, truncated = rest
        self.assertEqual(path, Path("/etc/hostname"))
        self.assertGreater(len(content), 0)

    def test_plain_text(self):
        self.assertEqual(
            Plugin._classify_clipboard("hello world", "different prompt"),
            ("text", "hello world"),
        )

    def test_duplicate_of_query_skipped(self):
        self.assertIsNone(Plugin._classify_clipboard("same", "same"))

    def test_empty_skipped(self):
        self.assertIsNone(Plugin._classify_clipboard("", "x"))
        self.assertIsNone(Plugin._classify_clipboard(None, "x"))

    def test_binary_path_does_not_fall_through(self):
        # /bin/ls exists but is binary — classify should return None,
        # never the bare path string as text.
        self.assertIsNone(Plugin._classify_clipboard("/bin/ls", "summarize"))


class HumanSizeTests(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(Plugin._human_size(0), "0 B")
        self.assertEqual(Plugin._human_size(512), "512 B")

    def test_kilobytes(self):
        self.assertEqual(Plugin._human_size(1024), "1.0 KB")
        self.assertEqual(Plugin._human_size(4096), "4.0 KB")
        self.assertEqual(Plugin._human_size(256 * 1024), "256.0 KB")

    def test_megabytes(self):
        self.assertEqual(Plugin._human_size(5 * 1024 * 1024), "5.0 MB")


class ClipboardPreviewTests(unittest.TestCase):
    def test_short(self):
        self.assertEqual(Plugin._clipboard_preview("hello"), "hello")

    def test_whitespace_flattened(self):
        self.assertEqual(Plugin._clipboard_preview("hello\n\nworld   foo"),
                         "hello world foo")

    def test_truncation(self):
        s = "x" * 200
        out = Plugin._clipboard_preview(s, n=80)
        self.assertEqual(len(out), 81)  # 80 chars + ellipsis
        self.assertTrue(out.endswith("…"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
