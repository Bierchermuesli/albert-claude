# Claude — Albert Python Plugin

> **Experimental.** still figure out the usecase :)

Ask [Claude](https://www.anthropic.com/claude) from the [Albert Launcher](https://albertlauncher.github.io/). One-shot prompts, fact lookups, clipboard in/output and command/code snippets — without opening a chat window.


<img width="893" height="314" alt="image" src="https://github.com/user-attachments/assets/803c7205-ea17-42a9-abca-03191bb59128" />

## Modes

| Trigger | Behaviour | Default model |
|---|---|---|
| `c <prompt>` | Two result rows: **fast** (default) and **deep**. Enter on either sends async; Alt picks the output channel. | Haiku 4.5 (fast) / Opus 4.7 (deep) |
| `c <prompt>?` | **Fact mode** — terse one-line answer rendered live in the result list. | Haiku 4.5 |
| `c <prompt>!` | **Command mode** — shell or code snippet only, no prose. Default action: paste at cursor. | Haiku 4.5 |

`?` and `!` fire on typing (with a ~300ms debounce) so the answer appears as you type. `c <prompt>` waits for **Enter** so deep-model calls are deliberate.

### Examples for `!` (command mode)

```
c find files modified today!     → find . -mtime -1
c regex for emails!              → ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
c kill process on port 8080!     → fuser -k 8080/tcp
c base64 encode "hello"!         → echo -n "hello" | base64
c rsync --delete syntax!         → rsync -av --delete src/ dst/
```

The system prompt for `!` strips all prose and code fences — you get just the command/snippet, ready to paste at the cursor.

## Output channels (Alt menu on `c <prompt>`)

- **Send + copy + notify** — default. Copies the response and shows a notification.
- **Send + paste at cursor** — uses `setClipboardTextAndPaste()` if supported. Great for VS Code / terminal flows.
- **Send + open in markdown viewer** — writes `/tmp/claude-*.md` and runs `xdg-open` (your preferred markdown handler).
- **Send + open in browser** — writes `/tmp/claude-*.html` rendered with code blocks and serves via `xdg-open`.
- **Send + add to CopyQ history** — keeps your current clipboard intact and stashes the response in CopyQ. Falls back to clipboard if `copyq` isn't installed.
- **Send + notification only** — fire and forget; no clipboard pollution.

`?` and `!` modes have a smaller subset of these on Alt, with the most ergonomic default per mode (copy for `?`, paste for `!`).

## Installation

```bash
pip install --user anthropic   # or use a venv: see Albert docs
git clone https://github.com/Bierchermuesli/albert-claude.git \
  ~/.local/share/albert/python/plugins/claude
```

Then enable the plugin in Albert's Python Modules settings.

## Configuration

- **API Key** — paste your `sk-ant-...` key, or set the `ANTHROPIC_API_KEY` environment variable. Get a key at <https://console.anthropic.com/settings/keys>.
- **Chat model** — used for `c <prompt>`. Default: `claude-opus-5-5`.
- **Quick model** — used for `?` and `!`. Default: `claude-haiku-5-5`.
- **System prompt** — override the default chat-mode system prompt. Leave empty to use the bundled launcher-tuned prompt.

## Caching

- Identical prompts (same mode + model) are cached in-memory for **30 minutes** so navigating back to a recent answer is instant.
- The system prompt is annotated for Anthropic prompt caching (`cache_control: ephemeral`); it only kicks in if the prefix exceeds the model's minimum cacheable size.

## Cost notes

- `?` and `!` fire on typing — they use Haiku (cheapest) and a 300ms typing debounce to limit waste.
- Bare `c <prompt>` only fires on Enter, so Opus calls are deliberate.

## Bugs / Feedback

Always welcome.
