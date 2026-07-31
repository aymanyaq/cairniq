"""Syntax-check the inline <script> blocks in every Jinja template.

Why this exists: an inline script is all-or-nothing. A single bad character
anywhere in the block means the browser discards the WHOLE block — every
function it defines and every listener it registers. On 2026-07-22 an
unescaped apostrophe in a quick-action prompt string ("each tranche's size")
inside templates/index.html killed the entire ~1100-line sidebar script, so
loadNews/loadPulse/loadPriority/loadCatalysts were never defined and every
right-rail panel sat on its static "Loading ..." placeholder forever. The
server was healthy and the daily cache was populated the whole time — nothing
server-side could have caught it, and no Python test touched it.

The check is a real JS parse (node --check), not a regex: the templates are
full of multi-line template literals with nested ${...} expressions, which
no line-oriented heuristic can read correctly.

Jinja is neutralized before parsing, in BOTH directions — once with every
{% if %} body kept and once with every {% if %}...{% endif %} block removed —
so a conditional block cannot hide a parse error in the branch that happens
not to render for the current config (e.g. `enable_guru_picks`).
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# Inline scripts only — <script src="..."> loads a real .js file, which is not
# rendered through Jinja and is linted on its own terms.
_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", re.S | re.I)
_TYPE_RE = re.compile(r'\btype\s*=\s*["\']([^"\']+)["\']', re.I)
_EXPR_RE = re.compile(r"\{\{.*?\}\}", re.S)
_TAG_RE = re.compile(r"\{%.*?%\}", re.S)
_IF_BLOCK_RE = re.compile(r"\{%-?\s*if\b.*?%\}.*?\{%-?\s*endif\s*-?%\}", re.S)

_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}

node = shutil.which("node")
pytestmark = pytest.mark.skipif(
    node is None,
    reason="node is required to parse inline template scripts (present on CI runners)",
)


def _inline_scripts(html: str):
    """Yield (line_offset, js_source) for each inline JS <script> block."""
    for m in _SCRIPT_RE.finditer(html):
        attrs, body = m.group(1), m.group(2)
        type_match = _TYPE_RE.search(attrs)
        if (type_match.group(1).strip().lower() if type_match else "") not in _JS_TYPES:
            continue  # e.g. application/json data islands are not JS
        if body.strip():
            yield html[: m.start(2)].count("\n") + 1, body


def _neutralize(js: str, drop_conditionals: bool) -> str:
    """Replace Jinja with syntactically inert JS.

    `{{ expr }}` becomes the literal 0 — valid wherever an expression or a
    string fragment is, which is everywhere these templates use it.
    """
    js = _EXPR_RE.sub("0", js)
    if drop_conditionals:
        js = _IF_BLOCK_RE.sub("", js)
    return _TAG_RE.sub("", js)


def _first_error(js: str) -> str | None:
    """Return node's parse error for `js`, or None if it parses."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        result = subprocess.run([node, "--check", path], capture_output=True, text=True)
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode == 0:
        return None
    # node reports "<tmpfile>:<line>" — swap in the template-relative line below.
    return result.stderr.replace(path, "<inline script>").strip()


@pytest.mark.parametrize(
    "template", sorted(TEMPLATE_DIR.rglob("*.html")), ids=lambda p: p.name
)
def test_inline_scripts_parse(template: Path):
    """Every inline <script> in every template must be valid JavaScript.

    A failure here means the browser will silently drop the entire block —
    the page renders but nothing in it works.
    """
    html = template.read_text(encoding="utf-8")
    for offset, body in _inline_scripts(html):
        for drop in (False, True):
            error = _first_error(_neutralize(body, drop_conditionals=drop))
            if error is None:
                continue
            branch = "with {% if %} bodies removed" if drop else "as written"
            line_match = re.search(r"<inline script>:(\d+)", error)
            where = (
                f"{template.name}:{offset + int(line_match.group(1)) - 1}"
                if line_match
                else template.name
            )
            pytest.fail(
                f"Inline <script> in {where} does not parse ({branch}).\n"
                f"The browser discards the whole block, so nothing it defines runs.\n\n{error}"
            )


def test_index_quick_action_prompts_are_terminated():
    """Regression guard for the 2026-07-22 outage specifically.

    The quick-action prompt strings in index.html are long, single-quoted, and
    routinely edited as prose (they are LLM prompts). An apostrophe typed
    straight into one of them closes the literal early. Each entry must sit on
    one line and end with a closing quote.
    """
    html = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    entries = re.findall(r"^\s*'btn-[\w-]+':\s*(.*)$", html, re.M)
    assert entries, "quick-action prompt map not found in index.html"
    for value in entries:
        assert re.fullmatch(r"'(?:[^'\\]|\\.)*',?", value.strip()), (
            "Unescaped apostrophe in a quick-action prompt — this breaks the whole "
            f"inline script:\n{value.strip()[:200]}"
        )
