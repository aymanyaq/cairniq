"""Keep every version label the app shows pinned to the actual release.

Why this exists: the version was typed out by hand in four independent places —
`server.APP_VERSION` (the `/api/health` payload the iOS client reads), the
sidebar in base.html, the Settings footer, and the README badge. All four still
said 2.2.0 long after v2.4.0 was tagged and released, and nothing anywhere could
notice: a stale string is valid Python, valid Jinja, and valid Markdown.

So the number now lives once, in `agent/version.py`, and these tests are what
keep that one copy honest. Cutting a release is `git tag vX.Y.Z` **and** bumping
that constant; skip the second half and the suite fails here.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.version import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "templates"

# A version LABEL in template prose: "Console v2.2.0", "Version 2.2.0-LTS".
# Scoped to the label wording on purpose — a bare `\d+\.\d+\.\d+` would also
# flag an unrelated vendored asset path like lib/vis-9.1.2/, which is not a
# claim about the app's version.
_TEMPLATE_VERSION_RE = re.compile(r"(?i)(?:version[\s:]*v?|\bv)\d+\.\d+\.\d+")
_README_BADGE_RE = re.compile(r"img\.shields\.io/badge/version-([0-9]+\.[0-9]+\.[0-9]+)-")


def _newest_release_tag() -> str | None:
    """Newest `vX.Y.Z` tag in this checkout, or None when git can't tell us.

    Returns None rather than failing when git is missing or the checkout has no
    tags (shallow clone, zip download, CI without `fetch-depth: 0`) — the point
    is to catch drift where the tags exist, not to demand git everywhere.
    """
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(
            ["git", "tag", "--list", "v[0-9]*", "--sort=-v:refname"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return tags[0] if tags else None


def test_version_matches_newest_release_tag():
    tag = _newest_release_tag()
    if tag is None:
        pytest.skip("no git tags available in this checkout")
    assert tag.lstrip("v") == __version__, (
        f"agent/version.py says {__version__} but the newest release tag is {tag}. "
        "Bump __version__ to match the tag (or tag the release you meant to cut) — "
        "every label in the app reads that constant."
    )


def test_readme_badge_matches_version():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    match = _README_BADGE_RE.search(readme)
    assert match, "README.md is missing its version badge"
    assert match.group(1) == __version__, (
        f"README badge says {match.group(1)}, agent/version.py says {__version__}"
    )


def test_health_endpoint_reports_the_shared_version():
    import server

    assert server.APP_VERSION == __version__


@pytest.mark.parametrize(
    "template_name", ["base.html", "terminal_settings.html"]
)
def test_version_labels_are_rendered_not_hardcoded(template_name):
    """The two templates that show a version must interpolate, never inline it.

    Restricted to the templates that actually carry a version label: a blanket
    scan would trip over unrelated semvers (a pinned dependency, an API version)
    in the other templates.
    """
    text = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    assert "app_version" in text, f"{template_name} no longer renders the version"
    stray = _TEMPLATE_VERSION_RE.findall(text)
    assert not stray, f"{template_name} hardcodes version literal(s): {stray}"
